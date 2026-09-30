"""Crash-consistent append-only replay storage for long DAgger runs.

Each completed collection round is written once as an atomic HDF5 shard.  A
checkpoint records the ordered shard roots and the last committed round.  A
resume reconstructs the exact bounded in-memory replay by reading newest
shards backwards until each pool's configured capacity is satisfied.

This deliberately keeps replay out of PyTorch checkpoints: an 18x103 fp16
history reservoir can be several GiB, while a round shard is also useful as an
auditable offline trajectory archive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import h5py
import numpy as np


REPLAY_SCHEMA = "starscream-dagger-replay-shards-v1"


def replay_contract_fingerprint(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(contract), sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_pool(arrays: Mapping[str, np.ndarray]) -> int:
    if any(np.asarray(value).ndim == 0 for value in arrays.values()):
        raise ValueError('DAgger replay arrays require a row dimension')
    lengths = {len(np.asarray(value)) for value in arrays.values()}
    if not lengths:
        raise ValueError("DAgger replay pool must contain at least one array")
    if len(lengths) != 1:
        raise ValueError(f"DAgger replay arrays lost alignment: {sorted(lengths)}")
    count = lengths.pop()
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype == object:
            raise ValueError(f"DAgger replay array {name!r} cannot use object dtype")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f'DAgger replay array {name!r} contains nonfinite values')
    return int(count)


def _write_pool(group: h5py.Group, arrays: Mapping[str, np.ndarray]) -> int:
    count = _validate_pool(arrays)
    group.attrs["rows"] = count
    for name, raw in arrays.items():
        value = np.asarray(raw)
        options: dict[str, Any] = {}
        if value.size and value.ndim and all(size > 0 for size in value.shape[1:]):
            options = {"compression": "lzf", "shuffle": True, "chunks": True}
        group.create_dataset(name, data=value, **options)
    return count


@dataclass
class DaggerReplayStore:
    """Own one run's replay shards and an optional chain of parent stores."""

    root: Path
    contract: Mapping[str, Any]
    sources: list[dict[str, Any]]

    @classmethod
    def create(
        cls,
        root: str | Path,
        contract: Mapping[str, Any],
        *,
        resume_metadata: Mapping[str, Any] | None = None,
        require_resume_state: bool = False,
    ) -> "DaggerReplayStore":
        root = Path(root).resolve()
        fingerprint = replay_contract_fingerprint(contract)
        sources: list[dict[str, Any]] = []
        if resume_metadata:
            if str(resume_metadata.get("schema")) != REPLAY_SCHEMA:
                raise ValueError("unsupported DAgger replay checkpoint schema")
            if str(resume_metadata.get("contract_fingerprint")) != fingerprint:
                raise ValueError("DAgger replay contract differs from resume checkpoint")
            sources = [dict(item) for item in resume_metadata.get("sources", ())]
            if not sources:
                raise ValueError("DAgger replay checkpoint has no shard sources")
            next_minimum = int(resume_metadata.get("committed_round", 0)) + 1
        elif require_resume_state:
            raise ValueError(
                "resume checkpoint has no persistent DAgger replay state; "
                "disable dagger_resume_require_replay_state only for legacy recovery"
            )
        else:
            next_minimum = 1
        resolved = str(root)
        if not sources or str(Path(str(sources[-1]["root"])).resolve()) != resolved:
            sources.append({
                "root": resolved,
                "min_round": next_minimum,
                "max_round": next_minimum - 1,
            })
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root, contract=dict(contract), sources=sources)

    @property
    def fingerprint(self) -> str:
        return replay_contract_fingerprint(self.contract)

    def write_round(
        self,
        round_index: int,
        *,
        online: Mapping[str, np.ndarray],
        permanent: Mapping[str, np.ndarray],
    ) -> Path:
        round_index = int(round_index)
        if round_index < 1:
            raise ValueError("DAgger replay shards require a positive round")
        _validate_pool(online)
        _validate_pool(permanent)
        destination = self.root / f"round-{round_index:05d}.h5"
        if destination.exists():
            committed = int(self.sources[-1].get("max_round", 0))
            if round_index <= committed:
                with h5py.File(destination, "r") as existing:
                    if (
                        str(existing.attrs.get("schema", "")) != REPLAY_SCHEMA
                        or str(existing.attrs.get("contract_fingerprint", ""))
                        != self.fingerprint
                        or int(existing.attrs.get("round", -1)) != round_index
                    ):
                        raise ValueError(
                            f"incompatible committed replay shard: {destination}"
                        )
                    for pool, arrays in [('online', online), ('permanent', permanent)]:
                        group = existing[pool]
                        if set(group) != set(arrays) or any(
                            group[name].shape != np.asarray(value).shape
                            or group[name].dtype != np.asarray(value).dtype
                            or not np.array_equal(group[name][:], value)
                            for name, value in arrays.items()
                        ):
                            raise ValueError(f'cannot change committed replay shard: {destination}')
                return destination
            # A process can die after atomically publishing a shard but before
            # committing the round in its checkpoint.  That shard is not part
            # of the resume transaction and must be replaced by the newly
            # collected round, otherwise memory and disk replay can diverge.
            destination.unlink()
        handle, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.root,
        )
        os.close(handle)
        temporary_path = Path(temporary)
        try:
            with h5py.File(temporary_path, "w") as archive:
                archive.attrs["schema"] = REPLAY_SCHEMA
                archive.attrs["round"] = round_index
                archive.attrs["contract_fingerprint"] = self.fingerprint
                archive.attrs["contract_json"] = json.dumps(
                    dict(self.contract), sort_keys=True, default=str,
                )
                _write_pool(archive.create_group("online"), online)
                _write_pool(archive.create_group("permanent"), permanent)
                archive.flush()
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        return destination

    def checkpoint_metadata(self, committed_round: int) -> dict[str, Any]:
        committed_round = int(committed_round)
        result = [dict(item) for item in self.sources]
        local_rounds = [
            int(path.stem.split("-")[-1])
            for path in self.root.glob("round-*.h5")
            if int(path.stem.split("-")[-1]) <= committed_round
        ]
        if local_rounds:
            result[-1]["max_round"] = max(
                int(result[-1].get("max_round", 0)), max(local_rounds),
            )
        self.sources = result
        return {
            "schema": REPLAY_SCHEMA,
            "contract": dict(self.contract),
            "contract_fingerprint": self.fingerprint,
            "committed_round": committed_round,
            "sources": result,
        }

    def _committed_shards(self, committed_round: int) -> list[Path]:
        shards: list[Path] = []
        previous_maximum = 0
        for source_index, source in enumerate(self.sources):
            root = Path(str(source["root"]))
            maximum = min(int(source.get("max_round", 0)), int(committed_round))
            minimum = int(source.get("min_round", previous_maximum + 1))
            previous_maximum = max(previous_maximum, maximum)
            if maximum < minimum:
                continue
            if not root.is_dir():
                raise FileNotFoundError(f"DAgger replay shard root is missing: {root}")
            by_round: dict[int, Path] = {}
            for path in sorted(root.glob("round-*.h5")):
                try:
                    round_index = int(path.stem.split("-")[-1])
                except ValueError:
                    continue
                if minimum <= round_index <= maximum:
                    by_round[round_index] = path
            missing = set(range(minimum, maximum + 1)) - set(by_round)
            if missing:
                raise FileNotFoundError(
                    f"DAgger replay source {source_index} is missing committed "
                    f"rounds: {sorted(missing)}"
                )
            shards.extend(by_round[index] for index in range(minimum, maximum + 1))
        return shards

    def restore_pool(
        self,
        pool: str,
        templates: Mapping[str, np.ndarray],
        *,
        capacity: int,
        committed_round: int,
    ) -> dict[str, np.ndarray]:
        if pool not in {"online", "permanent"}:
            raise ValueError("DAgger replay pool must be online or permanent")
        if capacity < 1:
            raise ValueError("DAgger replay capacity must be positive")
        remaining = int(capacity)
        selected = []
        for path in reversed(self._committed_shards(committed_round)):
            if remaining <= 0:
                break
            with h5py.File(path, "r") as archive:
                if (str(archive.attrs.get('schema', '')) != REPLAY_SCHEMA
                        or int(archive.attrs.get('round', -1)) != int(path.stem.split('-')[-1])):
                    raise ValueError(f'DAgger replay shard identity mismatch: {path}')
                if str(archive.attrs.get("contract_fingerprint", "")) != self.fingerprint:
                    raise ValueError(f"DAgger replay shard contract mismatch: {path}")
                group = archive[pool]
                rows = int(group.attrs.get("rows", 0))
                if rows < 0 or any(dataset.ndim == 0 or len(dataset) != rows for dataset in group.values()):
                    raise ValueError(f'DAgger replay shard row alignment mismatch: {path}/{pool}')
                take = min(rows, remaining)
                if take <= 0:
                    continue
                selected.append((path, rows, take))
                remaining -= take
        if not selected:
            return {name: np.asarray(template).copy() for name, template in templates.items()}
        # Allocate each final array once. The old implementation retained every
        # shard array while concatenating a second full replay pool on resume.
        count = capacity - remaining
        restored = {name: np.empty((count, *np.asarray(template).shape[1:]),
                                  dtype=np.asarray(template).dtype)
                    for name, template in templates.items()}
        offset = 0
        for path, rows, take in reversed(selected):
            with h5py.File(path, 'r') as archive:
                group = archive[pool]
                for name, target in restored.items():
                    if name not in group:
                        if name in {'histories', 'actions', 'previous', 'dynamics', 'dynamics_valid', 'tracks'}:
                            raise ValueError(f'DAgger replay missing required array: {path}/{pool}/{name}')
                        target[offset:offset+take] = -1 if name == "trajectory" else 0
                        continue
                    source = group[name]
                    if source.shape != (rows, *target.shape[1:]):
                        raise ValueError(f'DAgger replay shape mismatch for {pool}/{name}: '
                                         f'expected {(rows, *target.shape[1:])}, received {source.shape}')
                    if target[offset:offset+take].size:
                        source.read_direct(target, source_sel=np.s_[rows-take:rows],
                                           dest_sel=np.s_[offset:offset+take])
            offset += take
        _validate_pool(restored)
        return restored
