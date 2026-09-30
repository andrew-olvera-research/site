"""Incrementally build the project rich batch binding inside the Starscream image.

Do not run upstream setup.py: it removes reusable native dependency/build caches.
Installation is explicit, atomic, and keeps a content-addressed backup. Stop all
collectors before installing; already-running interpreters retain their old module.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if sys.platform != 'linux' or not 1 <= args.jobs <= 16:
        raise ValueError('run inside the Linux Starscream image with jobs1..16')
    root = Path(__file__).resolve().parents[2]
    source = root/'cpp/pybind_wrapper.cpp'
    flightlib = Path('/opt/flightmare/flightlib')
    build_dirs = list((flightlib/'build').glob('temp.*'))
    build_dirs = [p for p in build_dirs if (p/'CMakeCache.txt').is_file()]
    if len(build_dirs) != 1:
        raise RuntimeError('expected one configured Flightmare CMake build; build the Docker image first')
    shutil.copy2(source,flightlib/'src/wrapper/pybind_wrapper.cpp')
    subprocess.run(['cmake','--build',str(build_dirs[0]),'--target','flightgym','-j',str(args.jobs)],check=True)
    binaries = list((flightlib/'build').glob('lib.*/flightgym*.so'))
    if len(binaries) != 1:
        raise RuntimeError('expected one built flightgym binary')
    binary = binaries[0]
    print(f'Built: {binary}',flush=True)
    if args.install:
        spec = importlib.util.find_spec('flightgym')
        if spec is None or not spec.origin:
            raise RuntimeError('installed flightgym not found')
        target = Path(spec.origin).resolve()
        if target == binary.resolve() or 'site-packages' not in target.parts:
            raise RuntimeError('install without a PYTHONPATH override; expected existing site-packages binary')
        digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
        backup = target.with_name(target.name+'.backup-'+digest)
        if not backup.exists():
            shutil.copy2(target,backup)
        temporary = target.with_name(target.name+'.pending')
        shutil.copy2(binary,temporary)
        temporary.replace(target)  # Never overwrite a mapped inode in place.
        print(f'Installed: {target}\nPrevious binary retained: {backup}',flush=True)
    else:
        print(f'For qualification use PYTHONPATH={binary.parent}:{root}',flush=True)


if __name__ == '__main__':
    main()
