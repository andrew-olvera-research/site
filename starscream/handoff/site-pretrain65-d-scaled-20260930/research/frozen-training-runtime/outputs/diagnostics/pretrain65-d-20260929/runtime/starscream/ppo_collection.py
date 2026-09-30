"""Transport/packing only: no changes to simulator steps or policy math."""
import numpy as np
import pickle
import struct


class PackedPPOPipe:
    """Fixed numeric hot-path packets; pickle only reset/terminal diagnostics.

    Same synchronous request/response protocol, no stale observations or policy
    versions. Reward retains float64; features/actions retain float32. Unlike
    shared-memory slots, received arrays own their storage and cannot be raced.
    """
    _step_header = struct.Struct('!d?I')
    _batch_header = struct.Struct('!II')
    _lane = struct.Struct('!I')

    def __init__(self, connection):
        self.connection = connection

    def send(self, message):
        if isinstance(message, list) and message and all(
            item[1][0] == 'advance' and np.asarray(item[1][1]).dtype == np.float32
            and np.asarray(item[1][1]).shape == (4,) for item in message
        ):
            self.connection.send_bytes(b'a' + self._lane.pack(len(message)) + b''.join(
                self._lane.pack(lane) + action.tobytes() for lane, (_, action) in message))
            return
        if isinstance(message, tuple) and message[0] == 'batch' and message[1]:
            rows = message[1]
            first = rows[0][1]
            if first[0] == 'step' and first[5] is None:
                count = len(first[2])
                if all(row[0] == 'step' and row[5] is None and
                       np.asarray(row[2]).dtype == np.float32 and np.asarray(row[2]).shape == (count,) and
                       np.asarray(row[3]).dtype == np.float32 and np.asarray(row[3]).shape == (4,)
                       for _, row in rows):
                    self.connection.send_bytes(b's' + self._batch_header.pack(len(rows), count) + b''.join(
                        self._lane.pack(lane) + self._step_header.pack(row[1], row[4], count) +
                        row[2].tobytes() + row[3].tobytes() for lane, row in rows))
                    return
        if message[0] == 'advance':
            action = np.asarray(message[1])
            if action.dtype == np.float32 and action.shape == (4,):
                self.connection.send_bytes(b'A' + action.tobytes())
                return
        elif message[0] == 'step' and message[5] is None:
            _, reward, feature, context, done, _ = message
            feature, context = np.asarray(feature), np.asarray(context)
            if (feature.dtype == context.dtype == np.float32
                    and feature.ndim == 1 and context.shape == (4,)):
                self.connection.send_bytes(b'S' + self._step_header.pack(
                    reward, done, len(feature)) + feature.tobytes() + context.tobytes())
                return
        self.connection.send_bytes(b'P' + pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))

    def recv(self):
        packet = self.connection.recv_bytes()
        tag = packet[:1]
        if tag == b'a':
            count = self._lane.unpack_from(packet, 1)[0]
            if len(packet) != 5 + count * 20:
                raise ValueError('invalid packed PPO action batch')
            return [(self._lane.unpack_from(packet, 5+i*20)[0], ('advance',
                     np.frombuffer(packet, dtype=np.float32, count=4, offset=9+i*20).copy()))
                    for i in range(count)]
        if tag == b's':
            rows, features = self._batch_header.unpack_from(packet, 1)
            width = 4 + self._step_header.size + 4*(features+4)
            if len(packet) != 9 + rows*width:
                raise ValueError('invalid packed PPO step batch')
            out = []
            for i in range(rows):
                offset = 9+i*width
                lane = self._lane.unpack_from(packet, offset)[0]
                reward, done, count = self._step_header.unpack_from(packet, offset+4)
                if count != features:
                    raise ValueError('inconsistent packed PPO feature widths')
                data = np.frombuffer(packet, dtype=np.float32, count=features+4,
                                     offset=offset+4+self._step_header.size).copy()
                out.append((lane, ('step', reward, data[:features], data[features:], done, None)))
            return ('batch', out)
        if tag == b'A':
            if len(packet) != 17:
                raise ValueError('invalid packed PPO action')
            return ('advance', np.frombuffer(packet, dtype=np.float32, offset=1).copy())
        if tag == b'S':
            reward, done, count = self._step_header.unpack_from(packet, 1)
            offset = 1 + self._step_header.size
            if len(packet) != offset + 4 * (count + 4):
                raise ValueError('invalid packed PPO step')
            data = np.frombuffer(packet, dtype=np.float32, offset=offset).copy()
            return ('step', reward, data[:count], data[count:], done, None)
        if tag != b'P':
            raise ValueError('invalid packed PPO packet')
        return pickle.loads(packet[1:])

    def close(self):
        self.connection.close()


class MultiplexPipe:
    """Batch messages for independent cooperative simulator lanes in one process."""
    def __init__(self, connection, lanes):
        self.connection=connection;self.lanes=lanes
        self.pending={};self.responses={};self.closed=set()

    def lane(self,index):return LanePipe(self,index)

    def flush(self):
        if self.pending:
            self.connection.send(list(self.pending.items()))
            self.pending.clear()

    def receive(self,index):
        if index not in self.responses:
            self.flush()
            kind, values=self.connection.recv()
            if kind=='error':raise RuntimeError(values)
            if kind!='batch':raise RuntimeError('invalid simulator batch response')
            self.responses.update(values)
        return self.responses.pop(index)


class LanePipe:
    def __init__(self, owner,index):self.owner=owner;self.index=index
    def send(self,request):
        if self.index in self.owner.pending:raise RuntimeError('unconsumed lane request')
        self.owner.pending[self.index]=request
    def recv(self):return self.owner.receive(self.index)
    def flush(self):self.owner.flush()
    def close(self):
        self.owner.closed.add(self.index)
        if len(self.owner.closed)==self.owner.lanes:self.owner.connection.close()


def flush_lanes(slots):
    """Dispatch every worker group before waiting on any group's response.

    Lazy flush-on-recv serializes all process groups, defeating parallelism.
    Also handles a partially active final group without waiting for dead lanes.
    """
    for slot in slots:
        if isinstance(slot.connection, LanePipe):
            slot.connection.flush()


FLOAT_KEYS=('history','critic_input','action','advantage','return','task_delta',
    'speed_command','track_weight','rollout_laps','failure_return','dynamics_valid',
    'raw_action','old_location','old_log_std','old_log_prob','exploration_offset')


def pack_episode(records):
    """Release per-transition Python objects when each episode finishes.

    Preserve FP32 behavior samples and likelihoods exactly; never FP16-quantize
    PPO data. Only the representation of stored data changes.
    """
    out={k:np.asarray([r[k] for r in records],dtype=np.float32) for k in FLOAT_KEYS}
    out['track_id']=np.asarray([r['track_id'] for r in records],dtype=np.int64)
    return out


def join_episodes(episodes):
    if not episodes:raise ValueError('no PPO episodes')
    return {k:np.concatenate([e[k] for e in episodes],axis=0) for k in episodes[0]}
