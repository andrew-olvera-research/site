"""Typed DAgger hot-path transport, retaining the full worker response contract.

No shared mutable replay views: a received packet owns its numeric storage.
Control/reset/error messages and unknown schemas use the reference pickle path.
One message in flight per worker, same scheduling and teacher/actor decisions.
"""
import pickle
import struct
import numpy as np


class PackedDaggerPipe:
    BOOLS = (1, 5, 7, 8, 10, 14, 17, 21, 22, 23, 31, 32)
    INTS = (11, 13, 15, 25, 26)
    FLOATS = (12, 19, 20, 24, 27, 28, 30, 33)
    ARRAYS = (2, 3, 4, 6, 16, 18, 29, 34, 36, 37)
    HEADER = struct.Struct('<12?5q8d20i2I')
    ACTION = struct.Struct('<4f?')

    def __init__(self, connection):
        self.connection = connection
        self.sent_bytes = 0
        self.received_bytes = 0

    def _send_packet(self, packet):
        self.connection.send_bytes(packet)
        self.sent_bytes += len(packet)

    def fileno(self):
        return self.connection.fileno()

    def poll(self, timeout=0.):
        return self.connection.poll(timeout)

    def close(self):
        self.connection.close()

    def send(self, row):
        if row[0] == 'advance' and len(row) == 3:
            action = np.asarray(row[1])
            if action.dtype == np.float32 and action.shape == (4,):
                self._send_packet(b'A'+self.ACTION.pack(*action, bool(row[2])))
                return
        elif row[0] == 'step' and len(row) == 38:
            arrays = [row[i] for i in self.ARRAYS]
            if all(x is None or (isinstance(x, np.ndarray) and x.dtype == np.float32
                                 and x.ndim in (1, 2)) for x in arrays):
                shapes = [( -1, -1) if x is None else (len(x), -1 if x.ndim == 1 else x.shape[1])
                          for x in arrays]
                track, mode = row[9].encode('utf8'), row[35].encode('utf8')
                header = self.HEADER.pack(*(bool(row[i]) for i in self.BOOLS),
                    *(row[i] for i in self.INTS), *(row[i] for i in self.FLOATS),
                    *(v for shape in shapes for v in shape), len(track), len(mode))
                self._send_packet(b'T'+header+b''.join(x.tobytes() for x in arrays if x is not None)+track+mode)
                return
        self._send_packet(b'P'+pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL))

    def recv(self):
        packet = self.connection.recv_bytes()
        self.received_bytes += len(packet)
        tag = packet[:1]
        if tag == b'A':
            if len(packet) != 1+self.ACTION.size:
                raise ValueError('invalid packed DAgger action length')
            values = self.ACTION.unpack_from(packet, 1)
            return ('advance', np.asarray(values[:4], np.float32), values[4])
        if tag == b'T':
            if len(packet) < 1+self.HEADER.size:
                raise ValueError('truncated packed DAgger header')
            values = self.HEADER.unpack_from(packet, 1)
            shapes = list(zip(values[25:45:2], values[26:45:2]))
            if any(n < -1 or m < -1 or (n == -1 and m != -1) for n,m in shapes):
                raise ValueError('invalid packed DAgger array shape')
            counts = [0 if n == -1 else n*(1 if m == -1 else m) for n,m in shapes]
            offset = 1+self.HEADER.size
            total = sum(counts)
            if len(packet) != offset+4*total+sum(values[45:47]):
                raise ValueError('invalid packed DAgger payload length')
            data = np.frombuffer(packet, dtype='<f4', count=total, offset=offset).copy()
            row = [None]*38
            row[0] = 'step'
            for i,v in zip(self.BOOLS+self.INTS+self.FLOATS, values[:25]):
                row[i] = v
            cursor = 0
            for i,(n,m),count in zip(self.ARRAYS, shapes, counts):
                if n != -1:
                    row[i] = data[cursor:cursor+count].reshape((n,) if m == -1 else (n,m))
                cursor += count
            offset += 4*total
            row[9] = packet[offset:offset+values[45]].decode('utf8')
            row[35] = packet[offset+values[45]:].decode('utf8')
            return tuple(row)
        if tag != b'P':
            raise ValueError('invalid packed DAgger packet tag')
        return pickle.loads(packet[1:])
