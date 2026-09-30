"""Append-only numeric label storage without one retained ndarray per row."""
import operator
import numpy as np


class NumericRowBuffer:
    def __init__(self, block_rows=2048):
        if block_rows < 1:
            raise ValueError('block_rows must be positive')
        self.block_rows = int(block_rows)
        self.clear()

    def clear(self):
        self.blocks = []
        self.count = 0
        self.shape = None
        self.dtype = None

    def __len__(self):
        return self.count

    def _extend_array(self, rows):
        if not len(rows):
            return
        if self.shape is None:
            self.shape, self.dtype = rows.shape[1:], rows.dtype
        if rows.shape[1:] != self.shape or rows.dtype != self.dtype:
            raise ValueError('numeric label shape/dtype changed')
        offset = 0
        while offset < len(rows):
            used = self.count % self.block_rows
            if used == 0:
                self.blocks.append(np.empty((self.block_rows,*self.shape),self.dtype))
            take = min(self.block_rows-used,len(rows)-offset)
            self.blocks[-1][used:used+take] = rows[offset:offset+take]
            offset += take
            self.count += take

    def append(self, row):
        self._extend_array(np.asarray(row)[None])

    def chunks(self):
        for index, block in enumerate(self.blocks):
            yield block[:min(self.block_rows,self.count-index*self.block_rows)]

    def extend(self, rows):
        if rows is self:
            rows = np.asarray(self).copy()
        if isinstance(rows, NumericRowBuffer):
            for chunk in rows.chunks():
                self._extend_array(chunk)
        elif len(rows):
            self._extend_array(np.asarray(rows))

    def __getitem__(self, index):
        if isinstance(index,slice):
            return np.asarray(self)[index]
        index = operator.index(index)
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        return self.blocks[index//self.block_rows][index%self.block_rows]

    def __iter__(self):
        for chunk in self.chunks():
            yield from chunk

    def __array__(self, dtype=None, copy=None):
        if not self.blocks:
            return np.empty(0,dtype=dtype or float)
        if copy is False and (len(self.blocks)>1 or (dtype is not None and np.dtype(dtype)!=self.dtype)):
            raise ValueError('row buffer conversion requires a copy')
        result = self.blocks[0][:self.count] if len(self.blocks)==1 else np.concatenate(list(self.chunks()))
        if dtype is not None:
            result = result.astype(dtype,copy=False)
        return result.copy() if copy else result
