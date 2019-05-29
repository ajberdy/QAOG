"""
Quantum strings
"""

import numpy as np
from cirq import LineQubit, Moment, X, Rx


class Distribution:

    def __init__(self, dist, n):
        self._dist = dist
        self.n = n

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                item = int(item, 2)
            except ValueError:
                raise TypeError("distribution key must be number or binary string")

        if item in self._dist:
            return self._dist[item]
        return 0

    def __len__(self):
        for k, v in self._dist.items():
            if not v:
                del self._dist[k]
        return len(self._dist)

    def __repr__(self):
        return f"<Distribution>{{{', '.join(f'{k:0{self.n}b}: {v:.4f}' for k, v in self._dist.items())}}}"

    def is_singular(self):
        return len(self) == 1

    def normalized(self):
        total = sum(self._dist.values())
        for k, v in self._dist.items():
            self._dist[k] = v/total
        return self

    def __copy__(self):
        return Distribution(self._dist.copy(), self.n)

    def split_by_first_bit(self):
        zero, one = {}, {}
        for k, v in self._dist.items():
            if 2**(self.n - 1) & k:
                one[k - 2**(self.n - 1)] = v
            else:
                zero[k] = v

        lprob = sum(zero.values())
        zero_dist = Distribution(zero, self.n - 1).normalized()
        one_dist = Distribution(one, self.n - 1).normalized()

        return zero_dist, one_dist, lprob

    def sample(self):
        keys, probs = zip(*self._dist.items())
        return np.random.choice(keys, p=probs)


class QChar:

    def __init__(self, dist: Distribution):
        self.qbyte = [LineQubit(i) for i in range(8)]


class AOG:

    AND, OR, TERMINAL = range(3)

    def __init__(self, distribution: Distribution = None):

        if distribution is None:
            return

        if distribution.n == 1:
            if distribution.is_singular():
                self.type = AOG.TERMINAL
                self._data = distribution.sample()
            else:
                self.type = AOG.OR
                self._left = AOG.terminal(0)
                self._right = AOG.terminal(1)
                self._lprob = distribution[0]
        else:
            zero_child, one_child, lprob = distribution.split_by_first_bit()
            if lprob == 1:
                self.type = AOG.AND
                self._left = AOG.terminal(0)
                self._right = AOG(zero_child)
            elif lprob == 0:
                self.type = AOG.AND
                self._left = AOG.terminal(1)
                self._right = AOG(one_child)
            else:
                self.type = AOG.OR
                self._left = AOG.and_(AOG.terminal(0), AOG(zero_child))
                self._right = AOG.and_(AOG.terminal(1), AOG(one_child))
                self._lprob = lprob

    def pprint(self, depth=0):
        if self.type == AOG.TERMINAL:
            return f"T<{self._data}>"

        elif self.type == AOG.OR:
            return (f"O<({self._lprob:.2f})\n"
                    f"{depth * '| '}| {self._left.pprint(depth + 1)}\n"
                    f"{depth * '| '}| {self._right.pprint(depth + 1)}\n"
                    f"{depth * '| '}>")

        elif self.type == AOG.AND:
            return (f"A<\n"
                    f"{depth * '| '}| {self._left.pprint(depth + 1)}\n"
                    f"{depth * '| '}| {self._right.pprint(depth + 1)}\n"
                    f"{depth * '| '}>")

    @classmethod
    def terminal(cls, data):
        aog = AOG()
        aog.type = AOG.TERMINAL
        aog._data = data
        return aog

    @classmethod
    def and_(cls, left, right):
        aog = AOG()
        aog.type = AOG.AND
        aog._left = left
        aog._right = right
        return aog

    @classmethod
    def or_(cls, left, right, lprob):
        aog = AOG()
        aog.type = AOG.OR
        aog._left = left
        aog._right = right
        aog._lprob = lprob
        return aog

    def to_circuit(self):
        raise NotImplemented


if __name__ == '__main__':
    dist = {
        0b0011: 1/2,
        0b1011: 1/4,
        0b0101: 1/8,
        0b1110: 1/8
    }
    d = Distribution(dist, 4)
    print(d[3])
    print(d[0b11])
    print(d[0b0011])
    print(d["0b0011"])
    print(d["0011"])
    print(d["11"])

    print(d)
    print(d.split_by_first_bit())
    for i in range(10):
        print(d.sample())

    aog = AOG(d)
    print(aog.pprint())
    exit(0)
    print(AOG.terminal(0).pprint())
    print(AOG.terminal(1).pprint())
    print(AOG.and_(AOG.terminal(1), AOG.terminal(0)).pprint())
    print(AOG.or_(
        AOG.and_(AOG.terminal(1), AOG.terminal(0)),
        AOG.and_(AOG.terminal(1), AOG.terminal(1)), .5).pprint())
    print(AOG.and_(
        AOG.terminal(1),
        AOG.or_(AOG.terminal(0), AOG.terminal(1), .5)
    ).pprint())
