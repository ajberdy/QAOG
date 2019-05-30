"""
Quantum strings
"""
import math
from collections import defaultdict
from itertools import product

import numpy as np
from cirq import LineQubit, Moment, X, Rx, ControlledGate, Circuit, Simulator, measure


def bin_string_iter(n):
    """ generates all binary strings of length n in order """
    for bin_string in product([0, 1], repeat=n):
        yield list(bin_string)


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


class AOG:

    AND, OR, TERMINAL = range(3)

    def __init__(self, distribution: Distribution = None):

        if distribution is None:
            return

        self.n = distribution.n

        if distribution.n == 1:
            if distribution.is_singular():
                self.type = AOG.TERMINAL
                self._data = distribution.sample()
                self.depth = 0
            else:
                self.type = AOG.OR
                self._left = AOG.terminal(0)
                self._right = AOG.terminal(1)
                self._lprob = distribution[0]
                self.depth = 1
        else:
            zero_child, one_child, lprob = distribution.split_by_first_bit()
            if lprob == 1:
                self.type = AOG.AND
                self._left = AOG.terminal(0)
                self._right = AOG(zero_child)
                self.depth = self._right.depth
            elif lprob == 0:
                self.type = AOG.AND
                self._left = AOG.terminal(1)
                self._right = AOG(one_child)
                self.depth = self._right.depth
            else:
                self.type = AOG.OR
                self._left = AOG.and_(AOG.terminal(0), AOG(zero_child))
                self._right = AOG.and_(AOG.terminal(1), AOG(one_child))
                self._lprob = lprob
                self.depth = max(self._left.depth, self._right.depth) + 1

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
        aog.depth = 0
        aog.n = 1
        return aog

    @classmethod
    def and_(cls, left, right):
        aog = AOG()
        aog.type = AOG.AND
        aog._left = left
        aog._right = right
        aog.depth = max(aog._left.depth, aog._right.depth)
        aog.n = aog._left.n + aog._right.n
        return aog

    @classmethod
    def or_(cls, left, right, lprob):
        aog = AOG()
        aog.type = AOG.OR
        aog._left = left
        aog._right = right
        aog._lprob = lprob
        aog.depth = max(aog._left.depth, aog._right.depth) + 1
        aog.n = aog._left.n
        return aog

    @property
    def _theta(self):
        return 2*math.acos(math.sqrt(self._lprob))

    def to_circuit(self, history=None, ix=0, n=None, minterms=None, depth=None):

        if n is None:
            n = self.n

        if history is None:
            history = []

        if minterms is None:
            minterms = defaultdict(lambda: [])

        if depth is None:
            depth = self.depth

        c_on, c_off = split_control(history, n)
        qix = len(history)

        if self.type == AOG.TERMINAL:
            circuit = Circuit()
            if self._data:
                for cont in bin_string_iter(depth - len(history)):
                    aug_history = history + cont
                    aug_c_on, aug_c_off = split_control(aug_history, n)
                    circuit.append(controlled_circuit(X, [LineQubit(ix)], aug_c_on, aug_c_off))
            return circuit, history, 1

        elif self.type == AOG.OR:
            or_gate = Rx(self._theta)
            circuit = controlled_circuit(or_gate, [LineQubit(qix + n)], c_on, c_off)
            left_circuit, _, left_len = self._left.to_circuit(history + [0], ix, n, minterms, depth)
            right_circuit, _, right_len = self._right.to_circuit(history + [1], ix, n, minterms, depth)

            assert left_len == right_len    # debugging
            circuit.append(left_circuit)
            circuit.append(right_circuit)
            return circuit, history + [-1], left_len

        elif self.type == AOG.AND:
            left_circuit, history_after_left, left_len = self._left.to_circuit(history, ix, n, minterms, depth)
            right_circuit, history_after_right, right_len = \
                self._right.to_circuit(history_after_left, ix + left_len, n, minterms, depth)
            left_circuit.append(right_circuit)
            return left_circuit, history_after_right, left_len + right_len


def split_control(control_qubits, n=0):
    control_on = [LineQubit(i) for i, q in enumerate(control_qubits, start=n) if q == 1]
    control_off = [LineQubit(i) for i, q in enumerate(control_qubits, start=n) if q == 0]
    return control_on, control_off


def controlled_circuit(gate, qubits, on_qubits, off_qubits):
    flip_offs = [X(i) for i in off_qubits]
    control_qubits = on_qubits + off_qubits
    c_gate = gate.controlled_by(*control_qubits)
    return Circuit.from_ops([*flip_offs, c_gate(*qubits), *flip_offs])


if __name__ == '__main__':
    dist = {
        0b0011: 1/2,
        0b1011: 1/4,
        0b0101: 1/8,
        0b1110: 1/8
    }
    d = Distribution(dist, 4)
    aog = AOG(d)
    aog_circuit, _, _ = aog.to_circuit()
    print(aog_circuit)

    simulator = Simulator()
    result = simulator.simulate(aog_circuit)
    print(result)
