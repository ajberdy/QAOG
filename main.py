#!/usr/bin/env python3.6
"""
main.py [-h]

Quantum AOG
"""

import logging
import argparse
import math

import numpy as np
from pyquil import get_qc
from pyquil.api import WavefunctionSimulator

from pyquil.quil import Program, DefGate, merge_programs
from pyquil.gates import H, X, RX

PI = math.pi
SQRT2 = math.sqrt(2)

TERMINAL, OR, AND = range(3)


class TerminalNode:
    def __init__(self, value):
        self.type = TERMINAL
        self.value = value

    def __repr__(self):
        return f"T<{self.value}>"

    def pprint(self, _):
        return f"T<{self.value}>"


class OrNode:
    def __init__(self, left, right, theta):
        self.type = OR
        self.left = left
        self.right = right
        self.theta = theta

    def __repr__(self):
        return f"O<{self.left}|{self.right}>"

    def pprint(self, depth):
        return (f"O<\n"
                f"{depth*'  '}  {self.left.pprint(depth+1)}\n"
                f"{depth*'  '}  {self.right.pprint(depth+1)}\n"
                f"{depth*'  '}>")


class AndNode:
    def __init__(self, left, right):
        self.type = AND
        self.left = left
        self.right = right

    def __repr__(self):
        return f"A<{self.left},{self.right}>"

    def pprint(self, depth):
        return (f"A<\n"
                f"{depth*'  '}  {self.left.pprint(depth+1)}\n"
                f"{depth*'  '}  {self.right.pprint(depth+1)}\n"
                f"{depth*'  '}>")


def get_theta(left_prob):
    # return math.acos(1 - 2*left_prob)
    return 2*math.acos(math.sqrt(left_prob))


def toy_aog():
    hamster = TerminalNode("Hamster ")
    spring = TerminalNode("Spring ")
    winter = TerminalNode("Winter ")

    is_now = TerminalNode("is now ")
    was = TerminalNode("was ")

    coming = TerminalNode("coming.")
    leaving = TerminalNode("leaving.")
    warmer = TerminalNode("warmer.")
    colder = TerminalNode("colder.")
    jumping = TerminalNode("jumping.")

    is_now_or_was = OrNode(is_now, was, PI/2)
    coming_or_warmer = OrNode(coming, warmer, PI/2)
    leaving_or_colder = OrNode(leaving, colder, PI/2)

    coming_or_leaving = OrNode(coming, leaving, PI/2)
    coming_or_leaving_or_jumping = OrNode(coming_or_leaving, jumping, get_theta(2/3))

    spring_and_linker = AndNode(spring, is_now_or_was)
    winter_and_linker = AndNode(winter, is_now_or_was)
    hamster_and_linker = AndNode(hamster, is_now_or_was)

    spring_sentence = AndNode(spring_and_linker, coming_or_warmer)
    winter_sentence = AndNode(winter_and_linker, leaving_or_colder)
    hamster_sentence = AndNode(hamster_and_linker, coming_or_leaving_or_jumping)

    weather_sentence = OrNode(spring_sentence, winter_sentence, PI/2)
    weather_or_hamster_sentence = OrNode(weather_sentence, hamster_sentence, get_theta(.58))

    return weather_or_hamster_sentence


def coin_flip_aog():
    heads = TerminalNode("H")
    tails = TerminalNode("T")
    flip = OrNode(heads, tails, get_theta(.5))
    return flip


def parse_aog(qubits, node):
    if node.type == TERMINAL:
        return node.value, qubits

    elif node.type == AND:
        left_parse, remaining_qubits = parse_aog(qubits, node.left)
        right_parse, final_qubits_left = parse_aog(remaining_qubits, node.right)
        return left_parse + right_parse, final_qubits_left

    elif node.type == OR:
        head_qubits, tail_qubits = qubits[0], qubits[1:]
        child_selection = node.left if head_qubits == 0 else node.right
        return parse_aog(tail_qubits, child_selection)


def parse_args():
    parser = argparse.ArgumentParser(usage=__doc__.rstrip())
    args = parser.parse_args()
    return args


def controlled(program, gate, on_qubits, off_qubits):
    """ control a gate to via a substate over the outer qubits """

    if on_qubits == off_qubits == []:
        program += gate
        return program

    elif not on_qubits:
        head = off_qubits[0]
        program += X(head)
        controlled_gate = gate.controlled(head)
        program = controlled(program, controlled_gate, on_qubits, off_qubits[1:])
        program += X(head)
        return program

    else:
        head = on_qubits[0]
        controlled_gate = gate.controlled(head)
        return controlled(program, controlled_gate, on_qubits[1:], off_qubits)


def split_control(control_qubits):
    control_on = [i for i, q in enumerate(control_qubits) if q == 1]
    control_off = [i for i, q in enumerate(control_qubits) if q == 0]
    return control_on, control_off


def build_qaog(program, history, aog):
    c_on, c_off = split_control(history)
    ix = len(history)

    if aog.type == TERMINAL:
        return program, history

    elif aog.type == OR:
        or_gate = RX(aog.theta, ix)
        program = controlled(program, or_gate, c_on, c_off)
        program, _ = build_qaog(program, history + [0], aog.left)
        program, _ = build_qaog(program, history + [1], aog.right)
        return program, history + [-1]

    elif aog.type == AND:
        program, history_after_left = build_qaog(program, history, aog.left)
        program, history_after_right = build_qaog(program, history_after_left, aog.right)
        return program, history_after_right


def get_parse_dict(wavefunction, aog):
    def to_list(qstring):
        return np.array([int(q == "1") for q in reversed(qstring)])

    prob_dict = wavefunction.get_outcome_probs()
    parse_dict = {
        parse_aog(to_list(qubits), aog)[0]: prob for qubits, prob in prob_dict.items() if prob
    }
    return parse_dict


if __name__ == '__main__':
    args = parse_args()
    aog = toy_aog()
    pretty_aog = aog.pprint(depth=0)
    # print(pretty_aog)
    parsed = parse_aog([1, 0, 0, 1], aog)
    # print(parsed)

    qaog, history = build_qaog(Program(), [], aog)
    num_qubits = len(qaog.get_qubits())

    qc = get_qc(f'{num_qubits}q-qvm')
    wfn = WavefunctionSimulator().wavefunction(qaog)
    print(wfn)
    hamster_parse_dict = get_parse_dict(wfn, aog)
    for parse, prob in hamster_parse_dict.items():
        print(f"{prob:.4f} of: {parse}")

    qcaog, chistory = build_qaog(Program(), [], coin_flip_aog())
    nc = len(qcaog.get_qubits())
    cqc = get_qc(f'{nc}q-qvm')
    cwfn = WavefunctionSimulator().wavefunction(qcaog)
    print(cwfn)
    cparse_dict = get_parse_dict(cwfn, coin_flip_aog())
    print(cparse_dict)