# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json
from copy import deepcopy
from typing import List, Tuple

import qiskit
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.pulse import Schedule
from qiskit.qobj import PulseQobj


def create_rb_experiment(rb_seed_circs: List[QuantumCircuit],
                         control: int,
                         target: int,
                         backend: BaseBackend,
                         cnot_sched_control_target: Schedule = None,
                         cnot_sched_target_control: Schedule = None,
                         shots: int = 1024)\
        -> Tuple[List[PulseQobj], List[List[QuantumCircuit]]]:
    """ Create randomized benchmark qobj.

    Args:
        rb_seed_circs: RB circuits.
        control: index of control qubit.
        target: index of target qubit.
        backend: Target quantum system.
        cnot_sched_control_target: Schedule of CX(control, target)
        cnot_sched_target_control: Schedule of CX(target, control)
        shots: Number of shots.
    """
    back_defaults = backend.defaults()
    rb_inst_map = deepcopy(back_defaults.instruction_schedule_map)

    # update circuit instruction map
    if cnot_sched_control_target is not None and cnot_sched_target_control is not None:
        rb_inst_map.add('cx', qubits=(control, target), schedule=cnot_sched_control_target)
        rb_inst_map.add('cx', qubits=(target, control), schedule=cnot_sched_target_control)

    pulse_qobjs = []
    transpiled_circs = []
    for rb_seed_circ in rb_seed_circs:
        # transpile
        rb_seed_circ_transpiled = qiskit.transpile(rb_seed_circ, backend, optimization_level=0)
        transpiled_circs.append(rb_seed_circ_transpiled)
        # schedule
        rb_seed_sched = qiskit.schedule(rb_seed_circ_transpiled, backend, inst_map=rb_inst_map)
        # create pulse qobj
        pulse_qobjs.append(qiskit.assemble(rb_seed_sched, backend, meas_level=2, shots=shots))

    return pulse_qobjs, transpiled_circs


def cache_circuit(file: str,
                  data: List[List[QuantumCircuit]] = None) -> List[List[QuantumCircuit]]:
    """ Save and Load QASM data of RB experiments.

    Args:
        file: file name.
        data: list of RB QuantumCircuit object.
    """

    if data is not None:
        qasm_circs_seeds = []
        for rb_circs in data:
            qasm_circs_seed = []
            for rb_circ in rb_circs:
                qasm_circs_seed.append(rb_circ.qasm())
            qasm_circs_seeds.append(qasm_circs_seed)
        with open(file, 'w') as fp:
            json.dump(qasm_circs_seeds, fp)
        return None
    else:
        with open(file, 'r') as fp:
            qasm_circs_seeds = json.load(fp)
        quantum_circs_seeds = []
        for qasm_circs_seed in qasm_circs_seeds:
            quantum_circs_seed = []
            for qasm_circ_seed in qasm_circs_seed:
                quantum_circs_seed.append(QuantumCircuit.from_qasm_str(qasm_circ_seed))
            quantum_circs_seeds.append(quantum_circs_seed)

    return quantum_circs_seeds
