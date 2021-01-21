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

import warnings
from copy import deepcopy
from typing import List, Tuple, Union

import qiskit
import qiskit.ignis.mitigation.measurement as mit
import qiskit.ignis.verification.tomography as tomo
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.pulse import InstructionScheduleMap, Schedule
from qiskit.pulse.reschedule import align_measures
from qiskit.qobj import PulseQobj
from qiskit.quantum_info.operators import Choi
from qiskit.result import Result, marginal_counts

__reserved_registers = [0, 1]


def create_qpt_experiment(target_circuits: List[QuantumCircuit],
                          control: int,
                          target: int,
                          backend: BaseBackend,
                          mit_readout: bool = True,
                          inst_map: InstructionScheduleMap = None,
                          basis_gate: List[str] = None,
                          sanity_check: bool = False,
                          shots: int = 2048,
                          return_schedule=False)\
        -> Tuple[Union[PulseQobj, Schedule], List[List[QuantumCircuit]], List[str]]:
    """ Create circuits and schedules for QPT.

    Args:
        target_circuits: List of target circuits for QPT experiment.
        control: index of control qubit.
        target: index of target qubit.
        backend: Target quantum system.
        mit_readout: If use readout mitigation.
        inst_map: instruction mapping object.
        basis_gate: basis gates.
        sanity_check: check memory slot mapping of generated qobj.
        shots: Number of shots.
        return_schedule: set ``True`` when return schedule object instead of qobj.

    Returns:
        Qobj, Schedules, Quantum circuits, Measurement labels

    Additional Information:
        Bit ordering is little endian as a convention of computational science community,
        as the rest of qiskit does. When you measure the CR process tomography of q0 and q1,
        you will observe XZ (ZX) interaction when q0 (q1) is control qubit.
    """
    qubits = sorted([control, target])

    back_config = backend.configuration()
    back_defaults = backend.defaults()

    if inst_map is None:
        inst_map = back_defaults.circuit_instruction_map

    if basis_gate is None:
        basis_gate = back_config.basis_gates

    if isinstance(target_circuits, QuantumCircuit):
        target_circuits = [target_circuits]

    exp_circs = []

    # create the measurement circuits for error mitigation, optional
    qr = target_circuits[0].qregs[0]
    if mit_readout:
        meas_circs, meas_labels = mit.complete_meas_cal(qubit_list=qubits, qr=qr, circlabel='mcal')
        exp_circs.extend(meas_circs)
    else:
        meas_labels = []

    # create qpt circuit
    qpt_qcs_list = []
    for target_circuit in target_circuits:
        # extract quantum registers from target circuit
        qr = target_circuit.qregs[0]
        qr0 = qr[qubits[0]]
        qr1 = qr[qubits[1]]
        qpt_qcs = tomo.process_tomography_circuits(target_circuit, measured_qubits=[qr0, qr1])
        qpt_qcs_list.append(qpt_qcs)
        exp_circs.extend(qpt_qcs)

    # transpile
    exp_circs = qiskit.transpile(exp_circs, backend, basis_gates=basis_gate)

    # schedule with measure alignment
    exp_scheds = align_measures(qiskit.schedule(exp_circs, backend=backend, inst_map=inst_map),
                                inst_map=inst_map)

    if return_schedule:
        return exp_scheds, qpt_qcs_list, meas_labels

    # assemble pulse qobj
    qobj = qiskit.assemble(exp_scheds, backend=backend, meas_level=2, shots=shots)

    # sanity check
    if sanity_check:
        for experiment in qobj.experiments:
            for inst in experiment.instructions:
                if inst.name == 'acquire':
                    memory_slot_map = inst.memory_slot
                    if memory_slot_map[qubits[0]] != __reserved_registers[0] or \
                            memory_slot_map[qubits[0]] != __reserved_registers[1]:
                        warnings.warn('Wrong memory slots are assigned. '
                                      'QPT fitter may return invalid result.')

        assert len(qobj.experiments) <= back_config.max_experiments

    return qobj, qpt_qcs_list, meas_labels


def extract_choi_matrix(result: Result,
                        qpt_qcs_list: List[List[QuantumCircuit]],
                        meas_labels: List[str]) -> Choi:
    """ Estimate quantum channel from experiment.

    Args:
        result: Result of tomography experiment.
        qpt_qcs_list: Process tomography circuits.
        meas_labels: Measurement labels.

    Note:
        Need to:

            pip install cvxopt

    Yields:
        Quantum channel in Choi matrix representation.
    """
    def format_result(data_index, chunk):
        """Create new result object from partial result and marginalize."""
        new_result = deepcopy(result)
        new_result.results = []
        new_result.results.extend(result.results[data_index:data_index + chunk])

        return marginal_counts(new_result, __reserved_registers)

    # readout error mitigation
    if len(meas_labels) > 0:
        mit_result = format_result(data_index=0,
                                   chunk=len(meas_labels))
        meas_fitter = mit.CompleteMeasFitter(mit_result, meas_labels,
                                             qubit_list=[0, 1],
                                             circlabel='mcal')
        print('readout fidelity = %.3f' % meas_fitter.readout_fidelity())
    else:
        meas_fitter = None

    # format qpt result
    qpt_results = []
    for ind, qpt_qcs in enumerate(qpt_qcs_list):
        qpt_result = format_result(data_index=len(meas_labels) + ind * len(qpt_qcs),
                                   chunk=len(qpt_qcs))
        if meas_fitter:
            qpt_results.append(meas_fitter.filter.apply(qpt_result))
        else:
            qpt_results.append(qpt_result)

    # process tomography
    for qpt_result, qpt_circuit in zip(qpt_results, qpt_qcs_list):
        process_fitter = tomo.ProcessTomographyFitter(qpt_result, circuits=qpt_circuit)
        qpt_choi = process_fitter.fit(method='cvx', solver='CVXOPT')

        yield qpt_choi
