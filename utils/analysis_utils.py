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

import itertools
import json
import pickle
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Dict, Union

import math
import matplotlib.pyplot as plt
import numpy as np
import qiskit.quantum_info as qi
import scipy.linalg as la
import scipy.optimize as opt
from qiskit import QuantumCircuit
#from qiskit.extensions.standard import U3Gate
from qiskit.circuit.library.standard_gates import U3Gate
from qiskit.providers import BaseBackend
from qiskit.qobj import PulseQobj
from qiskit.quantum_info import Operator, Choi
from qiskit.result import Result

try:
    from qiskit.quantum_info import dnorm
    __HAS_DNORM = True
except ImportError:
    __HAS_DNORM = False


def hamiltonian_reconstruction(channels: List[Choi],
                               pauli_labels: List[str],
                               gate_time: float,
                               phase_shifts: List[float] = None,
                               shifter_label: str = 'ZI',
                               sanity_check: bool = False)\
        -> Tuple[Dict[str, np.ndarray], List[float]]:
    """ Extract Pauli term coefficient from quantum channel.
    Args:
        channels: quantum channels to reconstruct Hamiltonian.
        pauli_labels: name of Pauli terms
        gate_time: duration of gate
        phase_shifts: phase shift to unwrap 2pi uncertainty.
        shifter_label: pauli term to shift.
        sanity_check: do sanity check.

    Additional information:
        To remove 2pi uncertainty of Logm, we need decompose superop S to satisfy |M| < 2 pi.
        S_H = exp(w * L_H) where L_H is superop of Pauli term shift.
        According to BCH expansion::
            M = logm(S. S_H)
              = logm(exp(tG).exp(w L_H))
              = tG + w L_H + t*w*[G, L_H] + O(2 coms)
            M' = M - w L_H
        Then Pauli coefficients are::
            b = Tr[B^dag.M']
              = t*Tr[B^dag.G] + t*w*Tr[B^dag.[G, L_H]] + O(2 coms)
              = b_true + + t*w*Tr[B^dag.[G, L_H]] + O(2 coms)
        When commutator of G and L_H is zero, b = b_true.
        Optimizer finds w to calculate principal matrix log.
    """
    threshold_san1 = 1e-3
    threshold_san2 = 1e-1

    def hamiltonian_superop(ham):
        ham = qi.Operator(ham)
        dim, _ = ham.dim
        iden = np.eye(dim)
        super_op = -1j * np.kron(iden, ham.data) + 1j * np.kron(np.conj(ham.data), iden)
        return qi.SuperOp(super_op)

    if phase_shifts is None:
        phase_shifts = [0 for _ in range(len(channels))]

    coeffs = defaultdict(list)
    estimated_hamiltonian_fidelities = []
    for phase_shift, chan in zip(phase_shifts, channels):
        sup_s = qi.SuperOp(chan)
        sup_l_h = hamiltonian_superop(qi.Operator.from_label(shifter_label))

        def logm(w):
            sup_s_h = qi.SuperOp(la.expm(w * sup_l_h.data))
            return la.logm((sup_s @ sup_s_h).data)

        def cost_func(w):
            gen_m = logm(w) - w * sup_l_h.data
            target = qi.SuperOp(la.expm(gen_m))
            if __HAS_DNORM:
                return dnorm(sup_s - target)
            # use 2-norm when old version qiskit is used. both cost functions perform comparably.
            return la.norm(sup_s.data - target.data)

        def log_constraint(w):
            return 2 * np.pi - la.norm(logm(w))

        cons = ({'type': 'ineq', 'fun': log_constraint})

        opt_result = opt.minimize(cost_func, x0=phase_shift, constraints=cons, method='SLSQP')
        w_opt = opt_result.x[0]

        # opt status
        print('w_opt = %.3e, cost_func = %.3e, generator norm = %.3e' % (w_opt,
                                                                         cost_func(w_opt),
                                                                         log_constraint(w_opt)))

        # sanitary check 1
        sup_s_h_opt = qi.SuperOp(la.expm(w_opt * sup_l_h.data))
        com_norm = la.norm((sup_s @ sup_s_h_opt - sup_s_h_opt @ sup_s).data)
        print('Commutator [S, S_H] norm = %.3e' % com_norm)

        if sanity_check:
            assert com_norm < threshold_san1

        gen_m_opt = logm(w_opt) - w_opt * sup_l_h.data

        for pauli_label in pauli_labels:
            sup_b = hamiltonian_superop(0.5 * qi.Operator.from_label(pauli_label).data)
            sup_b_dag = sup_b.adjoint()

            renorm = np.real(np.trace((sup_b_dag @ sup_b).data))
            coeff = np.real(np.trace(np.dot(gen_m_opt, sup_b_dag.data)) / renorm)
            coeffs[pauli_label].append(coeff / gate_time)

        # sanitary check 2
        reconst_ham = np.zeros((4, 4))
        for pauli_label in pauli_labels:
            ham_op = 0.5 * qi.Operator.from_label(pauli_label).data
            reconst_ham = reconst_ham + coeffs[pauli_label][-1] * ham_op

        reconst_u = qi.Operator(la.expm(-1j * reconst_ham * gate_time))
        u_fid = qi.average_gate_fidelity(chan, reconst_u)
        estimated_hamiltonian_fidelities.append(u_fid)

        if sanity_check:
            assert 1 - u_fid < threshold_san2

    # list -> ndarray
    coeffs = dict(coeffs)
    for pauli_label in coeffs.keys():
        coeffs[pauli_label] = np.array(coeffs[pauli_label])

    return coeffs, estimated_hamiltonian_fidelities


def cr_3rd_order_perturbation(amp: float,
                              fit_j: float,
                              fit_c: float,
                              delta: float,
                              alpha_c: float,
                              alpha_t: float,
                              pauli: str = 'ZX') -> float:
    """ Theoretical curves of CR Hamiltonian by 3rd order perturbation method.

    Args:
        amp: CR pulse amplitude
        fit_j: exchange coupling strength
        fit_c: coefficient of drive strength
        delta: frequency detuning of control qubit
        alpha_c: anharmonicity of control qubit
        alpha_t: anharmonicity of target qubit
        pauli: name of Pauli terms to fit
    """
    omega = fit_c * amp

    a1 = alpha_c
    a2 = alpha_t
    d = delta

    if pauli == 'IX':
        coef = - fit_j * omega / (d + a1) \
               + d * a1 * fit_j * omega**3 / ((d + a1)**3 * (2*d + a1) * (2*d + 3*a1))
    elif pauli == 'IY':
        coef = 0 * omega
    elif pauli == 'IZ':
        coef = 0.5 * fit_j**2 * omega**2 * ((a1**3 - 2*a1*d**2 - 2*d**3) / (a1 * d**2 * (a1 + d)**2 * (d-a2))
                                            + (a1**2 + d**2) / (d**2 * a2 * (a1 + d)**2)
                                            + (6 * a1**5 + 4 * a1**4 * d - 6 * a1**3 * d**2 + 7 * a1**2 * d**3 + 12 * a1 * d**4 + 4 * d**5) / (d**2 * (a1 + d)**2 * (2 * a1 + d)**2 * (a1 + 2 * d) * (3 * a1 + 2 * d))
                                            + 2 / (a1 * (a1 + d) * (a1 + d - a2))
                                            + 2 / ((a1 + d) * (a1 + d - a2)**2)
                                            + 1 / (d * (d - a2)**2))
    elif pauli == 'ZI':
        coef = - a1 * omega**2 / (2 * d * (a1 + d)) \
               + fit_j**2 * omega**2 / (2 * (a1 + d)**3) * (2 * (a1**2 + a1 * d + d**2) * (a1 + d) / (a1 * d * (a2 - d))
                                                            + 0.5 * a1 * (4 * a1**2 / d**3
                                                                          + 11 * a1 / d**2
                                                                          + 3 * a1 / (2 * a1 + d)**2
                                                                          - 2 / (a1 + 2 * d)
                                                                          - 6 / (3 * a1 + 2 * d)
                                                                          + 12 / d)
                                                            + 2 * (a1 + d)**2 / (a1 * (a1 + d - a2))
                                                            + 2 * (a1 + d)**2 / (a1 + d - a2)**2
                                                            - 2 * a1 * (a1 + d) / (d * a2))
    elif pauli == 'ZX':
        coef = - fit_j * omega / d * (a1 / (a1 + d)) \
               + fit_j * omega**3 * a1**2 * (3 * a1**3 + 11 * a1**2 * d + 15 * a1 * d**2 + 9 * d**3) / (4 * d**3 * (a1 + d)**3 * (a1 + 2 * d) * (3 * a1 + 2 * d))
    elif pauli == 'ZY':
        coef = 0 * omega
    elif pauli == 'ZZ':
        coef = fit_j**2 / (2 * (a1 + d)**2) * (omega**2 * ((a1**3 - 2 * a1 * d**2 - 2 * d**3) / (a1 * d**2 * (a2 - d))
                                                           + 0.5 * (4 * (3 * a1 + d) * (a1**2 + a1 * d + d**2) / (d**2 * (2 * a1 + d)**2)
                                                                    - 16 * d / (3 * a1**2 + 8 * a1 * d + 4 * d**2))
                                                           + 2 * a1 / (d * a2)
                                                           - 2 * (a1 + d) / (a1 + d - a2)**2
                                                           - 2 * (a1 + d) / (a1 * (a1 + d - a2)))
                                               + 2 * (a1 + d) * (a1 + a2) / (d - a2))
    else:
        raise ValueError('Given Pauli term does not exist.')

    return coef


def local_fidelity_optimization(channel: Choi,
                                target_op: Operator) -> np.ndarray:
    """ Find local operation parameters to improve fidelity.

    Args:
        channel: Estimated quantum channel of target operation.
        target_op: Ideal target operator.

    Returns:
        Local operation parameters.
    """

    def local_rotations_inv(theta1, phi1, lam1, theta2, phi2, lam2):
        return Operator(U3Gate(-theta1, -lam1, -phi1)).\
            expand(Operator(U3Gate(-theta2, -lam2, -phi2)))

    def fidelity_objective(params):
        local_l = local_rotations_inv(*params[:6])
        local_r = local_rotations_inv(*params[6:])
        opt_oper = local_l.compose(target_op).compose(local_r)
        return 1 - qi.process_fidelity(channel, opt_oper)

    def to_favg(val):
        return (4 * val + 1) / 5

    raw_fid = qi.process_fidelity(channel, target_op)
    print('Original F_avg: %.5f' % to_favg(raw_fid))

    res = opt.dual_annealing(fidelity_objective, [(-np.pi, np.pi) for _ in range(12)])

    print('Optimized F_avg: %.5f' % to_favg(1 - res.fun))

    return res.x


def optimize_circuit(target_circuit: QuantumCircuit,
                     control: int,
                     target: int,
                     local_oper_params: np.ndarray) -> QuantumCircuit:
    """ Add local U3 operations to optimize the circuit.

    Args:
        target_circuit: input circuit to optimize.
        control: control qubit index.
        target: target qubit index.
        local_oper_params: local rotation parameters.
    """
    qubits = sorted([control, target])

    # extract quantum registers from target circuit
    qr = target_circuit.qregs[0]

    qr0 = qr[qubits[0]]
    qr1 = qr[qubits[1]]

    qc = QuantumCircuit(qr)
    qc.u3(*local_oper_params[0:3], qr0)
    qc.u3(*local_oper_params[3:6], qr1)
    qc += target_circuit
    qc.u3(*local_oper_params[6:9], qr0)
    qc.u3(*local_oper_params[9:12], qr1)

    return qc


def expectation_val(result: Result,
                    qubit_ind: int,
                    exp_name: str) \
        -> float:
    """ Calculate expectation value of measurement basis.

    Args:
        result: result of experiment with meas_level=2.
        qubit_ind: index of target qubit.
        exp_name: name of target experiment.
    """
    count_dict = result.get_counts(exp_name)

    expv = 0
    for key, val in count_dict.items():
        if key[::-1][qubit_ind] == '1':
            expv -= val
        else:
            expv += val

    return expv / sum(count_dict.values())


def plot_quantum_channel(channels: List[Choi],
                         axs_real: List[plt.Axes],
                         axs_imag: List[plt.Axes]) \
        -> Tuple[List[plt.Axes], List[plt.Axes]]:
    """ Hinton plot of chi matrix.

    Args:
        channels: channels to plot.
        axs_real: matplotlib axes for plotting real part.
        axs_imag: matplotlib axes for plotting imaginary part.
    """
    basis = ['i', 'x', 'y', 'z']

    for channel, ax1, ax2 in zip(channels, axs_real, axs_imag):
        mat = qi.Chi(channel).data
        num = int(math.log(mat.shape[0], len(basis)))

        # get the labels
        row_names = list(map(''.join, itertools.product(basis, repeat=num)))
        column_names = list(map(''.join, itertools.product(basis, repeat=num)))

        max_weight = 2 ** np.ceil(np.log(np.abs(mat).max()) / np.log(2))
        datareal = np.real(mat)
        dataimag = np.imag(mat)
        lx = len(datareal[0])  # Work out matrix dimensions
        ly = len(datareal[:, 0])
        # Real
        ax1.patch.set_facecolor('gray')
        ax1.set_aspect('equal', 'box')
        ax1.xaxis.set_major_locator(plt.NullLocator())
        ax1.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(datareal):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax1.add_patch(rect)

        ax1.set_xticks(np.arange(0, lx + 0.5, 1))
        ax1.set_yticks(np.arange(0, ly + 0.5, 1))
        ax1.set_yticklabels(row_names, fontsize=14)
        ax1.set_xticklabels(column_names, fontsize=14, rotation=90)

        ax1.autoscale_view()
        ax1.invert_yaxis()
        ax1.set_title(r'Re[$\chi$]', fontsize=14)

        # Imaginary
        ax2.patch.set_facecolor('gray')
        ax2.set_aspect('equal', 'box')
        ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(dataimag):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax2.add_patch(rect)

        ax2.set_xticks(np.arange(0, lx + 0.5, 1))
        ax2.set_yticks(np.arange(0, ly + 0.5, 1))
        ax2.set_yticklabels(row_names, fontsize=14)
        ax2.set_xticklabels(column_names, fontsize=14, rotation=90)

        ax2.autoscale_view()
        ax2.invert_yaxis()
        ax2.set_title(r'Im[$\chi$]', fontsize=14)

    return axs_real, axs_imag


class ExperimentRunner:
    _ext = 'json'

    def __init__(self, path: str, backend: BaseBackend, cached: bool = True):
        """ Create new runner. Data is cached in serialized format.
        This is more robust to the class data structure change.

        Args:
            path: path to data directory.
            backend: IBM Quantum backend object to run experiment.
            cached: set ``True`` when load cache data.
        """
        self._path = path
        self._backend = backend
        self._cached = cached

    def run(self, qobjs: List[PulseQobj], file_name: str) -> Union[List[Result], Result]:
        if not isinstance(qobjs, list):
            qobjs = [qobjs]

        file_path = '%s/%s.%s' % (self._path, file_name, ExperimentRunner._ext)

        if not self._cached:
            jobs = []
            for qobj in qobjs:
                job = self._backend.run(qobj)
                jobs.append(job)
            job_results = []
            for job in jobs:
                job_results.append(job.result(timeout=3600))
            self._save(job_results, file_path)
        else:
            job_results = self._load(file_path)

        if len(job_results) == 1:
            return job_results[0]
        else:
            return job_results

    @staticmethod
    def _save(results: List[Result], file_path: str):
        results = [ExperimentRunner._remove_backend_name(result) for result in results]
        dict_results = []
        for result in results:
            dict_results.append(result.to_dict())
        
        with open(file_path, 'w') as fp:
            json.dump(dict_results, fp,default=str)

    @staticmethod
    def _load(file_path: str) -> List[Result]:
        with open(file_path, 'r') as fp:
            dict_results = json.load(fp)

        results = []
        for dict_result in dict_results:
            results.append(Result.from_dict(dict_result))

        return results

    @staticmethod
    def _remove_backend_name(result_obj: Result):
        new_result = deepcopy(result_obj)
        new_result.backend_name = ''
        new_result.header.backend_name = ''

        return new_result
