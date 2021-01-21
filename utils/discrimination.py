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

from copy import deepcopy
import itertools

import numpy as np
from scipy.stats import beta


def find_neighborhood(coupling_map: list, qubit: int, dist: int):
    """
    Find all the qubits within a certain distance of a given
    qubit. Distance is measure using the coupling map.

    Args:
        coupling_map: coupling map of the qubits in the device.
        qubit: the center qubit.
        dist: the maximum distance to consider.
    """
    neighborhood = set()

    dfs(coupling_map, qubit, dist, neighborhood)

    return list(neighborhood)


def dfs(coupling_map: list, qubit: int, dist: int, neighborhood: set):
    """
    Depth first search on the coupling map.
    """
    neighborhood.add(qubit)

    if dist > 0:
        for edge in coupling_map:
            if edge[0] == qubit:
                neighborhood.add(edge[1])
                dfs(coupling_map, edge[1], dist-1, neighborhood)


def bit_strings(n: int):
    """ Returns all bit strings with length n. """
    names = []
    for include in itertools.product([True, False], repeat=n):
        name = ''.join(['1' if q else '0' for q in include])
        names.append(name)

    return names


def get_cal_names(cal_qubits: set, all_qubits: set):
    """
    Returns all calibration string names where a full
    basis calibration is done on cal_qubits and all other
    qubits are left in the ground state.

    Args:
        cal_qubits: qubit on which full calibration is done.
            cal_qubits is a sub_list of all_qubits.
        all_qubits: all the qubits in the experiment but not
            necessarily all the qubits on the chip.
    """

    cal_strings = bit_strings(len(cal_qubits))

    cal_names = []

    for cal_string in cal_strings:
        cal_name = ['0']*len(all_qubits)

        for idx, q in enumerate(all_qubits):
            if q in cal_qubits:
                idx2 = list(cal_qubits).index(q)
                cal_name[idx] = cal_string[idx2]

        cal_names.append('cal_' + ''.join(cal_name))

    return cal_names


def result_subset(result, names):
    """Extract the cals from the result"""
    new_result = deepcopy(result)
    new_results = []

    for res in new_result.results:
        if res.header.name in names:
            new_results.append(res)

    new_result.results = new_results

    return new_result


def get_fidelity(discriminator, results, shots, expected_states):
    """
    Computes the assignment fidelity as 1-(p01+p10)/2 where
    x is the number of states properly discriminated and n
    is the total number of trials.
    """
    xdata = np.array(discriminator.get_xdata(results, 2))

    ydata = []
    for es in expected_states:
        ydata.extend([es]*shots)
        
    ydata = np.array(ydata)

    predicted = discriminator.discriminate(xdata)
    
    indices0 = ydata == '0'
    indices1 = ydata == '1'
    p10 = sum(predicted[indices0] != ydata[indices0]) / len(ydata[indices0])
    p01 = sum(predicted[indices1] != ydata[indices1]) / len(ydata[indices1])
    
    f_assignment = 1 - 0.5*(p10+p01)

    low10, high10 = jeffreys_interval(predicted[indices0], ydata[indices0], 0.05)
    low01, high01 = jeffreys_interval(predicted[indices1], ydata[indices1], 0.05)

    high = 1.0 - 0.5*(low01+low10)
    low = 1.0 - 0.5*(high01 + high10)
        
    return f_assignment, low - f_assignment, high - f_assignment


def jeffreys_interval(predicted, ydata, alpha):
    """
    Computes Jeffrey's interval used to obtain the confidance
    interval for the assignment fidelity.
    """
    n = len(predicted)  # number of observations
    x = sum(predicted != ydata)  # number of successes

    a = x + 0.5
    b = n - x + 0.5

    return beta.ppf(alpha/2, a, b), beta.ppf(1.0-alpha/2, a, b)

    beta.pdf


def print_fidelity(phi: tuple, qubit: int, qubits: str):
    """
    Args:
        phi: tuple of assignment fidelity.
        qubit: qubit discriminated.
        qubits: text of qubits used to discriminate.
    """
    text = 'Fidelity of Q{} discriminator using qubits '.format(qubit) + \
            qubits + ': {:.2f}%, +{:.2f}%, {:.2f}%'.format(phi[0]*100, phi[2]*100, phi[1]*100)

    print(text)
