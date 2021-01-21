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

from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import List, Tuple, Dict

import numpy as np
from qiskit import pulse, circuit, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterExpression
from qiskit.pulse import InstructionScheduleMap

# Qiskit parameter expression for variable
var_duration = circuit.Parameter('duration')
var_amp = circuit.Parameter('amp')
var_phi = circuit.Parameter('phi')
var_sigma = circuit.Parameter('sigma')
var_risefall = circuit.Parameter('risefall')


# parameter names
__reserved = ['duration', 'amp', 'phi', 'sigma', 'risefall']


def create_cr_circuit(cross_resonance_params: Dict[str, float],
                      control: int,
                      target: int,
                      u_index: int,
                      q_regs: QuantumRegister,
                      inst_map: InstructionScheduleMap,
                      basis_gates: List[str],
                      use_echo: bool
                      ) -> Tuple[QuantumCircuit, InstructionScheduleMap, List[str]]:
    """ Create circuit for cross resonance experiments.

    Args:
        cross_resonance_params: Pulse parameters for creating cr pulses.
        control: index of control qubit.
        target: index of target qubit.
        u_index: index of control channel.
        q_regs: QuantumRegister object to generate circuit.
        inst_map: instruction mapping object. use backend default if not provided.
        basis_gates: list of basis gate names.
        use_echo: set true if use echo sequence.

    Returns:
        CR circuit, New command def, New basis gates, Active pulse channels
    """
    # create new instruction map and basis gates
    extended_inst_map = deepcopy(inst_map)
    extended_basis = deepcopy(basis_gates)

    if 'x' not in extended_basis:
        extended_basis += ['x']

    # analyze parameters
    sched_vars = OrderedDict()
    sched_args = OrderedDict()
    for param in __reserved:
        value = cross_resonance_params.get(param, 0)

        if isinstance(value, ParameterExpression):
            sched_vars[param] = value
        else:
            sched_args[param] = value

    # custom schedule generator
    def cr_designer(*args, flip):
        """create cr pulse schedule."""
        # parse parameters for qiskit expression format
        config = {}
        for param in __reserved:
            if param in sched_vars:
                # NOTE: circuit bind method stores parameters as args.
                expression = args[list(sched_vars.keys()).index(param)]
            else:
                expression = sched_args[param]
            config[param] = float(expression)
            if param == 'duration':
                config[param] = int(config[param])

        # cross resonance pulse
        cr_pulse = _cr_pulse(duration=config['duration'], amp=config['amp'],
                             phi=config['phi'] + np.pi if flip else config['phi'],
                             sigma=config['sigma'], risefall=config['risefall'],
                             name='CR90%s_u_var' % ('m' if flip else 'p'))

        # create schedule
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.Delay(config['duration'], pulse.DriveChannel(control)))
        sched = sched.insert(0, pulse.Delay(config['duration'], pulse.DriveChannel(target)))
        if np.abs(config['amp']) > 0:
            sched = sched.insert(0, pulse.Play(cr_pulse, pulse.ControlChannel(u_index)))
        else:
            sched = sched.insert(0, pulse.Delay(config['duration'], pulse.ControlChannel(u_index)))
        return sched

    cr_sched_p = partial(cr_designer, flip=False)
    cr_sched_m = partial(cr_designer, flip=True)

    # update instruction map and basis gates
    extended_inst_map.add('zx_p', qubits=[control, target], schedule=cr_sched_p)
    extended_inst_map.add('zx_m', qubits=[control, target], schedule=cr_sched_m)
    extended_basis.extend(['zx_p', 'zx_m'])

    # custom gate generator
    def cr_p_circ(*args):
        return circuit.Gate(name='zx_p', num_qubits=2, params=list(args))

    def cr_m_circ(*args):
        return circuit.Gate(name='zx_m', num_qubits=2, params=list(args))

    # create parametrized circuit
    qc = QuantumCircuit(q_regs)

    qr_c = q_regs[control]
    qr_t = q_regs[target]

    sched_params = tuple(sched_vars.values())

    if use_echo:
        qc.append(cr_p_circ(*sched_params), qargs=[qr_c, qr_t])
        qc.x(qr_c)
        qc.append(cr_m_circ(*sched_params), qargs=[qr_c, qr_t])
        qc.x(qr_c)
    else:
        qc.append(cr_p_circ(*sched_params), qargs=[qr_c, qr_t])

    return qc, extended_inst_map, extended_basis


def _cr_pulse(**kwargs):
    """ Wrapper of gaussian square pulse generator.
    ::
        amp = amp * exp(-1j * phi)
    """
    kwargs['amp'] = kwargs.get('amp', 0) * np.exp(-1j*kwargs.pop('phi', 0))

    return pulse.pulse_lib.gaussian_square(**kwargs)
