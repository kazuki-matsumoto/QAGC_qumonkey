import sys
from typing import Any
from qiskit import *
import qiskit.quantum_info as qi
import numpy as np
import random

from openfermion.transforms import jordan_wigner
from openfermion.utils import load_operator
from quri_parts.algo.optimizer import Adam, OptimizerStatus
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)

from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState
from quri_parts.qiskit.circuit import circuit_from_qiskit
from quri_parts.circuit import UnboundParametricQuantumCircuit
from quri_parts.openfermion.operator import operator_from_openfermion_op

sys.path.append("../")
from utils.challenge_2023 import ChallengeSampling, QuantumCircuitTimeExceededError


challenge_sampling = ChallengeSampling(noise=True)

C_DEPTH = 1
N_QUBITS = 8


class Ansatz:
    def __init__(self, n_qubits, depth):
        
        self.n_qubits = n_qubits
        self.depth = depth

    def U_in(self, qc):
        
        for i in [0, 1, 4, 5]:
            qc.x(i)
        
        return qc
    
    def U(self):
        
        u_in = self.U_in(QuantumCircuit(self.n_qubits))
        
        parametric = UnboundParametricQuantumCircuit(self.n_qubits)
        parametric = parametric.combine(circuit_from_qiskit(qiskit_circuit = u_in).gates)
        
        
        for _ in range(self.depth):
            
            # givens_rotation phase
            for i in [1, 5]:
                parametric = parametric.combine(self.givens_rotation(target=i))
            for i in range(0, self.n_qubits-1, 2):
                parametric = parametric.combine(self.givens_rotation(target=i))
            for i in [1, 5]:
                parametric = parametric.combine(self.givens_rotation(target=i))
            
            # onsite_gate phase
            parametric = parametric.combine(self.onsite_gate())
            
            # hopping_gate phase
            for i in range(0, self.n_qubits-1, 2):
                parametric = parametric.combine(self.hopping_gate(target=i))
            for i in [1, 5]:
                parametric = parametric.combine(self.hopping_gate(target=i))
        
        return parametric
    
    def root_iS_gate(self, target):
        
        tmp_qc = QuantumCircuit(target + 2)
        
        root_iS_gate = qi.Operator([[1, 0, 0, 0],
                                    [0, 1/np.sqrt(2), 1.j/np.sqrt(2), 0],
                                    [0, 1.j/np.sqrt(2), 1/np.sqrt(2), 0],
                                    [0, 0, 0, 1]])
        
        for t in range(target):
            tmp_qc.id(t)
        
        tmp_qc.unitary(root_iS_gate, [i for i in range(target, target + 2)], label='√iS')
        
        root_iS_gate = circuit_from_qiskit(qiskit_circuit = tmp_qc).gates
        
        return root_iS_gate
    
    def givens_rotation(self, target):
        
        givens_rotation = UnboundParametricQuantumCircuit(self.n_qubits)
        givens_rotation = givens_rotation.combine(self.root_iS_gate(target))
        givens_rotation.add_ParametricRZ_gate(target + 0)
        givens_rotation.add_ParametricRZ_gate(target + 1)
        givens_rotation = givens_rotation.combine(self.root_iS_gate(target))
        givens_rotation.add_Z_gate(target + 0)
        
        return givens_rotation
    
    def onsite_gate(self):

        
        qr = QuantumRegister(self.n_qubits)
        onsite_gate = QuantumCircuit(qr, name='O')
        
        for i in range(4):
            phi = random.uniform(-np.pi, np.pi)
            onsite_gate.cp(phi, qr[i], qr[i+4])

        onsite_gate = circuit_from_qiskit(qiskit_circuit = onsite_gate).gates
        
        return onsite_gate
    
    def hopping_gate(self, target):

        # hopping_gate = QuantumCircuit(n_qubits, name='H')
        hopping_gate = UnboundParametricQuantumCircuit(self.n_qubits)
        
        hopping_gate.add_RZ_gate(target + 0, -np.pi/4)
        hopping_gate.add_RZ_gate(target + 1, np.pi/4)
        hopping_gate = hopping_gate.combine(self.root_iS_gate(target))
        hopping_gate.add_ParametricRZ_gate(target + 0)
        hopping_gate.add_ParametricRZ_gate(target + 1)
        hopping_gate = hopping_gate.combine(self.root_iS_gate(target))
        hopping_gate.add_RZ_gate(target + 0, 5*np.pi/4)
        hopping_gate.add_RZ_gate(target + 1, -np.pi/4)

        return hopping_gate


class Optimizer:
    def __init__(self, hw_oracle, hamiltonian, n_qubits, depth):
        
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.depth = depth
        
        self.hardware_type = "sc"
        self.shots_allocator = create_equipartition_shots_allocator()
        self.measurement_factory = bitwise_commuting_pauli_measurement
        
        n_shots = 10**3
        
        self.estimator = (
            challenge_sampling.create_concurrent_parametric_sampling_estimator(
                n_shots, self.measurement_factory, self.shots_allocator, self.hardware_type
            )
        )
        
        self.hw_oracle = hw_oracle
        
        self.theta = []
    
    
    def cost_fn(self, param_values):
        
        estimate = self.estimator(self.hamiltonian, self.parametric_state, [param_values])
        
        return estimate[0].value.real
    
    
    def grad_fn(self, param_values):
        
        grad = parameter_shift_gradient_estimates(
                self.hamiltonian, self.parametric_state, param_values, self.estimator
        )
        
        return np.asarray([i.real for i in grad.values])


    def optimize(self):
        
        iterations = 1
             
        iterationTotal = 0
        n_shots = 10**5
        
        num_params = 28 * self.depth
        
        parameters = np.random.rand(num_params)
        
        optimizer = Adam(ftol=10e-5)
        opt_state = optimizer.get_init_state(parameters)
        ansatz = Ansatz(self.n_qubits, self.depth)
        hw_hf = ansatz.U()
        hw_hf = hw_hf.combine(self.hw_oracle)
        self.parametric_state = ParametricCircuitQuantumState(self.n_qubits, hw_hf)


        while True:
            try:
                # for k in range(40):
                opt_state = optimizer.step(opt_state, self.cost_fn, self.grad_fn)
                print(f"iteration {iterationTotal+1}")
                cost = opt_state.cost
                parameters = opt_state.params
                print(cost)
                
                iterationTotal += 1
            
            except QuantumCircuitTimeExceededError:
            
                print("Reached the limit of shots")
                return cost, iterationTotal
            
            
            if opt_state.status == OptimizerStatus.FAILED:
                print("Optimizer failed")
                break
            if opt_state.status == OptimizerStatus.CONVERGED:
                print("Optimizer converged")
                break
            
        return cost, iterationTotal


class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self) -> tuple[Any, float]:
        energy_final = self.get_result()
        qc_time_final = challenge_sampling.total_quantum_circuit_time

        return energy_final, qc_time_final

    def get_result(self) -> float:

        ham = load_operator(
            file_name=f"{N_QUBITS}_qubits_H",
            data_directory="../hamiltonian",
            plain_text=False,
        )
        
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)
        hf_gates = ComputationalBasisState(N_QUBITS, bits=0b00001111).circuit.gates
        
        # optimize
        optimizer = Optimizer(hf_gates, hamiltonian, N_QUBITS, C_DEPTH)
        cost, iteration = optimizer.optimize()
        
        print(f"iteration used: {iteration}")
        
        return cost


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())
