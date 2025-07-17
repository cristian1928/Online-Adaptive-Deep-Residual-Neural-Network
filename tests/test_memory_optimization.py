"""Test memory optimization by checking for pre-allocated arrays."""

import numpy as np
from src.core.neural_network import NeuralNetwork
from src.core.entity import Agent, Target
from src.simulation.dynamics import attitude_mrp, chua, trophic_dynamics


def test_neural_network_preallocation() -> None:
    """Test that neural network has pre-allocated temporary arrays."""
    config = {
        'time_step_delta': 0.1,
        'final_time': 1.0,
        'inner_activation': 'tanh',
        'output_activation': 'tanh', 
        'shortcut_activation': 'tanh',
        'num_blocks': 2,
        'num_layers': 2,
        'num_neurons': 3,
        'output_size': 2,
        'weight_bounds': 1.0,
        'maximum_learning_rate': 1.0,
        'minimum_learning_rate': 0.1,
        'initial_learning_rate': 0.5
    }

    def test_input(step):
        return np.array([1.0, 0.5])

    nn = NeuralNetwork(test_input, config)
    
    # Check that pre-allocated arrays exist
    assert hasattr(nn, '_temp_activated_layers'), "Neural network should have pre-allocated activated layers"
    assert hasattr(nn, '_temp_unactivated_layers'), "Neural network should have pre-allocated unactivated layers"
    assert hasattr(nn, '_temp_weight_matrices'), "Neural network should have pre-allocated weight matrices"
    assert hasattr(nn, '_temp_kron_eye_neurons'), "Neural network should have pre-allocated kronecker eye matrices"
    assert hasattr(nn, '_temp_neural_network_output'), "Neural network should have pre-allocated output array"
    
    # Check that arrays have correct dimensions
    assert len(nn._temp_activated_layers) == config['num_blocks'] + 1, "Should have arrays for all blocks"
    assert len(nn._temp_weight_matrices) == config['num_layers'] + 1, "Should have weight matrices for all layers"
    assert nn._temp_kron_eye_neurons.shape == (config['num_neurons'], config['num_neurons']), "Kronecker eye should match neurons"
    assert nn._temp_neural_network_output.shape == (config['output_size'], 1), "Output array should match output size"


def test_agent_preallocation() -> None:
    """Test that agent has pre-allocated temporary arrays."""
    config = {
        'time_step_delta': 0.1,
        'final_time': 1.0,
        'num_states': 3,
        'inner_activation': 'tanh',
        'output_activation': 'tanh', 
        'shortcut_activation': 'tanh',
        'num_blocks': 1,
        'num_layers': 2,
        'num_neurons': 3,
        'output_size': 3,
        'weight_bounds': 1.0,
        'maximum_learning_rate': 1.0,
        'minimum_learning_rate': 0.1,
        'initial_learning_rate': 0.5,
        'k1': 1.0,
        'dynamics_type': 'trophic_dynamics'
    }

    # Create target and agent
    target_pos = np.array([40.0, 9.0, 2.0])
    time_steps = int(config['final_time'] / config['time_step_delta'])
    target = Target(target_pos, time_steps, config)
    
    agent_pos = np.array([0.0, 0.0, 0.0])
    agent = Agent(agent_pos, time_steps, config, target, 'test_agent')
    
    # Check that agent has pre-allocated arrays
    assert hasattr(agent, '_temp_loss_reshaped'), "Agent should have pre-allocated loss reshape array"
    assert hasattr(agent, '_temp_nn_output_flat'), "Agent should have pre-allocated NN output flat array"
    
    # Check array dimensions
    assert agent._temp_loss_reshaped.shape == (config['num_states'], 1), "Loss reshape array should match state dimensions"
    assert agent._temp_nn_output_flat.shape == (config['num_states'],), "NN output flat array should match state dimensions"


def test_dynamics_function_consistency() -> None:
    """Test that optimized dynamics functions produce consistent results."""
    # Test multiple calls to ensure pre-allocated arrays work correctly
    state = np.array([0.25, 0.10, -0.30])
    
    # Call dynamics function multiple times
    result1 = attitude_mrp(state)
    result2 = attitude_mrp(state)
    result3 = attitude_mrp(state)
    
    # Results should be identical (within numerical precision)
    assert np.allclose(result1, result2), "Multiple calls should produce identical results"
    assert np.allclose(result2, result3), "Multiple calls should produce identical results"
    
    # Test other dynamics functions
    chua_state = np.array([0.2, 0.0, 0.0])
    chua_result1 = chua(chua_state)
    chua_result2 = chua(chua_state)
    assert np.allclose(chua_result1, chua_result2), "Chua function should be consistent"
    
    trophic_state = np.array([40.0, 9.0, 2.0])
    trophic_result1 = trophic_dynamics(trophic_state)
    trophic_result2 = trophic_dynamics(trophic_state)
    assert np.allclose(trophic_result1, trophic_result2), "Trophic dynamics should be consistent"


def test_no_accidental_array_mutation() -> None:
    """Test that pre-allocated arrays don't get accidentally mutated between calls."""
    config = {
        'time_step_delta': 0.1,
        'final_time': 1.0,
        'inner_activation': 'tanh',
        'output_activation': 'tanh', 
        'shortcut_activation': 'tanh',
        'num_blocks': 1,
        'num_layers': 2,
        'num_neurons': 2,
        'output_size': 2,
        'weight_bounds': 1.0,
        'maximum_learning_rate': 1.0,
        'minimum_learning_rate': 0.1,
        'initial_learning_rate': 0.5
    }

    def test_input(step):
        return np.array([1.0, 0.5])

    nn = NeuralNetwork(test_input, config)
    
    # Record initial state of pre-allocated arrays
    initial_kron_eye = nn._temp_kron_eye_neurons.copy()
    initial_output = nn._temp_neural_network_output.copy()
    
    # Perform forward pass
    _ = nn.predict(1)
    
    # Check that identity matrices haven't been modified
    assert np.allclose(nn._temp_kron_eye_neurons, initial_kron_eye), "Kronecker eye matrix should not be mutated"
    
    # Output array is expected to change, but should be reset for next use
    nn._temp_neural_network_output.fill(0.0)
    assert np.allclose(nn._temp_neural_network_output, initial_output), "Output array should be resettable"