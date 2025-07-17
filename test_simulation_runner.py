"""
Test runner for the simulation that can extract tracking error without plotting dependencies
"""
import json
import sys
import os
import io
from contextlib import redirect_stdout
from typing import Dict, Any
import numpy as np
import pandas as pd
from main import run_simulation
from data_manager import get_simulation_data


def run_simulation_without_plotting(config: Dict[str, Any]) -> float:
    """
    Run simulation and return RMS tracking error without plotting
    """
    # Patch matplotlib to avoid LaTeX issues 
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Mock the plotting functions to avoid LaTeX dependency issues
    import matplotlib.pyplot as plt
    
    # Override plt.show to do nothing
    original_show = plt.show
    plt.show = lambda: None
    
    try:
        # Run the simulation
        run_simulation(config)
        
        # Get simulation data
        agent_types, agents_state_data, target_state_data = get_simulation_data()
        
        # Calculate RMS tracking error
        if agents_state_data and len(agents_state_data) > 0:
            te = agents_state_data[0]["Tracking Error Norm"]
            rms_error = np.sqrt(np.mean(te**2))
            return float(rms_error)
        else:
            raise ValueError("No agent state data found")
            
    finally:
        # Restore original show function
        plt.show = original_show


def test_reference_config():
    """Test with reference configuration"""
    with open('reference_config.json', 'r') as f:
        config = json.load(f)
    
    # Run simulation and get tracking error
    tracking_error = run_simulation_without_plotting(config)
    print(f"Tracking error norm: {tracking_error}")
    
    # Expected value from the issue
    expected_error = 10.183181125499472
    tolerance = 1e-6
    
    if abs(tracking_error - expected_error) <= tolerance:
        print("✓ Test passed - tracking error matches expected value")
        return True
    else:
        print(f"✗ Test failed - expected {expected_error}, got {tracking_error}")
        return False


if __name__ == "__main__":
    success = test_reference_config()
    sys.exit(0 if success else 1)