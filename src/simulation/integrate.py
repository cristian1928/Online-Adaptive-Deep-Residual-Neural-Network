from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# Pre-allocated arrays for integration to avoid dynamic allocation
_INTEGRATION_TEMP = {
    'y0_buffer': np.zeros(10),  # Adjust size as needed
    'result_buffer': np.zeros(10),
}

def integrate_step(state: Union[NDArray[np.float64], float], step: int, dt: float, derivative: Callable[[float, Union[NDArray[np.float64], float]], Union[NDArray[np.float64], float]]
) -> Union[NDArray[np.float64], float]:
    orig_shape = np.shape(state)
    
    # Reuse buffer for y0 if possible
    state_flat = np.asarray(state).ravel()
    state_size = len(state_flat)
    
    if state_size <= len(_INTEGRATION_TEMP['y0_buffer']):
        y0 = _INTEGRATION_TEMP['y0_buffer'][:state_size]
        y0[:] = state_flat
    else:
        y0 = state_flat  # Fallback for larger states
    
    def wrapped(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]: 
        return np.asarray(derivative(t, y.reshape(orig_shape))).ravel()
    
    t0 = step * dt
    sol = solve_ivp(wrapped, [t0, t0 + dt], y0[:state_size] if state_size <= len(_INTEGRATION_TEMP['y0_buffer']) else y0, rtol=1e-9, atol=1e-12)
    return sol.y[:, -1].reshape(orig_shape)