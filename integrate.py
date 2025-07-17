from scipy.integrate import solve_ivp
import numpy as np
from typing import Callable, Any, Union

def integrate_step(initial_state: Union[np.ndarray, float], step: int, time_step_delta: float, 
                  derivative_func: Callable[[float, Union[np.ndarray, float]], Union[np.ndarray, float]]) -> Union[np.ndarray, float]:
    orig_shape = np.shape(initial_state)
    y0 = np.asarray(initial_state).ravel()
    
    def wrapped_derivative(t: float, y: np.ndarray) -> np.ndarray:
        y_reshaped = y.reshape(orig_shape)
        dy_dt = derivative_func(t, y_reshaped)
        return np.asarray(dy_dt).ravel()
    
    sol = solve_ivp(wrapped_derivative, [step, step + time_step_delta], y0)
    return sol.y[:, -1].reshape(orig_shape)