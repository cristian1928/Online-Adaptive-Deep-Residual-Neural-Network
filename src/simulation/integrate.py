from scipy.integrate import solve_ivp
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Union

def integrate_step(
    state: Union[NDArray[np.float64], float], 
    step: int, 
    dt: float, 
    derivative: Callable[[float, Union[NDArray[np.float64], float]], Union[NDArray[np.float64], float]]
) -> Union[NDArray[np.float64], float]:
    orig_shape = np.shape(state)
    y0 = np.asarray(state).ravel()
    def wrapped(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]: 
        return np.asarray(derivative(t, y.reshape(orig_shape))).ravel()
    t0 = step * dt
    sol = solve_ivp(wrapped, [t0, t0 + dt], y0, rtol=1e-9, atol=1e-12)
    return sol.y[:, -1].reshape(orig_shape)