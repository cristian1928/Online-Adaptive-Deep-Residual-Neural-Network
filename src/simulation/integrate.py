from scipy.integrate import solve_ivp
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, Union, Tuple

def integrate_step(
    state: npt.NDArray[np.floating[Any]], 
    step: int, 
    dt: float, 
    derivative: Callable[[float, npt.NDArray[np.floating[Any]]], npt.NDArray[np.floating[Any]]]
) -> npt.NDArray[np.floating[Any]]:
    orig_shape: Tuple[int, ...] = np.shape(state)
    y0: npt.NDArray[np.floating[Any]] = np.asarray(state).ravel()
    def wrapped(t: float, y: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]: 
        return np.asarray(derivative(t, y.reshape(orig_shape))).ravel()
    t0: float = step * dt
    sol = solve_ivp(wrapped, [t0, t0 + dt], y0, rtol=1e-9, atol=1e-12)
    return sol.y[:, -1].reshape(orig_shape)