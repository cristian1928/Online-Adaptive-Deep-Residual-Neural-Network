from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


def integrate_step(state: NDArray[np.float64], step: int, dt: float,  derivative: Callable[[float, NDArray[np.float64]], NDArray[np.float64]]) -> NDArray[np.float64]:
    orig_shape = state.shape
    y0 = state.ravel()

    def wrapped_derivative(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_reshaped = y.reshape(orig_shape)
        return np.asarray(derivative(t, y_reshaped)).ravel()

    t0 = step * dt
    sol = solve_ivp(wrapped_derivative, [t0, t0 + dt], y0, rtol=1e-9, atol=1e-12)
    return sol.y[:, -1].reshape(orig_shape)