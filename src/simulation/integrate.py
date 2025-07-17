from scipy.integrate import solve_ivp
import numpy as np

def integrate_step(state, step, dt, derivative):
    orig_shape = np.shape(state)
    y0 = np.asarray(state).ravel()
    def wrapped(t, y): return np.asarray(derivative(t, y.reshape(orig_shape))).ravel()
    t0 = step * dt
    sol = solve_ivp(wrapped, [t0, t0 + dt], y0, rtol=1e-9, atol=1e-12)
    return sol.y[:, -1].reshape(orig_shape)