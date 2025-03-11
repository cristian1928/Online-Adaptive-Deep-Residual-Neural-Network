import numpy as np
from scipy.integrate import solve_ivp

def agent_dynamics(positions, velocities, control_output):
    return control_output

def target_dynamics(positions, velocities):
    x, y, z = positions
    xDot, yDot, zDot = velocities
    drift = np.array([np.cos(x), np.sin(y), np.cos(2*z)])
    return drift

def integrate_step(initial_state, step, time_step_delta, derivative_func):
    orig_shape = np.shape(initial_state)
    y0 = np.asarray(initial_state).ravel()
    def wrapped_derivative(t, y):
        y_reshaped = y.reshape(orig_shape)
        dy_dt = derivative_func(t, y_reshaped)
        return np.asarray(dy_dt).ravel()
    sol = solve_ivp(wrapped_derivative, [step, step + time_step_delta], y0)
    y_final = sol.y[:, -1].reshape(orig_shape)
    return y_final