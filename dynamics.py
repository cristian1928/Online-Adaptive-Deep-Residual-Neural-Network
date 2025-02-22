import numpy as np
from scipy.integrate import solve_ivp

def agent_dynamics(control_output):
    return control_output

def target_dynamics(step, time_step_delta):
    t = step * time_step_delta
    dx_dt = -np.sqrt(2)*(np.sin(t)**3 + np.sin(t)) / ((np.sin(t)**2 + 1)**2)
    dy_dt = np.sqrt(2)*(np.cos(t)**2 - np.sin(t)**2) / ((np.sin(t)**2 + 1)**2)
    dz_dt = np.cos(t)
    drift = np.array([dx_dt, dy_dt, dz_dt])
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