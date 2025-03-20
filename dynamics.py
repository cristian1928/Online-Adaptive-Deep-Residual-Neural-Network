import numpy as np
from scipy.integrate import solve_ivp

def agent_dynamics(positions, velocities, control_output, step, time_step_delta, num_states, control_size):
    q = positions      # 3 position states
    q_dot = velocities # 3 velocity states
    t = step * time_step_delta
    
    # Define physical parameters
    m1, m2, m3 = 1.0, 2.0, 1.5  # masses (kg)
    k1, k2 = 10.0, 8.0          # spring constants (N/m)
    c1, c2 = 0.8, 0.5           # damping coefficients (N·s/m)
    l = 0.5                     # characteristic length (m)
    
    # Drift terms (acceleration from natural system dynamics)
    f = np.zeros(num_states)
    
    # Physical interpretations:
    # State 0: Angular position of a pendulum-like system
    # State 1: Position of a mass connected by springs to states 0 and 2
    # State 2: Position of a mass with nonlinear response
    
    # Drift terms - accelerations due to system physics
    f[0] = -(9.81/l)*np.sin(q[0]) - (k1/m1)*(q[0] - q[1]) - (c1/m1)*q_dot[0]
    f[1] = (k1/m2)*(q[0] - q[1]) - (k2/m2)*(q[1] - q[2]) - (c1/m2)*q_dot[1]
    f[2] = (k2/m3)*(q[1] - q[2]) - (c2/m3)*np.sign(q_dot[2])*q_dot[2]**2
    
    # Control effectiveness matrix (maps control inputs to accelerations)
    # rows = number of states, columns = number of control inputs
    g = np.zeros((num_states, control_size))
    
    # Actuator 1: Direct torque/force on pendulum system
    g[0, 0] = 1.0/m1
    
    # Actuator 2: Force between first and second masses
    g[0, 1] = -0.5/m1
    g[1, 1] = 0.5/m2
    
    # Actuator 3: Force between second and third masses
    g[1, 2] = -0.7/m2
    g[2, 2] = 0.7/m3
    
    # Actuator 4: Variable stiffness control for first spring
    spring_factor = 0.4 * (q[0] - q[1])
    g[0, 3] = -spring_factor/m1
    g[1, 3] = spring_factor/m2
    
    # Actuator 5: Variable damping control
    damping_factor = 0.3 * q_dot[2]
    g[2, 4] = -damping_factor/m3
    
    # Pseudoinverse for control allocation
    g_plus = np.linalg.pinv(g)
    
    # External disturbances (wind forces, unmodeled dynamics, etc.)
    omega = np.zeros(num_states)
    omega[0] = 0.2*np.sin(2*t)                # Periodic disturbance on first mass
    omega[1] = 0.15*np.cos(3*t)               # Periodic disturbance on second mass
    omega[2] = 0.1*np.sin(t)*np.cos(2*t)      # Mixed frequency disturbance on third mass
    
    # Return the acceleration values, control effectiveness matrix, and its pseudoinverse
    return f + (g @ control_output) + omega, g, g_plus

def target_dynamics(positions, velocities, num_states):
    q = positions
    q_dot = velocities
    
    # Physical parameters
    m_target = 1.2    # mass of target (kg)
    k_target = 4.0    # spring constant (N/m)
    c_target = 0.3    # damping coefficient (N·s/m)
    omega = 2.0       # natural frequency (rad/s)
    
    # Drift terms - accelerations for target trajectory
    f = np.zeros(num_states)
    
    # Target state 0: Oscillatory motion with coupling to state 1
    # Physically represents a pendulum-like motion
    f[0] = -omega**2 * np.sin(q[0]) - c_target/m_target * q_dot[0] + 0.15 * q[1] * np.cos(q_dot[1])
    
    # Target state 1: Nonlinear oscillator with coupling to states 0 and 2
    # Physically represents a mass-spring-damper with nonlinear effects
    f[1] = -k_target/m_target * np.sin(q[1]) - c_target/m_target * q_dot[1] + 0.2 * q[0] * q_dot[0] - 0.1 * q[2]
    
    # Target state 2: Driven oscillator with coupling to state 1
    # Physically represents a forced oscillator with damping
    f[2] = -0.8 * omega**2 * np.sin(q[2]) - 0.5 * c_target/m_target * q_dot[2] + 0.25 * q[1] * np.cos(q_dot[1])
    
    return f

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