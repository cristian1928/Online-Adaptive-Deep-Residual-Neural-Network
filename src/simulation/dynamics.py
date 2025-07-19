from typing import Callable, Dict, List

import numpy as np
from numpy.typing import NDArray


# =================================================
# Attitude kinematics in Modified Rodrigues Parameters
# =================================================
def attitude_mrp(state: NDArray[np.float64]) -> NDArray[np.float64]:
    def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y, z = v
        return np.array([[ 0, -z,  y], [ z,  0, -x], [-y,  x,  0]])

    # Initial conditions:  r = [0.25, 0.10, -0.30]   (‖r‖ < 1)
    r = state
    r2 = np.dot(r, r)
    B  = (1 - r2)*np.eye(3) + 2*_skew(r) + 2*np.outer(r, r)

    # constant body torque  → almost-constant angular momentum
    J      = np.diag([2.0, 1.2, 1.6])        # inertia tensor (kg·m²)
    tau_b  = np.array([0.0, 0.15, 0.0])      # body torque (N·m)
    omega  = np.linalg.inv(J) @ tau_b        # angular velocity (rad/s)

    r_dot = 0.5 * B @ omega
    return np.asarray(r_dot)

# ================================================
# Chua double-scroll chaotic circuit (dimensionless)
# ================================================
def chua(state: NDArray[np.float64]) -> NDArray[np.float64]:
    # Initial conditions:  x = 0.2,  y = 0.0,  z = 0.0
    x, y, z = state
    α  = 15.6
    β  = 28.0
    m0 = -1.143
    m1 = -0.714

    # piece-wise linear Chua diode
    g = m1*x + 0.5*(m0 - m1)*(abs(x + 1) - abs(x - 1))

    x_dot = α * (y - x - g)
    y_dot = x - y + z
    z_dot = -β * y
    return np.array([x_dot, y_dot, z_dot])

# =======================================================
# Three-tier ecological food-chain model
# =======================================================
def trophic_dynamics(state: NDArray[np.float64]) -> NDArray[np.float64]:
    # Initial conditions:  H=40, P=9, T=2   (population counts or biomass units)
    H, P, T = state
    # Parameters
    r_H   = 0.6     # prey intrinsic growth
    K     = 100.0   # prey carrying capacity
    a_HP  = 0.02    # predation rate (H→P)
    a_PT  = 0.01    # predation rate (P→T)
    d_P   = 0.3     # predator natural death
    d_T   = 0.1     # top-predator death

    H_dot = r_H * H * (1 - H / K) - a_HP * H * P
    P_dot = -d_P * P + a_HP * H * P - a_PT * P * T
    T_dot = -d_T * T + a_PT * P * T
    return np.array([H_dot, P_dot, T_dot])

# =======================================================
# Charged particle velocity in a magnetic field
# =======================================================
def charged_particle_motion(state: NDArray[np.float64]) -> NDArray[np.float64]:
    # Initial conditions: vx=0.1, vy=1.0, vz=0.5 (velocity components m/s)
    vx, vy, vz = state
    # Parameters
    q_over_m = 1.6     # charge-to-mass ratio (C/kg)
    B_field  = np.array([0.0, 0.0, 2.0])  # constant magnetic field (Tesla)
    
    # Lorentz force: F = q * (v x B)  -->  a = (q/m) * (v x B)
    v_vector   = np.array([vx, vy, vz])
    accel      = q_over_m * np.cross(v_vector, B_field)
    
    # vx_dot, vy_dot, vz_dot are the components of acceleration
    return accel

def custom(state: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.zeros_like(state)

# =======================================================
# Dynamics mapping and configuration system
# =======================================================

def get_dynamics_function(dynamics_type: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    dynamics_map: Dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
        "attitude_mrp": attitude_mrp,
        "chua": chua,
        "trophic_dynamics": trophic_dynamics,
        "charged_particle_motion": charged_particle_motion,
        "custom": custom
    }
    
    return dynamics_map[dynamics_type]

def get_initial_conditions(dynamics_type: str) -> List[float]:
    initial_conditions_map: Dict[str, List[float]] = {
        "attitude_mrp": [0.25, 0.10, -0.30],  # Modified Rodrigues Parameters (||r|| < 1)
        "chua": [0.2, 0.0, 0.0],  # Chua circuit initial conditions
        "trophic_dynamics": [40, 9, 2],  # Ecological food-chain model (H, P, T)
        "charged_particle_motion": [0.1, 1.0, 0.5], # Particle velocity (vx, vy, vz) in m/s
        "custom": [0.0, 0.0, 0.0]  # Change based on custom dynamics
    }
    
    return initial_conditions_map[dynamics_type]