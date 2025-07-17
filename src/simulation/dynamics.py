from typing import Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

# Pre-allocated arrays for dynamics computations to avoid dynamic allocation
_TEMP_ARRAYS = {
    'skew_matrix': np.zeros((3, 3)),
    'identity_3x3': np.eye(3),
    'result_3d': np.zeros(3),
    'outer_product': np.zeros((3, 3)),
    'B_matrix': np.zeros((3, 3)),
    'temp_matrix': np.zeros((3, 3)),
}

# =================================================
# Attitude kinematics in Modified Rodrigues Parameters
# =================================================
def attitude_mrp(state: NDArray[np.float64]) -> NDArray[np.float64]:
    def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y, z = v
        # Reuse pre-allocated skew matrix
        skew_mat = _TEMP_ARRAYS['skew_matrix']
        skew_mat[0, 0] = 0.0;  skew_mat[0, 1] = -z;  skew_mat[0, 2] = y
        skew_mat[1, 0] = z;   skew_mat[1, 1] = 0.0;  skew_mat[1, 2] = -x
        skew_mat[2, 0] = -y;  skew_mat[2, 1] = x;   skew_mat[2, 2] = 0.0
        return skew_mat

    # Initial conditions:  r = [0.25, 0.10, -0.30]   (‖r‖ < 1)
    r = state
    r2 = np.dot(r, r)
    
    # Reuse pre-allocated matrices
    identity = _TEMP_ARRAYS['identity_3x3']
    outer_prod = _TEMP_ARRAYS['outer_product']
    B_mat = _TEMP_ARRAYS['B_matrix']
    result = _TEMP_ARRAYS['result_3d']
    
    # Compute outer product in-place
    np.outer(r, r, out=outer_prod)
    
    # Compute B matrix components in-place
    B_mat[:] = (1 - r2) * identity + 2 * _skew(r) + 2 * outer_prod

    # constant body torque  → almost-constant angular momentum
    J      = np.diag([2.0, 1.2, 1.6])        # inertia tensor (kg·m²)
    tau_b  = np.array([0.0, 0.15, 0.0])      # body torque (N·m)
    omega  = np.linalg.inv(J) @ tau_b        # angular velocity (rad/s)

    # Compute result in-place
    result[:] = 0.5 * (B_mat @ omega)
    return result.copy()  # Return copy to avoid mutation

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

    # Reuse pre-allocated result array
    result = _TEMP_ARRAYS['result_3d']
    result[0] = α * (y - x - g)  # x_dot
    result[1] = x - y + z        # y_dot
    result[2] = -β * y           # z_dot
    return result.copy()

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

    # Reuse pre-allocated result array
    result = _TEMP_ARRAYS['result_3d']
    result[0] = r_H * H * (1 - H / K) - a_HP * H * P  # H_dot
    result[1] = -d_P * P + a_HP * H * P - a_PT * P * T  # P_dot
    result[2] = -d_T * T + a_PT * P * T  # T_dot
    return result.copy()

def custom(state: NDArray[np.float64]) -> NDArray[np.float64]:
    # Reuse pre-allocated result array
    result = _TEMP_ARRAYS['result_3d']
    result.fill(0.0)  # Fill with zeros instead of creating new array
    return result.copy()

# =======================================================
# Dynamics mapping and configuration system
# =======================================================

def get_dynamics_function(dynamics_type: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    dynamics_map: Dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
        "attitude_mrp": attitude_mrp,
        "chua": chua,
        "trophic_dynamics": trophic_dynamics,
        "custom": custom
    }
    
    return dynamics_map[dynamics_type]

def get_initial_conditions(dynamics_type: str) -> List[float]:
    initial_conditions_map: Dict[str, List[float]] = {
        "attitude_mrp": [0.25, 0.10, -0.30],  # Modified Rodrigues Parameters (||r|| < 1)
        "chua": [0.2, 0.0, 0.0],  # Chua circuit initial conditions
        "trophic_dynamics": [40, 9, 2],  # Ecological food-chain model (H, P, T)
        "custom": [0.0, 0.0, 0.0]  # Default for custom dynamics
    }
    
    return initial_conditions_map[dynamics_type]