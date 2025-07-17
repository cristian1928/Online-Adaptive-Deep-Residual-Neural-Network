import numpy as np
from typing import Dict, Callable, List

def attitude_mrp(state: np.ndarray) -> np.ndarray:
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = v
        return np.array([[ 0, -z,  y], [ z,  0, -x], [-y,  x,  0]])

    r = state
    r2 = np.dot(r, r)
    B  = (1 - r2)*np.eye(3) + 2*_skew(r) + 2*np.outer(r, r)

    J      = np.diag([2.0, 1.2, 1.6])
    tau_b  = np.array([0.0, 0.15, 0.0])
    omega  = np.linalg.inv(J) @ tau_b

    r_dot = 0.5 * B @ omega
    return r_dot
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> parent of ed2ed2d (Merge pull request #14 from cristian1928/copilot/fix-12)
=======
>>>>>>> parent of ed2ed2d (Merge pull request #14 from cristian1928/copilot/fix-12)
=======
>>>>>>> parent of ed2ed2d (Merge pull request #14 from cristian1928/copilot/fix-12)

def chua(state: np.ndarray) -> np.ndarray:
    x, y, z = state
    α  = 15.6
    β  = 28.0
    m0 = -1.143
    m1 = -0.714

    g = m1*x + 0.5*(m0 - m1)*(abs(x + 1) - abs(x - 1))

    x_dot = α * (y - x - g)
    y_dot = x - y + z
    z_dot = -β * y
    return np.array([x_dot, y_dot, z_dot])

def trophic_dynamics(state: np.ndarray) -> np.ndarray:
    H, P, T = state
    r_H   = 0.6
    K     = 100.0
    a_HP  = 0.02
    a_PT  = 0.01
    d_P   = 0.3
    d_T   = 0.1

    H_dot = r_H * H * (1 - H / K) - a_HP * H * P
    P_dot = -d_P * P + a_HP * H * P - a_PT * P * T
    T_dot = -d_T * T + a_PT * P * T
    return np.array([H_dot, P_dot, T_dot])

def custom(state: np.ndarray) -> np.ndarray:
    return np.zeros_like(state)

def get_dynamics_function(dynamics_type: str) -> Callable[[np.ndarray], np.ndarray]:
    dynamics_map: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "attitude_mrp": attitude_mrp,
        "chua": chua,
        "trophic_dynamics": trophic_dynamics,
        "custom": custom
    }
    
    return dynamics_map[dynamics_type]

def get_initial_conditions(dynamics_type: str) -> List[float]:
    initial_conditions_map: Dict[str, List[float]] = {
        "attitude_mrp": [0.25, 0.10, -0.30],
        "chua": [0.2, 0.0, 0.0],
        "trophic_dynamics": [40, 9, 2],
        "custom": [0.0, 0.0, 0.0]
    }
    
    return initial_conditions_map[dynamics_type]