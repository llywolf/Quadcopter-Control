import numpy as np

def print_tracking_metrics(method_name, pos, pos_ref, u, w_min=115.0, w_max=907.0, steady_tail=50):
    """
    pos      : actual position array, shape (N, 3)
    pos_ref  : reference position array, shape (N, 3)
    u        : motor speed commands, shape (N, 4), in rad/s
    """

    pos = np.asarray(pos)
    pos_ref = np.asarray(pos_ref)
    u = np.asarray(u)

    # Use same length for all signals
    N = min(len(pos), len(pos_ref), len(u))
    pos = pos[:N, :3]
    pos_ref = pos_ref[:N, :3]
    u = u[:N]

    # Position error
    err = pos - pos_ref
    err_norm = np.linalg.norm(err, axis=1)

    # Metrics
    rmse = np.sqrt(np.mean(err_norm**2))
    max_error = np.max(err_norm)

    tail = min(steady_tail, len(err_norm))
    steady_state_error = np.mean(err_norm[-tail:])

    command_smoothness = np.sum(np.diff(u, axis=0)**2)

    commands_valid = np.all((u >= w_min) & (u <= w_max))
    commands_valid_text = "yes" if commands_valid else "no"

    print("\n" + "=" * 60)
    print(f"Metrics for {method_name}")
    print("=" * 60)
    print(f"RMSE position [m]: {rmse:.6f}")
    print(f"Max position error [m]: {max_error:.6f}")
    print(f"Steady-state error [m]: {steady_state_error:.6f}")
    print(f"Command smoothness sum||Delta u_k||^2: {command_smoothness:.6f}")
    print(f"Commands within [{w_min}, {w_max}] rad/s: {commands_valid_text}")

    return {
        "rmse": rmse,
        "max_error": max_error,
        "steady_state_error": steady_state_error,
        "smoothness": command_smoothness,
        "commands_valid": commands_valid_text
    }