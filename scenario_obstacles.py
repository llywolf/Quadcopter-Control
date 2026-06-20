import os
import numpy as np

# Shared file used for the NMPC, Backstepping and Flatness controllers.
# NMPC saves the generated obstacle vertices here.
SCENARIO_FILE = os.path.join("data", "generated_scenario_obstacles.npz")


def save_obstacle_vertices(shapes, filename=SCENARIO_FILE):
    """
    Save the randomized obstacle vertices.

    Parameters
    ----------
    shapes : list
        List of obstacle vertices. Each element should be shaped (n_vertices, 2).
    filename : str
        Output .npz file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    shapes_obj = np.array(
        [np.asarray(shape, dtype=float) for shape in shapes],
        dtype=object,
    )

    np.savez(filename, shapes=shapes_obj)
    print(f"Saved shared obstacle scenario to: {filename}")


def load_obstacle_vertices(filename=SCENARIO_FILE):
    """
    Load randomized obstacle vertices.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Could not find {filename}. Run the NMPC file first, "
            "or use save_obstacle_vertices(shapes)."
        )

    data = np.load(filename, allow_pickle=True)
    return [np.asarray(shape, dtype=float) for shape in data["shapes"]]


def load_base_obstacles(filename=SCENARIO_FILE):
    """
    Returns obstacles in the same format used by the plotting code:
        [{"vertices": V1}, {"vertices": V2}, ...]
    """
    shapes = load_obstacle_vertices(filename)
    return [{"vertices": shape} for shape in shapes]
