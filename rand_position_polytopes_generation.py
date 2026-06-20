import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point, box as shapely_box


DEFAULT_TEMPLATE_SHAPES = [
    [[5, 5], [9, 5], [7, 10]],
    [[12, 2], [17, 3], [16, 7], [11, 8]],
    [[10, 14], [12, 12], [14, 12], [16, 16], [12, 16]],
    [[3, 14], [3, 19], [6.5, 14], [6.5, 19]],
]


def order_convex_vertices(vertices):
    """
    Orders the vertices of a convex polygon.

    This is useful because some shapes, such as rectangles, may not have their
    points written in the correct polygon order.
    """
    V = np.asarray(vertices, dtype=float)

    hull = ConvexHull(V)
    return V[hull.vertices]


def randomize_obstacle_positions(
    template_shapes,
    bounds,
    start,
    goal,
    start_goal_clearance=1.0,
    obstacle_gap=0.3,
    max_tries=1000,
    seed=None,
):
    """
    Randomly translates existing obstacle shapes.

    The obstacle shape, size and orientation are preserved.
    Only the obstacle position is changed.

    Parameters
    ----------
    template_shapes : list
        List of obstacle vertices.

    bounds : dict
        Dictionary with keys:
        xmin, xmax, ymin, ymax

    start : np.ndarray
        Start position, usually shape (2, 1).

    goal : np.ndarray
        Goal position, usually shape (2, 1).

    start_goal_clearance : float
        Minimum clearance from start and goal.

    obstacle_gap : float
        Minimum gap between obstacles.

    max_tries : int
        Maximum number of attempts for each obstacle.

    seed : int or None
        Use an integer for repeatable obstacle generation.

    Returns
    -------
    randomized_shapes : list
        List of randomized obstacle vertices.
    """

    rng = np.random.default_rng(seed)

    xmin = bounds["xmin"]
    xmax = bounds["xmax"]
    ymin = bounds["ymin"]
    ymax = bounds["ymax"]

    workspace = shapely_box(xmin, ymin, xmax, ymax)

    start = np.asarray(start).flatten()
    goal = np.asarray(goal).flatten()

    start_area = Point(start[0], start[1]).buffer(start_goal_clearance)
    goal_area = Point(goal[0], goal[1]).buffer(start_goal_clearance)

    randomized_shapes = []
    placed_polygons = []

    for shape in template_shapes:
        V = order_convex_vertices(shape)

        centroid = np.mean(V, axis=0)

        # Local coordinates keep the original size and orientation.
        V_local = V - centroid

        min_cx = xmin - np.min(V_local[:, 0])
        max_cx = xmax - np.max(V_local[:, 0])

        min_cy = ymin - np.min(V_local[:, 1])
        max_cy = ymax - np.max(V_local[:, 1])

        placed = False

        for _ in range(max_tries):
            new_centroid = np.array(
                [
                    rng.uniform(min_cx, max_cx),
                    rng.uniform(min_cy, max_cy),
                ]
            )

            V_new = V_local + new_centroid
            poly_new = Polygon(V_new)

            # Keep obstacle fully inside the workspace.
            if not workspace.covers(poly_new):
                continue

            # Keep obstacle away from start and goal.
            if poly_new.intersects(start_area):
                continue

            if poly_new.intersects(goal_area):
                continue

            # Avoid obstacle-obstacle overlap.
            overlaps_existing = False

            for existing_poly in placed_polygons:
                if poly_new.buffer(obstacle_gap).intersects(existing_poly):
                    overlaps_existing = True
                    break

            if overlaps_existing:
                continue

            randomized_shapes.append(V_new.tolist())
            placed_polygons.append(poly_new)
            placed = True
            break

        if not placed:
            raise RuntimeError(
                "Could not place one obstacle. "
                "Try reducing start_goal_clearance or obstacle_gap."
            )

    return randomized_shapes