import bspline_reference as random_reference
import initial_reference as initial_reference


def load_reference(use_random_reference=True):
    """
    Selects which B-spline referenceto use.

    use_random_reference=True
        Uses bspline_reference.py

    use_random_reference=False
        Uses initial_reference.py
    """

    if use_random_reference:
        return random_reference

    return initial_reference