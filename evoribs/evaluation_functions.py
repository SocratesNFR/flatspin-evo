import numpy as np # type: ignore
from scipy import ndimage # type: ignore
from skimage import measure # pyright: ignore[reportMissingModuleSource]

from flatspin.data import load_output

EVALUATIONS = {}


def evaluation(measure_dim):
    """Registers a function `fn(ds) -> (objective, measures)` under its own name."""
    def register(fn):
        fn.measure_dim = measure_dim
        EVALUATIONS[fn.__name__] = fn
        return fn
    return register

@evaluation(measure_dim=3)
def example_evaluation(ds,*, param1):
    raise NotImplementedError



def load_states(ds, t=slice(None), spin_dir=(1,0), grid_size=None):
    """
    use spin_dir=(1,0) for pinwheel diamond , then -1 -> off and +1 -> on
    for pinhweel diamond use grid_size = 2*model.size + 1
    """

    states = load_output(ds, "mag", grid_size=grid_size, t=t, flatten=False)
    direction = np.array(spin_dir, dtype=float)

    # Compute dot product between each magnetization vector and the direction
    dot_products = np.einsum('...i,i->...', states, direction)  # Efficient batch dot product

    # Map values: Positive -> 1 (aligned), Perpendicular or opposite -> 0
    states = dot_products > 0

    return states

def largest_domain_mass(state):
    """Number of cells in the largest 8-connected foreground component."""
    labels = measure.label(state, background=0, connectivity=2)
    if labels.max() == 0:
        return 0
    return int(np.bincount(labels.ravel())[1:].max())

def find_cycle_length(states):
    """Distance between the first repeated state and its earlier occurrence,
    or len(states) if no state repeats."""
    seen = {}
    for i, s in enumerate(states):
        key = s.tobytes()
        if key in seen:
            return i - seen[key]
        seen[key] = i
    return len(states)

def border_activity(states, thickness):
    """(T,) bool: True for frames with any active cell within `thickness` of the border."""
    border = np.zeros(states.shape[1:], dtype=bool)
    border[:thickness, :] = border[-thickness:, :] = True
    border[:, :thickness] = border[:, -thickness:] = True
    return states[:, border].any(axis=1)

@evaluation(measure_dim=2)
def novelty_life_fitness(ds, *, grid_size=None, state_step=1, burn_in=0,
                         spin_dir=(1, 0), illegal_border_size=None, objective2=None,
                         mass_log_range=(0.6, 2.3)):
    states = load_states(ds, t=slice(burn_in, None, state_step),
                         grid_size=grid_size, spin_dir=spin_dir)
    masses = states.sum(axis=(1, 2))  # total active cells per frame

    fail = (np.nan, [np.nan, np.nan], {})
    if np.any(masses == 0):
        return fail
    if illegal_border_size and border_activity(states, illegal_border_size).any():
        return fail

    mean_mass = float(np.mean(masses))
    lo, hi = mass_log_range
    mass_measure = (float(np.clip(np.log10(mean_mass), lo, hi)) - lo) / (hi - lo)


    cycle_len = find_cycle_length(states)
    is_cyclic = cycle_len < len(states)

    # Primary objective: 0 means the goal is met (cycles, or glides with constant mass).
    fitness = 0.0 if is_cyclic else -float(np.sum(np.abs(np.diff(masses))))

    achieved = fitness == 0
    if achieved and objective2 is not None:
        fitness = objective2(states, masses, is_cyclic, cycle_len)
        assert fitness >= 0, "secondary_objective must be positive"

    info = {"cycle_len": cycle_len, "mean_mass": mean_mass,
            "mode": "cycle" if is_cyclic else ("glide" if achieved else "none")}
    return fitness, [cycle_len / len(states), mass_measure], info