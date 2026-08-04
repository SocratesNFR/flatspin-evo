import numpy as np
from matplotlib import pyplot as plt
from itertools import permutations
from functools import wraps, lru_cache
from copy import copy
from collections import OrderedDict
from PIL import Image
from joblib import Parallel, delayed
import skimage
from tqdm.auto import tqdm
import pickle as pkl
import warnings
import math
import shutil
import os

from flatspin import plotting
from flatspin.data import Dataset, read_table, load_output, is_archive_format, match_column, save_table
from flatspin.grid import Grid
from flatspin.utils import import_class, pop_params
from flatspin.cmdline import eval_params





class ProgressBar(tqdm):
    pass


class ParallelProgress(Parallel):
    def __init__(self, progress_bar, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_bar = progress_bar

    def print_progress(self):
        inc = self.n_completed_tasks - self._progress_bar.n
        self._progress_bar.update(inc)


def evaluate_outer(outer_pop, basepath, *, max_age=0, acc=np.sum, safe=True, append_history=False,**kwargs):
    """uses given accumulator func to reduce the fitness components to one value"""
    for i in outer_pop:
        fit_comp = i.fitness_components
        if safe and (fit_comp is None or np.nan in fit_comp or None in fit_comp):
            i.fitness = np.nan
            warnings.warn(f"NaN fitness component encountered in individual {i.id}. Setting fitness to NaN.")
            continue

        if not append_history:
            i.fitness =  acc(fit_comp)
            continue

        # accumulate the components first, then accumulate result with the previously stored fitnesses
        i.push_fitness_history(acc(fit_comp))
        i.fitness =  acc(i.fitness_history)


def ignore_NaN_fits(func):
    """decorator to set individuals with nan in their fitness components to have nan fitness,
    and only propergates non-nan individuals to the decorated function"""
    @wraps(func)
    def wrapper(outer_pop, *args, **kwargs):
        non_nans = []
        for indv in outer_pop:
            # print(indv.fitness_components)
            if np.isnan(indv.fitness_components).any():
                indv.fitness = np.nan
            else:
                non_nans.append(indv)

        if len(non_nans) > 0:
            func(non_nans, *args, **kwargs)

    return wrapper


@ignore_NaN_fits
def evaluate_outer_novelty_search(outer_pop, basepath, *, kNeigbours=5, plot=False, plot_bounds=None, gen=0, auto_normalise=False, **kwargs):
    from scipy.spatial import cKDTree

    novelty_file = os.path.join(basepath, "noveltyData.pkl")
    scale_file = os.path.join(basepath, "noveltyScales.pkl")

    pop_fitness_components = [indv.fitness_components for indv in outer_pop]
    new_pop_fitness_components = [indv.fitness_components for indv in outer_pop if indv.gen >= gen]
    groups = {}
    chosen_ones = []
    chosen_fitness_components = []
    for indv in outer_pop:
        key = tuple(indv.fitness_components)
        if key not in groups:
            groups[key] = []
            chosen_ones.append(indv)
            chosen_fitness_components.append(indv.fitness_components)
        groups[key].append(indv)


    if auto_normalise and os.path.exists(scale_file):
        with open(scale_file, "rb") as f:
            scale_factors = pkl.load(f)
    else:
        scale_factors = np.ones(len(chosen_fitness_components[0]))

    # Load unscaled data if novelty file exists
    if not os.path.exists(novelty_file):
        # If no novelty file make data and give all fitness 0
        unscaled_data = np.array(chosen_fitness_components)
        kdFitness = [0] * len(outer_pop)

    else:
        with open(novelty_file, "rb") as f:
            unscaled_data = pkl.load(f)

        data = unscaled_data * scale_factors
        kdTree = cKDTree(data)

        kdFitness = kdTree.query(chosen_fitness_components * scale_factors, k=kNeigbours)[0].mean(axis=1)
        # check stack has no duplicate rows
        archive_set = {tuple(row) for row in unscaled_data}
        unique_new = [fc for fc in chosen_fitness_components
                    if tuple(fc) not in archive_set]

        if unique_new:
            unscaled_data = np.vstack((unscaled_data, unique_new))

    for indv, fit in zip(chosen_ones, kdFitness):
        indv.fitness = fit

    for indv in outer_pop:
        if indv in chosen_ones:
            continue
        indv.fitness = 0

    if auto_normalise:
        scale_factors = unscaled_data.max(axis=0) - unscaled_data.min(axis=0)
        scale_factors[scale_factors == 0] = 1  # Avoid zero-range dimensions
        scale_factors = 1.0 / scale_factors
        with open(scale_file, "wb") as f:
            pkl.dump(scale_factors, f)

    with open(novelty_file, "wb") as f:
        pkl.dump(unscaled_data, f)

    # if plot and len(new_pop_fitness_components) > 0:
    #     plot_novelty(kdTree, basepath, gen, new_pop_fitness_components, plot_bounds)


def plot_novelty(kdTree, basepath, gen, new_pop_fitness_components, plot_bounds):
    import matplotlib
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(kdTree.data[:, 0], kdTree.data[:, 1], "o", color=[1, 0, 0, .1])
    fit_comp_array = np.array(new_pop_fitness_components)
    ax.plot(fit_comp_array[:, 0], fit_comp_array[:, 1], "o", color=[0, 0, 1, .7])
    if plot_bounds is not None:
        ax.set_xlim(plot_bounds[0])
        ax.set_ylim(plot_bounds[1])
    plt.savefig(os.path.join(basepath, f"novelty{gen}.png"))
    plt.close(fig)




def evaluate_outer_find_all(outer_pop, basepath, *, max_value=19, min_value=1, **kwargs):
    novelty_file = os.path.join(basepath, "novelty.pkl")
    if not os.path.exists(novelty_file):
        found = [-1] * (1 + max_value - min_value)
        found_id = [-1] * (1 + max_value - min_value)
    else:
        with open(novelty_file, "rb") as f:
            found, found_id = pkl.load(f)

    for i in outer_pop:
        fit = np.sum(i.fitness_components)
        if not np.isfinite(fit) or fit > max_value or fit < min_value:
            i.fitness = np.nan
            continue

        fit -= min_value

        dist = dist2missing(fit, found)
        if not np.isfinite(dist):
            # all found
            i.fitness = -1
            continue
        if dist == 0:
            i.fitness = 0
            # zero out nearby
            zero_upto_missing(found, fit)
            found_id[fit] = i.id

            continue

        if dist - int(dist) == 0:
            dist = int(dist)
            found[fit] = dist
        i.fitness = dist
    with open(novelty_file, "wb") as f:
        pkl.dump((found, found_id), f)
    print(found)
    print(found_id)


def dist2missing(x, found, missing=-1):
    """given index x, find smallest distance to a missing value in found"""
    if found[x] == missing:
        return 0
    if found[x] != 0:
        return found[x]
    left_dist = np.inf
    count = 0
    for j in range(x, -1, -1):
        if found[j] == missing:
            left_dist = count
            break
        elif found[j] != 0:
            left_dist = found[j] + count
            break
        count += 1
    right_dist = np.inf
    count = 0
    for j in range(x, len(found)):
        if found[j] == missing:
            right_dist = count
            break
        elif found[j] != 0:
            right_dist = found[j] + count
            break
        count += 1
    dist = np.min((left_dist, right_dist))
    return dist


def zero_upto_missing(found, x, missing=-1):
    """zero out values to left and right of x upto a missing value, missing values are negative"""
    found[x] = 0
    for i in range(x, -1, -1):
        if found[i] == missing:
            break
        found[i] = 0
    for i in range(x, len(found)):
        if found[i] == missing:
            break
        found[i] = 0


def scale_to_unit(x, upper, lower):
    return (x - lower) / (upper - lower)


def ignore_empty_pop(func):
    @wraps(func)
    def wrapper(pop, *args, **kwargs):
        if len(pop) == 0:
            return pop
        else:
            return func(pop, *args, **kwargs)

    return wrapper

def shuffled_copy(lst):
    lst_copy = lst[:]
    np.random.shuffle(lst_copy)
    return lst_copy

def make_matches_old(run_param_groups, dependent_params={}, fights_per_indv=1, merge_stratergy={}):
    """given a list of run_param_groups, make matches between them, returning one run_param for each match"""
    matches = []
    for _ in range(fights_per_indv):
        fights = [[rp] for rp in run_param_groups[0]]
        for group in run_param_groups[1:]:
            for i, rp in enumerate(shuffled_copy(group)):
                fights[i].append(rp)
        matches.extend(fights)

    merged_matches = [merge_run_params(*match, dependent_params=dependent_params, **merge_stratergy) for match in matches]
    return merged_matches

def make_matches(run_param_groups, dependent_params={}, fights_per_indv=1, merge_stratergy={}):
    """given a list of run_param_groups, make matches between them using rotation to ensure no repeat matchups
    good for 2 pops, but not so good for more than 2 as all appart from group[0] will be repeat :(
    """
    groups = [shuffled_copy(g) for g in run_param_groups]
    n = len(groups[0])
    assert fights_per_indv <= n, "fights_per_indv must be <= population size to guarantee no repeat matchups"

    matches = []
    for k in range(fights_per_indv):
        for i in range(n):
            match = [groups[0][i]] + [g[(i + k) % n] for g in groups[1:]]
            matches.append(match)

    return [merge_run_params(*match, dependent_params=dependent_params, **merge_stratergy) for match in matches]

def merge_run_params(*rps, dependent_params={}, **merge_stratergy):
    """given run params for each participant in a match, merge them into one run param dict for the match, using the merge_stratergy to resolve conflicts,
    if no merge stratergy for a given keyword, numerics are averaged and others are listed"""
    run_params = {}
    merge_stratergy["indv_id"] = lambda ids: ids # just keep ids as lst
    for rp in rps:
        for k, v in rp.items():
            if k not in run_params:
                run_params[k] = []
            run_params[k].append(v)

    for k, v in list(run_params.items()):
        if k in merge_stratergy:
            run_params[k] = merge_stratergy[k](v)
        else:
            # Default behavior: average numeric values, list others
            if all(isinstance(x, (int, float)) for x in v):
                run_params[k] = sum(v) / len(v)
            else:
                run_params[k] = v

    for i, id in enumerate(run_params["indv_id"]):
        run_params[f"indv_id_{i}"] = id
    del run_params["indv_id"]

    if dependent_params:
        # get any dependent params in dependent_params and update run param with them
        dp = eval_params(dependent_params, run_params)
        run_params.update(dp)
    return run_params

def load_states(ds, t=slice(None), spin_dir=(1,1), grid_size=None):
    """
    use spin_dir=(1,1) for pinwheel diamond , then -1 -> off and +1 -> on
    """

    states = load_output(ds, "mag", grid_size=grid_size, t=t, flatten=False)

    direction = np.array(spin_dir, dtype=float)
    # direction /= np.linalg.norm(direction) # normalize not needed for sign comparison

    # Compute dot product between each magnetization vector and the direction
    dot_products = np.einsum('...i,i->...', states, direction)  # Efficient batch dot product

    # Map values: Positive -> 1 (aligned), Perpendicular or opposite -> 0
    states = dot_products > 0

    return states

def jousting_fitness(pops, gen, outdir, dependent_params={}, n_fights=3, **kwargs):
    individual_class = type(pops[0][0])

    init_dir = os.path.join(outdir, "init")
    if os.path.exists(init_dir): # clean up old inits, easy to reconstruct if needed
        shutil.rmtree(init_dir)
    run_param_groups = [
       [indv.genome2run_params(outdir, encode_and_save=False) for indv in pop]
    for pop in pops
    ]

    def init_merge(inits):  # takes a list, not *args
        merged = np.where(np.all(np.stack(inits) == -1, axis=0), -1, 1)
        return individual_class.encode_and_save_init(merged, outdir)

    run_params=make_matches(run_param_groups, fights_per_indv=n_fights, dependent_params=dependent_params, merge_stratergy={"init": init_merge})

    id2indv = {individual.id: individual for pop in pops for individual in pop}

    def fit_func(dsi):
        # this function will return nothing, but will append to the fitness components of any relevant individuals in the population based on the results of the match
        indv_0 = id2indv[dsi.index["indv_id_0"].values[0]]
        indv_1 = id2indv[dsi.index["indv_id_1"].values[0]]

        size = np.array(dsi.params["size"])
        grid_size = (size * 2 + 1).tolist()
        states = load_states(dsi, -1, grid_size=grid_size, spin_dir=(1,0))

        half = size[0]
        score = states[0,:,:half].sum() - states[0,:,half+1:].sum()

        indv_0.fitness_components = (indv_0.fitness_components or []) + [-score]
        indv_1.fitness_components = (indv_1.fitness_components or []) + [score]



    individual_class.flatspin_eval(list(id2indv.values()), run_params, score_func=fit_func, gen=gen, outdir=outdir, **kwargs)

def proliferate_fitness(pop, gen, outdir, t_start=7, t_end=-1, luckyknot=False,t_check_first=16, border=4, ignore_range=((20,30),(20,30)),**kwargs):
    individual_class = type(pop[0])

    init_dir = os.path.join(outdir, "init")
    if os.path.exists(init_dir): # clean up old inits, easy to reconstruct if needed
        shutil.rmtree(init_dir)

    run_params=[indv.genome2run_params(outdir) for indv in pop]
    id2indv = {individual.id: individual for individual in pop}

    def fit_func(dsi):
        # this function will return nothing, but will append to the fitness components of any relevant individuals in the population based on the results of the match
        indv = id2indv[dsi.index["indv_id"].values[0]]

        if luckyknot:
            size = np.array(dsi.params["size"])
            grid_size = (size).tolist()
            spin_dir = (1, 0)
        else:
            size = np.array(dsi.params["size"]) +(1,0) # this was for size=(35,35) diamond, not sure if it's general
            grid_size = (size).tolist()
            spin_dir = (1, 1)


        # ---  check border over the first t_check_first states ---
        early_states = load_states(dsi, list(range(t_check_first)), grid_size=grid_size, spin_dir=spin_dir)
        early_states = np.asarray(early_states)  # shape: (t_check_first, H, W)

        border_clean = (
            np.all(early_states[:, :border, :] == 0) and
            np.all(early_states[:, -border:, :] == 0) and
            np.all(early_states[:, :, :border] == 0) and
            np.all(early_states[:, :, -border:] == 0)
        )

        if not border_clean:
            indv.fitness_components = (indv.fitness_components or []) + [-50]
            return

        states = load_states(dsi, [t_start, t_end], grid_size=grid_size, spin_dir=spin_dir)

        _, n_comp_start = skimage.measure.label(states[0], background=0, connectivity=2, return_num=True)


        end, n_comp_end = skimage.measure.label(states[1], background=0, connectivity=2, return_num=True)

        in_comps = set(np.unique(end))
        ((ir00, ir01), (ir10, ir11)) = ignore_range
        end[ir00:ir01, ir10:ir11] = 0
        out_comps = set(np.unique(end))

        n_exclusive_out = len(out_comps - in_comps)
        n_out = len(out_comps) - 1 # subtract 1 for the background component

        # if want to cut middle after instead.
        # end[10:20, 10:20] = 0
        # n_comp_end = len(np.unique(end)) - 1

        # score = end_score - max(n_comp_start, 1)
        score = n_out + n_exclusive_out - max(n_comp_start, 1) # domains completely outside get a bonus
        indv.fitness_components = (indv.fitness_components or []) + [score]

    individual_class.flatspin_eval(pop, run_params, score_func=fit_func, gen=gen, outdir=outdir, **kwargs)

known_fits={
    "default": evaluate_outer,
    "find_all": evaluate_outer_find_all,
    "novelty": evaluate_outer_novelty_search,
    "jousting": jousting_fitness,
    "proliferate": proliferate_fitness,
}
