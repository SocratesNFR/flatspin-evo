import numpy as np
from matplotlib import pyplot as plt
from itertools import permutations
from functools import wraps, lru_cache
from copy import copy
from collections import OrderedDict
from time import sleep
from PIL import Image

from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pickle as pkl
import warnings

from flatspin import plotting
from flatspin.data import Dataset, read_table, load_output, is_archive_format, match_column, save_table
from flatspin.grid import Grid
from flatspin.utils import get_default_params, import_class
from flatspin.runner import run, run_dist, run_local
from flatspin.sweep import sweep
from flatspin.cmdline import eval_params

import os
import pandas as pd
import shlex
import sys


class ProgressBar(tqdm):
    pass


class ParallelProgress(Parallel):
    def __init__(self, progress_bar, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_bar = progress_bar

    def print_progress(self):
        inc = self.n_completed_tasks - self._progress_bar.n
        self._progress_bar.update(inc)


def flatspin_eval(fit_func, pop, gen, outdir, *, run_params=None, shared_params=None, do_not_override_default=False,
                  sweep_params=None, condition=None, group_by=None, max_jobs=1000,
                  repeat=1, repeat_spec=None, preprocessing=None, dont_run=False, dependent_params={}, **flatspin_kwargs):
    """
    fit_func is a function that takes a dataset and produces an iterable (or single value) of fitness components.
    if an Individual already has fitness components the value(s) will be appended
    (allows for multiple datasets per Individual)
    using group_by, it is possible to use mutliple datasets to determine the fitness of an individual
    """

    if len(pop) < 1:
        return pop
    sweep_list = (list(sweep(sweep_params, repeat, repeat_spec, params=flatspin_kwargs)) if sweep_params else [])
    default_shared = pop[0].get_default_shared_params(outdir, gen)
    if shared_params is None:
        shared_params = default_shared
    elif not do_not_override_default:
        default_shared.update(shared_params)
        shared_params = default_shared

    shared_params.update(flatspin_kwargs)

    if run_params is None:
        run_params = pop[0].get_default_run_params(pop, sweep_list, condition=condition)

    if preprocessing:
        run_params = preprocessing(run_params)

    if len(run_params) > 0:
        id2indv = {individual.id: individual for individual in pop}
        evolved_params = [
            id2indv[rp["indv_id"]].evolved_params_values for rp in run_params
        ]
        evo_run(run_params, shared_params, gen, evolved_params, max_jobs=max_jobs, wait=group_by, dont_run=dont_run, dependent_params=dependent_params)
        dataset = Dataset.read(shared_params["basepath"])
        queue = dataset
        if group_by:
            _, queue = zip(*dataset.groupby(group_by))
        queue = list(queue)
        while len(queue) > 0:
            ds = queue.pop(0)
            with np.errstate():
                indv_id = ds.index["indv_id"].values[0]
                try:
                    # calculate fitness of a dataset
                    fit_components = fit_func(ds)
                    try:
                        fit_components = list(fit_components)
                    except (TypeError):
                        fit_components = [fit_components]

                    # assign the fitness of the correct individual

                    indv = id2indv[indv_id]
                    if indv.fitness_components is not None:
                        indv.fitness_components += fit_components
                    else:
                        indv.fitness_components = fit_components
                except Exception as e:  # not done saving file
                    if shared_params.get("run", "local") != "dist" or group_by:
                        raise e
                    if type(e) not in (FileNotFoundError):
                        print(type(e), e)
                    queue.append(ds)  # queue.append((indv_id, ds))
                    sleep(2)

    # individuals that have not been evaluated (malformed)
    evaluated = set([rp["indv_id"] for rp in run_params])
    for indv in [i for i in pop if i.id not in evaluated]:
        indv.fitness_components = [np.nan]
    return pop


def evo_run(runs_params, shared_params, gen, evolved_params=None, wait=False, max_jobs=1000, dont_run=False, dependent_params={}):
    """modified from run_sweep.py main()"""
    if not evolved_params:
        evolved_params = []
    model_name = shared_params.pop("model", "CustomSpinIce")
    model_class = import_class(model_name, "flatspin.model")
    encoder_name = shared_params.get("encoder", "Sine")
    encoder_class = (import_class(encoder_name, "flatspin.encoder") if type(encoder_name) is str else encoder_name)

    data_format = shared_params.get("format", "npz")

    params = get_default_params(run)
    params["encoder"] = f"{encoder_class.__module__}.{encoder_class.__name__}"
    params.update(get_default_params(model_class))
    params.update(get_default_params(encoder_class))
    params.update(shared_params)

    info = {
        "model": f"{model_class.__module__}.{model_class.__name__}",
        "model_name": model_name,
        "data_format": data_format,
        "command": " ".join(map(shlex.quote, sys.argv)),
    }

    ext = data_format if is_archive_format(data_format) else "out"

    outdir_tpl = "gen{:d}indv{:d}"

    basepath = params["basepath"]
    if os.path.exists(basepath):
        # Refuse to overwrite an existing dataset
        raise FileExistsError(basepath)
    os.makedirs(basepath)

    index = []
    filenames = []
    # Generate queue
    for i, run_params in enumerate(runs_params):
        newparams = copy(params)
        newparams.update(run_params)
        if evolved_params:
            # get any flatspin params in evolved_params and update run param with them
            run_params.update(
                {k: v for k, v in evolved_params[i].items() if k in newparams}
            )
        if dependent_params:
            # get any dependent params in dependent_params and update run param with them
            dp = eval_params(dependent_params, run_params)
            run_params.update(dp)

        sub_run_name = newparams.get("sub_run_name", "x")
        outdir = outdir_tpl.format(gen, newparams["indv_id"]) + f"{sub_run_name}.{ext}"
        filenames.append(outdir)
        row = OrderedDict(run_params)
        row.update({"outdir": outdir})
        index.append(row)

    # Save dataset
    index = pd.DataFrame(index)
    dataset = Dataset(index, params, info, basepath)
    dataset.save()

    if dont_run:
        return
    # Run!
    # print("Starting sweep with {} runs".format(len(dataset)))
    rs = np.random.get_state()
    run_type = shared_params.get("run", "local")
    if run_type == "local":
        run_local(dataset, False)

    elif run_type == "dist":
        run_dist(dataset, wait=wait, max_jobs=max_jobs)

    np.random.set_state(rs)
    return


def evaluate_outer(outer_pop, basepath, *, max_age=0, acc=np.sum, **kwargs):
    """uses given accumulator func to reduce the fitness components to one value"""
    for i in outer_pop:
        i.fitness = acc(i.fitness_components)


def ignore_NaN_fits(func):
    """decorator to set individuals with nan in their fitness components to have nan fitness,
    and only propergates non-nan individuals to the decorated function"""
    @wraps(func)
    def wrapper(outer_pop, *args, **kwargs):
        non_nans = []
        for indv in outer_pop:
            if np.isnan(indv.fitness_components).any():
                indv.fitness = np.nan
            else:
                non_nans.append(indv)

        if len(non_nans) > 0:
            func(non_nans, *args, **kwargs)

    return wrapper


@ignore_NaN_fits
def evaluate_outer_novelty_search(outer_pop, basepath, *, kNeigbours=5, plot=False, plot_bounds=None, gen=0, **kwargs):
    from scipy.spatial import cKDTree
    import matplotlib

    novelty_file = os.path.join(basepath, "noveltyTree.pkl")
    pop_fitness_components = [indv.fitness_components for indv in outer_pop]
    new_pop_fitness_components = [indv.fitness_components for indv in outer_pop if indv.gen >= gen]
    # if no novelty tree exists, create one and give individuals fitness = 0
    if not os.path.exists(novelty_file):
        kdTree = cKDTree(pop_fitness_components)
        for indv in outer_pop:
            indv.fitness = 0
    else:
        # use the novelty tree to find the fitness, then update and save the new tree
        with open(novelty_file, "rb") as f:
            kdTree = pkl.load(f)
        kdFitness = kdTree.query(pop_fitness_components, k=kNeigbours)[0].mean(axis=1)
        for indv, fit in zip(outer_pop, kdFitness):
            indv.fitness = fit
        # add new individuals to the tree (don't re-add old individuals)
        if len(new_pop_fitness_components) > 0:
            kdTree = cKDTree(np.vstack((kdTree.data, new_pop_fitness_components)))

    with open(novelty_file, "wb") as f:
        pkl.dump(kdTree, f)

    if plot and len(new_pop_fitness_components) > 0:
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


@ignore_empty_pop
def target_state_num_fitness(pop, gen, outdir, target, state_step=None, **flatspin_kwargs):
    def fit_func(ds):
        nonlocal state_step
        if state_step is None:
            state_step = ds.params["spp"]
        spin = read_table(ds.tablefile("spin"))
        fitn = abs(len(np.unique(spin.iloc[::state_step, 1:], axis=0)) - target)
        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, **flatspin_kwargs)
    return pop


@ignore_empty_pop
def flips_fitness(pop, gen, outdir, num_angles=1, other_sizes_fractions=None, sweep_params=None, repeat=1,
                  repeat_spec=None, **flatspin_kwargs):
    shared_params = pop[0].get_default_shared_params(outdir, gen)
    shared_params.update(flatspin_kwargs)
    if not other_sizes_fractions:
        other_sizes_fractions = []
    if num_angles > 1:
        shared_params["input"] = [0, 1] * (shared_params["periods"] // 2)
    sweep_list = (list(sweep(sweep_params, repeat, repeat_spec, params=flatspin_kwargs)) if sweep_params else [])
    run_params = pop[0].get_default_run_params(pop, sweep_list, condition=flatspin_kwargs.get("condition"))
    frac_run_params = []
    if len(run_params) > 0:
        for rp in run_params:
            rp.setdefault("sub_run_name", "")

            for frac in other_sizes_fractions:
                angles_frac = rp["magnet_angles"][
                    : int(np.ceil(len(rp["magnet_angles"]) * frac))
                ]
                coords_frac = rp["magnet_coords"][
                    : int(np.ceil(len(rp["magnet_coords"]) * frac))
                ]
                frp = {
                    "indv_id": rp["indv_id"],
                    "magnet_coords": coords_frac,
                    "magnet_angles": angles_frac,
                    "sub_run_name": rp["sub_run_name"] + f"_frac{frac}",
                }
                frac_run_params.append(dict(rp, **frp))
            rp["sub_run_name"] += f"_frac{1}"

    def fit_func(ds):
        # fitness is number of steps, but ignores steps from first fifth of the run
        steps = read_table(ds.tablefile("steps"))
        fitn = (steps.iloc[-1]["steps"] - steps.iloc[(shared_params["spp"] * shared_params["periods"]) // 5]["steps"])
        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, shared_params=shared_params, run_params=run_params + frac_run_params,
                        repeat=repeat, repeat_spec=repeat_spec, **flatspin_kwargs)
    return pop


def simple_flips_fitness(pop, gen, outdir, num_angles=1, percent=True, **flatspin_kwargs):
    shared_params = pop[0].get_default_shared_params(outdir, gen)
    shared_params.update(flatspin_kwargs)
    if num_angles > 1:
        shared_params["input"] = [0, 1] * (shared_params["periods"] // 2)

    id2indv = {indv.id: indv for indv in pop}

    def fit_func(ds):
        # fitness is number of steps, but ignores steps from first fifth of the run
        steps = read_table(ds.tablefile("steps"))
        fitn = (steps.iloc[-1]["steps"] - steps.iloc[(shared_params["spp"] * shared_params["periods"]) // 5]["steps"])
        if percent:
            n_mags = len(id2indv[ds.index["indv_id"].values[0]].coords)
            fitn /= n_mags
        return fitn

    def condition(indv):
        return len(indv.coords) > 0

    pop = flatspin_eval(fit_func, pop, gen, outdir, shared_params=shared_params, condition=condition, **flatspin_kwargs)
    return pop


def music_fitness(pop, gen, outdir, grid_size=(3, 3), scale_size=12, dur_values=5, velo_values=5, **flatspin_kwargs):

    def fit_func(ds):
        UV = load_output(ds, "mag", grid_size=grid_size, flatten=False)
        U = UV[..., 0]  # x components
        V = UV[..., 1]  # y components

        # pitch
        angle = plotting.vector_colors(U, V)
        norm_angle = angle * (scale_size / (2 * np.pi))  # [0,scale_size)
        norm_angle = norm_angle.round().astype(int)
        norm_angle[norm_angle >= scale_size] = 0

        fitn = zipfness(norm_angle.flatten())
        # duration
        if dur_values > 1:
            angle_diffs = np.abs(np.diff(angle.reshape(angle.shape[0], -1), axis=0))
            angle_diffs = scale_to_unit(angle_diffs, 1, velo_values).astype(int)
            fitn += zipfness(angle_diffs.flatten())
        # velocity
        if velo_values > 1:
            magn = np.linalg.norm(UV, axis=-1).flatten()
            # scale magnitudes to 0-magn_values and discretize
            magn = np.round(magn * velo_values / np.max(magn)).astype(int)
            fitn += zipfness(magn)

        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, **flatspin_kwargs)
    return pop


def zipfness(x):
    """zipfness of a vector x"""
    x_counts = np.bincount(x).astype(float)
    x_counts = x_counts[x_counts > 0]
    x_counts[::-1].sort()  # sort in descending order
    x_counts /= x_counts[0]  # normalize to first element == 1

    target = zipf(len(x_counts))
    zipfness = np.sum(np.abs(x_counts - target))
    return zipfness

@lru_cache
def zipf(n):
    "return zipf vector starting at 1"
    zpf = 1 / np.arange(1, n + 1)
    return tuple(zpf)


@ignore_empty_pop
def majority_fitness(pop, gen, outdir, sweep_params, test_at=None, match=True, **flatspin_kwargs):
    if not test_at:
        test_at=[0.2, 0.4, 0.6, 0.8]

    if "test_perc" in sweep_params:
        warnings.warn("majority fitness function overwriting value of 'test_perc'")

    if "random_prob" in sweep_params:
        warnings.warn("majority fitness function overwriting value of 'random_prob'")
    sweep_params=dict(sweep_params, test_perc=str(test_at), random_prob="[test_perc]")

    def preprocessing(run_params):
        """mod angles, enforce odd number of spins"""
        for run in run_params:
            run["magnet_angles"] %= np.pi
            if len(run["magnet_angles"]) % 2 == 0:
                run["magnet_angles"]=run["magnet_angles"][:-1]
                run["magnet_coords"]=run["magnet_coords"][:-1]
            run["random_seed"]=np.random.randint(999999)
        return run_params

    def fit_func(ds):
        spin=read_table(ds.tablefile("spin"))
        majority_symbol=spin.iloc[0].mode()[0]
        if match:
            fitn=np.sum(spin.iloc[-1] == majority_symbol)
        else:
            fitn=np.sum(spin.iloc[-1] != majority_symbol)
        return fitn

    pop=flatspin_eval(fit_func, pop, gen, outdir, preprocessing=preprocessing, init="random", sweep_params=sweep_params,
                        **flatspin_kwargs)
    return pop


@ ignore_empty_pop
def image_match_fitness(pop, gen, outdir, image_file_loc, num_blocks=33, threshold=True, **flatspin_kwargs):
    img=np.asarray(Image.open(image_file_loc))
    lst=[]
    step=len(img) / num_blocks
    for y in range(num_blocks):
        row=[]
        for x in range(num_blocks):
            a=img[
                int(x * step): int((x + 1) * step), int(y * step): int((y + 1) * step)
            ]
            row.append(np.mean(a))
        lst.append(row)

    target=np.array(lst)
    target=np.flipud(target).flatten()
    if threshold:
        target=(target > (255 / 2)) * 255

    def fit_func(ds):
        UV=load_output(ds, "mag", t=-1, grid_size=(num_blocks,) * 2, flatten=False)
        U=UV[..., 0]  # x components
        V=UV[..., 1]  # y components
        angle=plotting.vector_colors(U, V)
        colour=np.cos(angle).flatten()
        magn=np.linalg.norm(UV, axis=-1).flatten()

        # scale colour by magnitude between -1 and 1
        colour=colour * magn / np.max(magn)
        # scale colour from 0 to 255
        colour=(colour + 1) * (255 / 2)

        fitn=np.sum(np.abs(colour - target))
        return fitn

    pop=flatspin_eval(fit_func, pop, gen, outdir, **flatspin_kwargs)
    return pop


def mean_abs_diff_error(y_true, y_pred):
    # print(f"y_true: {y_true}")
    # print(f"y_pred: {y_pred}")
    np.abs(y_true - y_pred)
    return np.abs(y_true - y_pred).mean()


@ ignore_empty_pop
def xor_fitness(pop, gen, outdir, quantity="spin", grid_size=None, crop_width=None, win_shape=None, win_step=None, cv_folds=10,
                alpha=1, sweep_params=None, encoder="Constant", angle0=-45, angle1=45, H0=0, H=1000, input=1000, spp=1, **kwargs):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import make_scorer, accuracy_score

    sweep_params=sweep_params if sweep_params else {}

    def reduce_angle(a):
        return (a + 180) % 360 - 180

    def diff_bisector(a0, a1, bool0=0, bool1=0):
        a0 += 180 * bool0
        a1 += 180 * bool1
        return a0 + reduce_angle(a1 - a0) / 2

    # calculate simulated angles from the logical axes
    logic_values=("00", "01", "10", "11")
    # default_angles = [diff_bisector(angle0, angle1, b0, b1) for b0 in (0, 1) for b1 in (0, 1)]
    sweep_params["logical_val"]=str(logic_values)

    id2indv={individual.id: individual for individual in pop}

    def preprocessing(run_params):
        """calculate phi value from logical value"""
        for run_param in run_params:
            ind=id2indv[run_param["indv_id"]]
            a0=(
                ind.evolved_params_values["angle0"]
                if "angle0" in ind.evolved_params_values
                else angle0
            )
            a1=(
                ind.evolved_params_values["angle1"]
                if "angle1" in ind.evolved_params_values
                else angle1
            )
            b0, b1=[b == "1" for b in run_param["logical_val"]]
            run_param["phi"]=diff_bisector(a0, a1, b0, b1)
        return run_params

    if np.isscalar(input):
        if "input" in sweep_params:
            print("Overwriting 'input' in xor_fitness()!!")
        input=[1] + [0] * (input - 1)

    def fit_func(dataset):
        scores=[]

        X=[]  # reservoir outputs
        y=[]  # targets
        for ds in dataset:
            logic_val=ds.index["logical_val"].values[0]
            target=logic_val in ["01", "10"]  # calculate xor
            y.append(target)
            x=read_table(ds.tablefile("spin")).iloc[-1].values[1:]
            X.append(x)
        X=np.array(X)
        y=np.array(y)
        # print(f"X: {X}")
        # print(X.shape)
        # print(f"y: {y}")
        # print(y.shape)
        readout=Ridge(alpha=alpha)
        # readout.fit(X, y)

        cv=KFold(n_splits=cv_folds, shuffle=False)
        cv_scores=cross_val_score(readout, X, y, cv=cv,
                                    scoring=make_scorer(mean_abs_diff_error, greater_is_better=False), n_jobs=1)
        # score is -error (max better)
        scores.append(cv_scores)
        fitness_components=np.mean(scores, axis=-1)

        return fitness_components

    pop=flatspin_eval(fit_func, pop, gen, outdir, encoder=encoder, sweep_params=sweep_params, H=H, H0=H0, input=input,
                        spp=spp, preprocessing=preprocessing, **kwargs)
    return pop


def mem_capacity_fitness(pop, gen, outdir, n_delays=10, t_start=None, **kwargs):
    from mem_capacity import do_mem_capacity

    if t_start is None:
        t_start=0.1 * len(kwargs["input"]) + 1

    def fit_func(ds):
        delays=np.arange(0, n_delays + 1)
        spp=int(ds.params["spp"])
        t=slice(int(t_start * spp) - 1, None, spp)
        scores=do_mem_capacity(ds, delays, t=t)
        fitness_components=scores.mean(axis=-1)
        # print("MC", np.sum(fitness_components), len(ds))
        return fitness_components

    pop=flatspin_eval(fit_func, pop, gen, outdir, **kwargs)

    return pop


def correlation_fitness(pop, gen, outdir, target, **kwargs):
    from runAnalysis import fitnessFunction

    def fit_func(x):
        return abs(fitnessFunction(x) - target)

    pop=flatspin_eval(fit_func, pop, gen, outdir, **kwargs)

    return pop


def parity_fitness(pop, gen, outdir, n_delays=10, n_bits=3, **kwargs):
    from parity import do_parity

    def fit_func(ds):
        delays=np.arange(0, n_delays)
        spp=int(ds.params["spp"])
        t=slice(spp - 1, None, spp)
        scores=do_parity(ds, delays, n_bits, t=t)
        fitness_components=scores.mean(axis=-1)
        print(f"PARITY{n_bits}", np.sum(fitness_components))
        return fitness_components

    pop=flatspin_eval(fit_func, pop, gen, outdir, **kwargs)

    return pop


def state_num_fitness(pop, gen, outdir, state_step=None, **flatspin_kwargs):
    def fit_func(ds):
        nonlocal state_step
        if state_step is None:
            state_step=ds.params["spp"]
        spin=read_table(ds.tablefile("spin"))
        fitn=len(np.unique(spin.iloc[::state_step, 1:], axis=0))
        return fitn

    pop=flatspin_eval(fit_func, pop, gen, outdir, **flatspin_kwargs)
    return pop


@ ignore_empty_pop
def state_num_fitness2(pop, gen, outdir, t=-1, bit_len=3, sweep_params=None, group_by=None, tessellate_shape=None,
                    squint_grid_size=None, polar_coords=True, fit_acc="mode", **flatspin_kwargs):
    from scipy.stats import mode
    max_state_count=2**bit_len
    input=str([list(f"{i:b}".zfill(bit_len)) for i in range(max_state_count)])

    if not sweep_params:
        sweep_params={}
    if "init" in sweep_params or "input" in sweep_params:
        warnings.warn("Overiding input in fitness function")
    sweep_params=dict(sweep_params, input=input)

    if not group_by:
        group_by=[]
    if "indv_id" not in group_by:
        group_by.append("indv_id")
    id2indv={individual.id: individual for individual in pop}
    nndist=flatspin_kwargs.get("neighbor_dist", pop[0].get_default_shared_params(select_param="neighbor_distance"))

    total_spinices=np.prod(tessellate_shape) if tessellate_shape else 1

    if squint_grid_size and total_spinices > 1:
        for indv in pop:
            indv.grids=[Grid.fixed_grid(indv.coords[:], squint_grid_size) for cell in range(total_spinices)]

    if t == -1:
        def filter(df):
            return df["t"] == df["t"].max()
    else:
        def filter(df):
            return df["t"] == t

    def preprocessing(run_params):
        if tessellate_shape is not None:
            # do tessellating
            made_files=set()
            for run in run_params:
                i_id=run["indv_id"]
                indv=id2indv[i_id]

                if i_id not in made_files:
                    pos, angles, labels=indv.fast_tessellate(tessellate_shape, padding=2 * nndist, centre=False, return_labels=True)
                    save_table(pos, os.path.join(outdir, f"indv_{i_id}_geom.npz", "coords"))
                    save_table(angles, os.path.join(outdir, f"indv_{i_id}_geom.npz", "angles"))
                    save_table(labels, os.path.join(outdir, f"indv_{i_id}_geom.npz", "labels"))
                    made_files.add(i_id)

                run["magnet_angles"]=os.path.join(outdir, f"indv_{i_id}_geom.npz", "angles")
                run["magnet_coords"]=os.path.join(outdir, f"indv_{i_id}_geom.npz", "coords")
                run["labels"]=os.path.join(outdir, f"indv_{i_id}_geom.npz", "labels")

        return run_params

    def fit_func(ds):
        spin=better_read_tables(ds.tablefile("spin"), filter)

        state_num=[]
        for cell in range(total_spinices):
            if total_spinices > 1:
                cell_spin=spin[match_column(f"spin({cell},*)", spin)]
            else:
                cell_spin=spin[match_column("spin*", spin)]
            state_num.append(len(cell_spin.drop_duplicates()))

        if len(state_num) == 1:
            fitn=state_num[0]
        else:
            if fit_acc == "mode":
                mode_res=mode(state_num)
                fitn=mode_res.mode[0], 100 * mode_res.count[0] / len(state_num)
            elif fit_acc == "mean":
                fitn=np.mean(state_num), 100 / (np.std(state_num) + 1)
            else:
                raise ValueError("Unknown fit_acc")
            if polar_coords:
                fitn=pol2cart(fitn[1], np.pi * (1 - fitn[0] / max_state_count))
        return fitn

    pop=flatspin_eval(fit_func, pop, gen, outdir, sweep_params=sweep_params, group_by=group_by, preprocessing=preprocessing, **flatspin_kwargs)
    return pop


def ordered_statenumbers(arr):
    _, inv, counts=np.unique(arr, return_inverse=1, return_counts=1, axis=0)

    if len(counts) == 1:
        return inv
    res=np.zeros(len(inv))
    sort_counts=np.argsort(counts)[::-1]
    for i in range(len(counts)):
        res[inv == i]=np.where(sort_counts == i)[0][0]

    return res


@ ignore_empty_pop
def learn_function_fitness(pop, gen, outdir, t=-1, bit_len=3, sweep_params=None, group_by=None, tessellate_shape=None,
                    squint_grid_size=None, polar_coords=True, fit_acc="mode", function=None, **flatspin_kwargs):
    from scipy.stats import mode
    max_state_count=2**bit_len
    input=str([list(f"{i:b}".zfill(bit_len)) for i in range(max_state_count)])

    def strList2int(s):
        return int("".join([c for c in s if c not in "[], "]), 2)

    func_image_size=bit_len  # number of desired output states
    if type(function) == str and function.startswith("mod"):
        # convert list of bits (str) to int then mod 4
        mod_base=int(function[3:])

        def function(input):
            return strList2int(input) % mod_base

        # get all permutations of 1 2 3 ... mod_base
        perms=np.array(list(permutations(range(mod_base))))
        perms=np.tile(perms, (1, max_state_count // mod_base + 1))[:, :max_state_count]

    elif function == "prime":
        assert bit_len <= 4, "Prime function only implemented for bit_len < 5"

        def function(input):
            return int(input) in [2, 3, 5, 7, 11, 13]
        perms=np.array([[function(i) for i in range(16)], [not function(i) for i in range(16)]])

    if not sweep_params:
        sweep_params={}
    if "init" in sweep_params or "input" in sweep_params:
        warnings.warn("Overiding input in fitness function")
    sweep_params=dict(sweep_params, input=input)

    if not group_by:
        group_by=[]
    if "indv_id" not in group_by:
        group_by.append("indv_id")
    id2indv={individual.id: individual for individual in pop}
    nndist=flatspin_kwargs.get("neighbor_dist", pop[0].get_default_shared_params(select_param="neighbor_distance"))

    total_spinices=np.prod(tessellate_shape) if tessellate_shape else 1

    if t == -1:
        def filter(df):
            return df["t"] == df["t"].max()
    else:
        def filter(df):
            return df["t"] == t

    def preprocessing(run_params):
        if tessellate_shape is not None:
            # do tessellating
            made_files=set()
            for run in run_params:
                i_id=run["indv_id"]
                indv=id2indv[i_id]

                if i_id not in made_files:
                    pos, angles, labels=indv.fast_tessellate(tessellate_shape, padding=2 * nndist,
                        centre=False, return_labels=True)
                    save_table(pos, os.path.join(outdir, f"indv_{i_id}_geom.npz", "coords"))
                    save_table(angles, os.path.join(outdir, f"indv_{i_id}_geom.npz", "angles"))
                    save_table(labels, os.path.join(outdir, f"indv_{i_id}_geom.npz", "labels"))
                    made_files.add(i_id)

                run["magnet_angles"]=os.path.join(outdir, f"indv_{i_id}_geom.npz", "angles")
                run["magnet_coords"]=os.path.join(outdir, f"indv_{i_id}_geom.npz", "coords")
                run["labels"]=os.path.join(outdir, f"indv_{i_id}_geom.npz", "labels")

        return run_params

    def fit_func(ds):
        spin=better_read_tables(ds.tablefile("spin"), filter)
        nonlocal perms

        for cell in range(total_spinices):
            if total_spinices > 1:
                cell_spin=spin[match_column(f"spin({cell},*)", spin)]
            else:
                cell_spin=spin[match_column("spin*", spin)]
            states=ordered_statenumbers(cell_spin)
            fitn=np.abs(perms != states).sum(axis=1).min()
        """
        if len(state_num) == 1:
            fitn = state_num[0]
        else:
            pass # not implemented
        """
        return fitn

    pop=flatspin_eval(fit_func, pop, gen, outdir, sweep_params=sweep_params, group_by=group_by, preprocessing=preprocessing, **flatspin_kwargs)
    return pop


def pol2cart(r, theta):
    x=r * np.cos(theta)
    y=r * np.sin(theta)
    return [x, y]


def better_read_tables(filenames, filter=None, index_col=None):
    dfs=[read_table(f, index_col) for f in filenames]
    if filter is not None:
        dfs=[df[filter(df)] for df in dfs]

    table=dfs[0]
    for df in dfs[1:]:
        table=table.append(df)
    return table


def pheno_size_fitness(pop, gen, outdir, **flatspin_kwargs):
    id2indv={individual.id: individual for individual in pop}
    shared_params={"spp": 1, "periods": 1, "H": 0, "neighbor_distance": 1}

    def fit_func(ds):
        return len(id2indv[ds.index["indv_id"].values[0]].coords)

    pop=flatspin_eval(fit_func, pop, gen, outdir, condition=lambda x: True, shared_params=shared_params, **flatspin_kwargs)
    return pop


def directionality_fitness(pop, gen, outdir, tessellate_shape=(8, 8), pad=80, **flatspin_kwargs):
    id2indv={individual.id: individual for individual in pop}
    shared_params={"spp": 1, "periods": 1, "H": 0, "neighbor_distance": 1, "timesteps": 1, "alpha": 37839}
    from flatspin.model import CustomSpinIce

    def tess(indv):
        xpad, ypad=pad, pad
        pos, angles, labels=indv.fast_tessellate(tessellate_shape, padding=np.array([xpad, ypad]), centre=False, return_labels=True)
        return pos, angles, labels

    def total_h_par_dir(model):
        if model.cl:
            model._init_cl()  # TODO: fixme
        h_dip_perp=np.nan * np.zeros((model.spin_count, model.num_neighbors, 2))
        for i in model.indices():
            for jj, j in enumerate(model.neighbors(i)):
                rij=model.pos[i] - model.pos[j]
                rdir=rij / np.linalg.norm(rij)

                h_perp=rdir * np.abs(model._h_dip[i, jj, 1])
                h_dip_perp[i, jj]=h_perp

        return np.abs(np.nansum(h_dip_perp))

    id2pos={}
    id2angles={}
    id2labels={}

    def do_fit_func(id):
        indv=id2indv[id]
        pos, angles, labels=tess(indv)
        # print(pos)

        model=CustomSpinIce(magnet_coords=pos, magnet_angles=angles, labels=labels, neighbor_distance=np.inf, alpha=37839)
        return total_h_par_dir(model), (pos, angles, labels)

    def preprocessing(run_params):
        if tessellate_shape is not None:
            # do tessellating
            made_files=set()
            for run in run_params:
                i_id=run["indv_id"]

                run["magnet_angles"]=id2angles[i_id]
                run["magnet_coords"]=id2pos[i_id]
                run["labels"]=id2labels[i_id]

        return run_params

    id2fit={}
    progress_bar=ProgressBar(desc="calc fitness", total=len(id2indv))
    parallel=ParallelProgress(progress_bar, n_jobs=-1)

    def helper(id):
        fit, pal=do_fit_func(id)
        return id, (fit, pal)

    id2fit.update(parallel(delayed(helper)(id) for id in id2indv))
    progress_bar.close()
    # get the extra data out of id2fit
    for id in id2fit:
        id2pos[id], id2angles[id], id2labels[id]=id2fit[id][1]
        id2fit[id]=id2fit[id][0]

    def fit_func(ds):
        return id2fit[ds.index["indv_id"].values[0]]

    pop=flatspin_eval(fit_func, pop, gen, outdir, dont_run=False, condition=lambda x: True,
                        shared_params=shared_params, preprocessing=preprocessing, **flatspin_kwargs)
    return pop


def smile_fitness(pop, gen, outdir, min_mags=100, **flatspin_kwargs):
    def is_in_smiley(xy):
        x, y=xy[:, 0], xy[:, 1]
        left_eye=(x - 600)**2 + (y - 600)**2 < (400)**2
        right_eye=(x + 600)**2 + (y - 600)**2 < (400)**2
        mouth=np.logical_and(0.00015 * x**2 - 500 > y, y > 0.0005 * x**2 - 1500)
        return np.logical_or(np.logical_or(left_eye, right_eye), mouth)

    id2indv={individual.id: individual for individual in pop}
    shared_params={"spp": 1, "periods": 1, "H": 0, "neighbor_distance": 1}

    def fit_func(ds):
        return id2indv[ds.index["indv_id"].values[0]].fitness

    for indv in pop:
        if len(indv.coords) < min_mags:
            indv.fitness=np.nan
        else:
            indv.fitness=np.sum(is_in_smiley(indv.coords) * 2 - 1)

    pop=flatspin_eval(fit_func, pop, gen, outdir, condition=lambda x: True,
                        shared_params=shared_params, **flatspin_kwargs)
    return pop


def ca_rule_fitness(pop, gen, outdir, target, group_by=None, sweep_params=None, img_basepath="", compare="direct",
                    **flatspin_kwargs):
    from analyze_sweep import find_rule

    # \from ca_encoder import CARotateEncoder
    default_shared_params={
        "run": "local",
        "encoder": "ca_encoder.CARotateEncoder",
        "spp": 10,
        "periods": 1,
        "timesteps": 10,
        "basepath": os.path.join(outdir, f"gen{gen}"),
    }
    input="[[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]]"
    init=str(
        [
            os.path.join(img_basepath, "init_half_0.png"),
            os.path.join(img_basepath, "init_half_1.png"),
        ]
    )
    if not sweep_params:
        sweep_params={}
    if "init" in sweep_params or "input" in sweep_params:
        warnings.warn("Overiding 'init' and input in fitness function")
    sweep_params=dict(sweep_params, input=input, init=init)
    if not group_by:
        group_by=[]
    if "indv_id" not in group_by:
        group_by.append("indv_id")
    if "random_seed" in sweep_params and "random_seed" not in group_by:
        group_by.append("random_seed")

    if compare == "langton":
        langtons_table={
            x: "{0:08b}".format(x).count("1") / 8 for x in range(0, 256)
        }  # lambda[rule]
    elif compare == "equiv":
        from ca_rule_tools import eq_rules

        equiv_rules=list(filter(lambda x: target in x, eq_rules))[0]
    id2indv={individual.id: individual for individual in pop}

    def fit_func(ds):
        """takes a group of ds of same indv_id and seed (one full run of all ca inputs on a system)"""
        rule=find_rule((None, ds))[1]
        if compare == "langton":
            fitn=abs(langtons_table[rule] - langtons_table[target])
        elif compare == "equiv":
            fitn=int(rule in equiv_rules)
        else:  # direct compare
            fitn=int(rule == target)
        id=ds.index["indv_id"].values[0]
        indv=id2indv[id]
        indv.fitness_info=[] if indv.fitness_info is None else indv.fitness_info
        indv.fitness_info.append(f"rule {rule}")

        return fitn

    pop=flatspin_eval(fit_func, pop, gen, outdir, group_by=group_by, sweep_params=sweep_params, shared_params=default_shared_params,
                        do_not_override_default=True, **flatspin_kwargs)
    return pop


def stray_field_ca_fitness(pop, gen, outdir, sweep_params, grid_size=(5, 1), target="ClassIII,ClassIV", angle0=0,
                           angle1=np.pi, padding=80, group_by=None, **flatspin_kwargs):
    from ca_rule_tools import full_classes
    """target can be rule number: 93
    a equivelence class : 'eq93'
    or 1 or more classes: 'ClassIV' / 'ClassIII,ClassIV'
    """

    if "init_input" in sweep_params:
        warnings.warn("majority fitness function overwriting value of 'init_input'")

    if type(grid_size) in (int, float):
        grid_size=(grid_size, grid_size)
    sweep_params=dict(
        sweep_params, init_input=str(list(range(2**(grid_size[0] * grid_size[1]))))
    )

    if not group_by:
        group_by=[]
    if "indv_id" not in group_by:
        group_by.append("indv_id")

    # parse target
    if type(target) is not str or target.isnumeric():
        target=(int(target),)
    elif "eq" in target:
        target=(int(target[2:]),)
    elif "Class" in target:
        target=frozenset((rule for class_num in target for rule in full_classes[class_num[len("Class"):]]))

    id2indv={individual.id: individual for individual in pop}

    def preprocessing(run_params):
        """do tessalting"""
        if grid_size is not None:
            # do tessellating
            made_files=set()
            for run in run_params:
                i_id=run["indv_id"]
                indv=id2indv[i_id]

                if i_id not in made_files:
                    pos, angles, labels=indv.fast_tessellate(grid_size, padding=padding, centre=False, return_labels=True)
                    save_table(pos, os.path.join(outdir, f"indv_{i_id}_geom.npz", "coords"))
                    save_table(angles, os.path.join(outdir, f"indv_{i_id}_geom.npz", "angles"))
                    save_table(labels, os.path.join(outdir, f"indv_{i_id}_geom.npz", "labels"))
                    made_files.add(i_id)

                run["magnet_angles"]=os.path.join(outdir, f"indv_{i_id}_geom.npz", "angles")
                run["magnet_coords"]=os.path.join(outdir, f"indv_{i_id}_geom.npz", "coords")
                run["labels"]=os.path.join(outdir, f"indv_{i_id}_geom.npz", "labels")

        return run_params

    def fit_func(ds):  # need to group to check rule, or do something clever in outer eval
        rule_table=OrderedDict()
        for run in ds:
            mag=load_output(run, "mag", t=-1, grid_size=grid_size, flatten=False)
            state=mag[:, 1] > 0  # check y comp as we move join and stimulate in x
            init_input=int(run.index['init_input'])
            binput=bin(init_input)[2:]
            rule_table[binput]=state

        rules=check_ca_rules(rule_table)
        # find class
        # calc hamming dist i.e sum(xor())
    pop=flatspin_eval(fit_func, pop, gen, outdir, preprocessing=preprocessing,
                        sweep_params=sweep_params, **flatspin_kwargs)
    return pop


def check_ca_rules(rule_table):
    num_cells=len(rule_table.keys()[0])
    rules=[]
    for i in range(1, num_cells - 1):
        inputs=[k[i - 1:i + 1] for k in rule_table]
        outputs=[rule_table[k][i] for k in rule_table]
        rules.append(check_ca_rule(inputs, outputs))
    return rules


def check_ca_rule(inputs, outputs):
    [y for _, y in sorted(zip(map(int, inputs), outputs), reverse=1)]
    sorted_outs=[outputs[i] for i in sorted(range(len(outputs)), key=lambda x: int(inputs[x]), reverse=1)]
    rule_num=int("".join(sorted_outs), 2)
    return rule_num


def std_grid_field_fitness(pop, gen, outdir, angles=np.linspace(0, 2 * np.pi, 8), grid_size=4, **flatspin_kwargs):
    shared_params={}
    shared_params["phi"]=360
    shared_params["input"]=(angles % (2 * np.pi)) / (2 * np.pi)

    if np.isscalar(grid_size):
        grid_size=(grid_size, grid_size)

        def fit_func(ds):
            mag=load_output(ds, "mag", t=ds.params["spp"], grid_size=grid_size, flatten=False)
            magnitude=np.linalg.norm(mag, axis=3)
            summ=np.sum(magnitude, axis=0)
            fitn=np.std(summ) * np.mean(summ)
            return fitn

    pop=flatspin_eval(
        fit_func, pop, gen, outdir, shared_params=shared_params, **flatspin_kwargs
    )
    return pop


def get_range(a):
    mn, mx=minmax(a)
    return mx - mn


def minmax(a):
    a=np.array(a) if type(a) != np.ndarray else a
    if a.ndim > 1:
        a=a.reshape(-1, a.shape[-1])

    return a.min(axis=0), a.max(axis=0)


def target_order_percent_fitness(pop, gen, outdir, grid_size=4, threshold=0.5, condition=None, **flatspin_kwargs):
    if np.isscalar(grid_size):
        grid_size=(grid_size, grid_size)

    for i in pop:
        i.grid=Grid.fixed_grid(np.array(i.coords), grid_size)

    id2indv={individual.id: individual for individual in pop}

    def fit_func(ds):
        mag=load_output(ds, "mag", t=-1, grid_size=grid_size, flatten=False)
        magnitude=np.linalg.norm(mag, axis=3)[0]
        indv=id2indv[ds.index["indv_id"].values[0]]
        cells_with_mags=[(x, y) for x, y in np.unique(indv.grid._grid_index, axis=0)]
        # old
        """
        # fitness is std of the magnitudes of the cells minus std of the number of magnets in each cell
        fitn = np.std([magnitude[x][y] for x, y in cells_with_mags]) - \
               np.std([len(indv.grid.point_index([x, y]))
                      for x, y in cells_with_mags])
        """
        fitn=abs(((np.array([magnitude[x][y] for x, y in cells_with_mags]) < threshold) * 2 - 1).sum())

        return fitn

    pop=flatspin_eval(
        fit_func, pop, gen, outdir, condition=condition, **flatspin_kwargs
    )
    return pop


known_fits={
    "target_state_num": target_state_num_fitness,
    "state_num": state_num_fitness,
    "flips": flips_fitness,
    "std_grid_field": std_grid_field_fitness,
    "target_order_percent": target_order_percent_fitness,
    "default": evaluate_outer,
    "find_all": evaluate_outer_find_all,
    "pheno_size": pheno_size_fitness,
    "image": image_match_fitness,
    "mem_capacity": mem_capacity_fitness,
    "parity": parity_fitness,
    "majority": majority_fitness,
    "correlation": correlation_fitness,
    "xor": xor_fitness,
    "ca_rule": ca_rule_fitness,
    "state_num2": state_num_fitness2,
    "novelty": evaluate_outer_novelty_search,
    "smile": smile_fitness,
    "learn_func": learn_function_fitness,
    "direction": directionality_fitness,
    "simple_flips": simple_flips_fitness,
    "music": music_fitness,
}
