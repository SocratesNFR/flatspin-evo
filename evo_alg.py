# vim: tw=120
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from scipy.stats import zscore

from collections import OrderedDict
import shlex
import copy
import os
import sys

from warnings import warn

from flatspin.runner import run, run_dist, run_local
from flatspin.data import Dataset, is_archive_format, read_csv
from flatspin.utils import get_default_params, import_class
from flatspin.sweep import sweep

import pickle as pkl


def rainbow_colours(num):
    cmap = plt.get_cmap('gist_rainbow')
    return cmap(np.linspace(0, 1, num))


def roulette_select(pop, pop_size, elitism=False, minimize_fit=True):
    fits = np.fromiter(map(lambda p: p.fitness, pop), count=len(pop), dtype=np.float64)

    is_finite = np.isfinite(fits)
    fits = fits[is_finite]
    nan_pop = [pop[i] for i in np.arange(len(pop))[~is_finite]]
    pop = [pop[i] for i in np.arange(len(pop))[is_finite]]

    if len(pop) < pop_size:
        # if not enough to select from: return all indvs with none nan fitness and select remainder randomly
        if len(nan_pop) > 0:
            pop += list(np.random.choice(nan_pop, pop_size - len(pop), replace=False))
        return pop

    if minimize_fit:
        fits = -1 * fits

    if elitism:
        best_index = np.argmax(fits)
        best = pop.pop(best_index)  # remove best to be added in at end
        if pop_size == 1:
            return [best]
        fits = np.delete(fits, best_index)

    zs = zscore(fits)
    fits = zs if not np.isnan(zs).any() else fits
    fits += np.abs(np.min(fits)) + 1
    summ = np.sum(fits)
    fits = fits / summ if summ != 0 else fits
    assert not np.isnan(fits).any()

    # print(fits)
    if elitism:
        new_pop = list(np.random.choice(pop, pop_size - 1, replace=False, p=fits)) + [best]
    else:
        new_pop = list(np.random.choice(pop, pop_size, replace=False, p=fits))

    return new_pop


def fittest_select(pop, pop_size, minimize_fit):
    fits = np.fromiter(map(lambda p: p.fitness, pop), count=len(pop), dtype=np.float64)

    is_finite = np.isfinite(fits)
    fits = fits[is_finite]
    nan_pop = [pop[i] for i in np.arange(len(pop))[~is_finite]]
    pop = [pop[i] for i in np.arange(len(pop))[is_finite]]
    if len(pop) < pop_size:
        # if not enough to select from: return all indvs with none nan fitness and select remainder randomly
        pop += list(np.random.choice(nan_pop, pop_size - len(pop), replace=False))
        return pop

    if minimize_fit:
        fits = -1 * fits
    best_indices = np.argpartition(fits, -pop_size)[-pop_size:]
    new_pop = [pop[i] for i in best_indices]

    return new_pop


def parse_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    # assert len(lines) % 3 == 0
    results = []

    for gen in range(0, len(lines)):
        d = {}
        """
        d = {k.strip(): v.strip() for k, v in [keyval.split(":") for keyval in lines[gen].split(";")]}
        for k in d:
            if "," in d[k]:
                d[k] = float(d[k].strip("][").split(","))
            else:
                d[k] = float(d[k].strip("]["))
        """
        d = lines[gen].strip()
        results.append(d)
    return results


def save_stats(outdir, pop, minimize_fitness):
    fits = np.array(list(map(lambda indv: indv.fitness, pop)))
    fits = fits[np.isfinite(fits)]
    finite_pop = [pop[i] for i in np.where(np.isfinite(fits))[0]]
    out = []
    if len(finite_pop) > 0:
        best = min(finite_pop, key=lambda indv: indv.fitness) if minimize_fitness else max(
            finite_pop, key=lambda indv: indv.fitness)
    else:
        best = pop[0]
    out.extend(repr(best))
    out.append("\n")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with(open(os.path.join(outdir, "stats.txt"), "a+")) as f:
        f.writelines(out)
    return best


def update_superdataset(dataset, outdir, pop, gen, minimize_fitness=True):
    pop = list(filter(lambda indv: np.isfinite(indv.fitness), pop))

    if len(pop) < 1:
        return

    best = pop[0]
    if len(pop) > 1:
        fn = min if minimize_fitness else max
        best = fn(pop, key=lambda indv: indv.fitness)

    for indv in pop:
        ind = dataset.index
        if "indv_id" in ind.columns and indv.id in ind["indv_id"].values:
            copy_row = ind[ind["indv_id"] == indv.id].iloc[0].copy()
            copy_row["gen"] = gen
            copy_row["fitness"] = indv.fitness
            copy_row["best"] = int(indv == best)
            dataset.index = ind.append(copy_row, ignore_index=True)
        else:
            ds = Dataset.read(os.path.join(outdir, f"gen{indv.gen}"))
            ds = ds.filter(indv_id=indv.id)
            ind = ds.index
            ind.insert(0, 'gen', gen)  # current generation
            ind.insert(2, 'fitness', indv.fitness)
            ind.insert(3, 'best', int(indv == best))

            # patch outdir
            ind['outdir'] = ind['outdir'].apply(lambda o: os.path.join(f"gen{indv.gen}", o))
            to_drop = [col for col in ['magnet_coords', 'magnet_angles', 'labels'] if col in ind]
            ind.drop(columns=to_drop, inplace=True)  # debug
            # fitness_componenets should be added last due to variable column number
            for i, comp in enumerate(indv.fitness_components):
                ind.insert(len(ind.columns), f"fitness_component{i}", comp)
            dataset.index = dataset.index.append(ind)

        if not dataset.params:
            dataset.params = ds.params


def save_snapshot(outdir, pop):
    with open(os.path.join(outdir, "snapshot.pkl"), "wb") as f:
        pkl.dump([repr(indv) for indv in pop], f)


def only_run_fitness_func(outdir, individual_class, evaluate_inner, evaluate_outer, minimize_fitness=True,
        *, individual_params={}, outer_eval_params={}, sweep_params=OrderedDict(), group_by=None, starting_pop=None,
        keep_id=False, **kwargs):

    check_args = np.unique(list(kwargs) + list(sweep_params), return_counts=True)
    check_args = [check_args[0][i] for i in range(len(check_args[0])) if check_args[1][i] > 1]
    if check_args:
        raise RuntimeError(f"param '{check_args[0]}' appears in multiple param groups")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if "model" in kwargs and kwargs["model"] != "CustomSpinIce":
        individual_params["fixed_geom"] = True
    elif ("magnet_angles" in kwargs or " magnet_coords" in kwargs) and not individual_params.get("fixed_geom", False):
        warn("""You suppplied magnet_angles or magnet_coords, but 'fixed_geom' is False,
                set -i fixed_geom=True if you do not want to evolve the geometry""")

    assert starting_pop, "starting population is required"
    try:  # try to read starting pop as a file, else assume it's a list of strings
        with open(starting_pop, "r") as f:
            starting_pop = f.read().splitlines()
    except Exception:
        pass

    pop = [individual_class.from_string(i, gen=0) for i in starting_pop]
    if not keep_id:
        for i, indv in enumerate(pop):
            indv.id = i

    evaluate_inner(pop, 0, outdir, sweep_params=sweep_params, group_by=group_by, **kwargs)
    evaluate_outer(pop, basepath=outdir, gen=0, **outer_eval_params)

    index = pd.DataFrame()
    params = None  # to be added by update_superdataset
    info = {'command': ' '.join(map(shlex.quote, sys.argv)), }
    dataset = Dataset(index, params, info, basepath=outdir)

    update_superdataset(dataset, outdir, pop, 0, minimize_fitness)
    dataset.save()


def parse_starting_pop(starting_pop, individual_class):
    try:
        with open(starting_pop, "r") as f:
            starting_pop = f.read().splitlines()
    except Exception:
        pass
    pop = [individual_class.from_string(i, id=None, gen=0) for i in starting_pop]
    return pop


def setup_continue_run(outdir, individual_class, start_gen):
    with open(os.path.join(outdir, "snapshot.pkl"), "rb") as f:
        pop = list(map(lambda i: individual_class.from_string(i), pkl.load(f)))

    dataset = Dataset.read(outdir)

    assert os.path.isdir(os.path.join(outdir, f"gen{start_gen-1}")), f"gen{start_gen-1} does not exist"

    if os.path.isdir(os.path.join(outdir, f"gen{start_gen}")):
        # rename last gen so not overwritten
        gen_string = f"gen{start_gen}"
        os.rename(os.path.join(outdir, gen_string), os.path.join(outdir, "old_" + gen_string))
        gen_string = "old_" + gen_string
    else:
        gen_string = f"gen{start_gen-1}"

    # find the largest id in the dataset so we dont overwrite
    newest_index = read_csv(os.path.join(outdir, gen_string, "index.csv"))
    max_id = max(newest_index["indv_id"].values)
    individual_class.set_id_start(max_id + 1)

    return pop, dataset


def main_check_args(individual_params, evolved_params, sweep_params, kwargs):
    check_args = np.unique(list(evolved_params) + list(kwargs) + list(sweep_params), return_counts=True)
    check_args = [check_args[0][i] for i in range(len(check_args[0])) if check_args[1][i] > 1]
    if check_args:
        raise RuntimeError(f"param '{check_args[0]}' appears in multiple param groups")

    # hacks to allow fixed geoms
    if "model" in kwargs and kwargs["model"] != "CustomSpinIce":
        individual_params["fixed_geom"] = True
    elif ("magnet_angles" in kwargs or " magnet_coords" in kwargs) and not individual_params.get("fixed_geom", False):
        warn("""You suppplied magnet_angles or magnet_coords, but 'fixed_geom' is False,
                set -i fixed_geom=True if you do not want to evolve the geometry""")


def setup_evolved_params(evolved_params, individual_class):
    for evo_param in evolved_params:
        evolved_params[evo_param] = {"low": evolved_params[evo_param][0],
                                    "high": evolved_params[evo_param][1],
                                    "shape": (evolved_params[evo_param][2:] if len(evolved_params[evo_param]) > 2 else None)}
    individual_class.set_evolved_params(evolved_params)


def main(outdir, individual_class, evaluate_inner, evaluate_outer, minimize_fitness=True, *,
         pop_size=100, generation_num=100, mut_prob=0.2, cx_prob=0.3,
         mut_strength=1, reval_inner=False, elitism=False, individual_params={},
         outer_eval_params={}, evolved_params={}, sweep_params=OrderedDict(), dependent_params={},
         stop_at_fitness=None, group_by=None,
         starting_pop=None, continue_run=False, starting_gen=1, **kwargs):

    print("Initialising")
    main_check_args(individual_params, evolved_params, sweep_params, kwargs)

    assert os.path.isdir(outdir) or not continue_run, "can't continue run without existing outdir"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    setup_evolved_params(evolved_params, individual_class)

    if continue_run:
        pop, dataset = setup_continue_run(outdir, individual_class, starting_gen)

    else:
        if starting_pop:
            pop = parse_starting_pop(starting_pop, individual_class)
        else:
            pop = [individual_class(**individual_params) for _ in range(pop_size)]
        evaluate_inner(pop, 0, outdir, sweep_params=sweep_params, group_by=group_by, dependent_params=dependent_params, **kwargs)
        evaluate_outer(pop, basepath=outdir, gen=0, **outer_eval_params)
        # create superdataset
        index = pd.DataFrame()
        params = None  # to be added by update_superdataset
        info = {'command': ' '.join(map(shlex.quote, sys.argv)), }
        dataset = Dataset(index, params, info, basepath=outdir)

        update_superdataset(dataset, outdir, pop, 0, minimize_fitness)
        dataset.save()

    gen_times = []
    for gen in range(starting_gen, generation_num + 1):
        print(f"starting gen {gen} of {generation_num}")
        if len(gen_times) > 0:
            tr = np.mean(gen_times[-max(10, len(gen_times) // 10):]) * (generation_num - gen)
            print(f"~{np.round(tr / 3600, 2)} hours remaining" if tr > 3600 else (
                f"~{np.round(tr / 60, 2)} minutes remaining" if tr > 60 else f"~{np.round(tr, 2)} seconds remaining"))
        time = datetime.now()

        # Mutate!
        print("    Mutate")
        new_kids = []
        for indv in pop:
            if np.random.rand() < mut_prob:
                new_kids += indv.mutate(mut_strength)

        # Crossover!
        print("    Crossover")
        for i, indv in enumerate(pop):  # TODO: replace with itertools combination or likewise
            if np.random.rand() < cx_prob:
                partner = np.random.choice(pop)  # can partner with itself, resulting in perfect copy
                new_kids += indv.crossover(partner)

        for indv in new_kids:
            indv.gen = gen

        # Eval
        print("    Evaluate")
        if reval_inner:  # do we re-evealuate all inner fitnesses?
            pop.extend(new_kids)
            list(map(individual_class.clear_fitness, pop))  # clear fitness and fitness componenets
            evaluate_inner(pop, gen, outdir, sweep_params=sweep_params, group_by=group_by, dependent_params=dependent_params, **kwargs)
        else:
            evaluate_inner(new_kids, gen, outdir, sweep_params=sweep_params, group_by=group_by, dependent_params=dependent_params, **kwargs)
            pop.extend(new_kids)
        evaluate_outer(pop, basepath=outdir, gen=gen, **outer_eval_params)

        update_superdataset(dataset, outdir, pop, gen, minimize_fitness)
        dataset.save()

        # Select
        print("    Select")
        pop = roulette_select(pop, pop_size, elitism, minimize_fitness)
        # pop = fittestSelect(pop, popSize)
        assert len(pop) <= pop_size

        best = save_stats(outdir, pop, minimize_fitness)
        print(f"best fitness: {best.fitness}")
        save_snapshot(outdir, pop)
        if stop_at_fitness is not None and np.isfinite(best.fitness) and (
                (minimize_fitness and best.fitness <= stop_at_fitness) or ((not minimize_fitness) and best.fitness >= stop_at_fitness)
        ):
            print(f"Halting early, fitness {best.fitness} achieved")
            print(stop_at_fitness is not None, minimize_fitness and best.fitness <= stop_at_fitness,
                  (not minimize_fitness) and best.fitness >= stop_at_fitness)
            return best
        gen_times.append((datetime.now() - time).total_seconds())
    # best.plot()
    return best
