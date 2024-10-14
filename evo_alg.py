# vim: tw=120
from functools import wraps
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

from flatspin.data import Dataset, read_csv

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


def nan_select_filter(func):
    @wraps(func)
    def wrapper(pop, pop_size, *args, **kwargs):
        nan_pop, fin_pop = [], []
        for indv in pop:
            (fin_pop if np.isfinite(indv.fitness) else nan_pop).append(indv)

        if fin_pop:
            pop = func(fin_pop, pop_size, *args, **kwargs)
        else:
            pop = []

        if len(pop) < pop_size:
            pop += list(np.random.choice(nan_pop, min(pop_size - len(pop), len(nan_pop)), replace=False))

        return pop
    return wrapper


@nan_select_filter
def fittest_select(pop, pop_size, minimize_fit):
    fits = np.fromiter(map(lambda p: p.fitness, pop), count=len(pop), dtype=np.float64)

    if minimize_fit:
        fits = -1 * fits
    best_indices = np.argpartition(fits, -pop_size)[-pop_size:]
    new_pop = [pop[i] for i in best_indices]

    return new_pop


@nan_select_filter
def tournament_select(pop, pop_size, tournament_size=7, elitism=False, minimize_fit=True):
    if len(pop) < pop_size:
        return pop
    new_pop = []
    if elitism:
        best = min(pop, key=lambda indv: indv.fitness) if minimize_fit else max(pop, key=lambda indv: indv.fitness)
        new_pop.append(best)
        pop.remove(best)

    for _ in range(pop_size - len(new_pop)):
        if len(pop) < tournament_size:
            if len(pop) == 0:
                break
            best = min(pop, key=lambda indv: indv.fitness) if minimize_fit else max(pop, key=lambda indv: indv.fitness)
        else:
            tournament = np.random.choice(pop, tournament_size, replace=False)
            if minimize_fit:
                best = min(tournament, key=lambda indv: indv.fitness)
            else:
                best = max(tournament, key=lambda indv: indv.fitness)
            new_pop.append(best)
            pop.remove(best)
    return new_pop

def do_tournament_multi(tourn_pop, tourn_size, n_components, minimize_fit, randomize_order):
    # len(tourn_pop) should be n_components**tourn_size, pad with Nones
    comps = range(n_components)
    if randomize_order:
        comps = list(comps)
        np.random.shuffle(comps)

    losers = []
    for comp in comps:
        victors =[]
        for i in range(0, len(tourn_pop), tourn_size):
            tournament = tourn_pop[i:i+tourn_size]
            best, rest = winner(tournament, minimize_fit, comp)
            losers += rest
            victors.append(best)
        tourn_pop = victors

    assert len(victors) == 1
    return victors[0], losers

def winner(indvs, minimize_fit, component):
    # correctly accepts Nones, but doesn't return any as losers, only as a winner if all None
    tourn = [i for i in indvs if i!=None]
    if len(tourn) ==0:
        return indvs[0], []

    agg = min if minimize_fit else max
    idxs = list(range(len(tourn)))
    best_i = agg(idxs, key=lambda i: tourn[i].fitness_components[component])

    return tourn[best_i], tourn[:best_i] + tourn[best_i+1:]


@nan_select_filter
def multi_tournament_select(pop, pop_size, tournament_size=2, elitism=False, minimize_fit=True, randomise_order=False, num_select_objectives=None):
    if elitism:
        warn("elitism is undefined for multi-object tournament, has no effect")

    if num_select_objectives is None:
        num_select_objectives = len(pop[0].fitness_components)
    if len(pop) < pop_size:
        return pop

    pop = pop.copy()
    new_pop = []
    rejects = []

    full_tournament_size = tournament_size**num_select_objectives
    while len(new_pop) < pop_size:
        tournament = try_choice(pop, full_tournament_size, pad=True, replace=False)
        best, rest = do_tournament_multi(tournament, tournament_size, num_select_objectives, minimize_fit, randomise_order)

        pop.remove(best) # remove the best n rest
        pop = [p for p in pop if p not in rest]

        new_pop.append(best)
        rejects += rest
        if len(pop)==0:
            np.random.shuffle(rejects)
            pop, rejects = rejects, pop



    return new_pop

def try_choice(arr, size, pad=False, replace=False):
    s = size
    if not replace:
        s = min(s, len(arr))
    chosen = np.random.choice(arr, s, replace=replace)

    if pad and  s < size:
        chosen = np.concatenate((chosen, [None] * (size - s)))

    return chosen

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


def update_superdataset(dataset, outdir, pop, gen, minimize_fitness=True, dataset_params=None):
    #pop = list(filter(lambda indv: np.isfinite(indv.fitness), pop))

    if len(pop) < 1:
        return
    dataset_params = dataset_params or []
    best = pop[0]
    if len(pop) > 1:
        fn = min if minimize_fitness else max
        best = fn(pop, key=lambda indv: indv.fitness)

    for indv in pop:
        ind = dataset.index
        if "indv_id" in ind.columns and indv.id in ind["indv_id"].values:
            copy_row = ind[ind["indv_id"] == indv.id].iloc[:1].copy() # copy the row, use :1 range to keep as dataframe
            copy_row["gen"] = gen
            copy_row["fitness"] = indv.fitness
            copy_row["best"] = int(indv == best)
            #dataset.index = ind.append(copy_row, ignore_index=True)
            dataset.index = pd.concat([dataset.index, copy_row], ignore_index=True)
        else:
            ds = Dataset.read(os.path.join(outdir, f"gen{indv.gen}"))
            ds = ds.filter(indv_id=indv.id)
            ind = ds.index
            ind.insert(0, 'gen', gen)  # current generation
            ind.insert(2, 'fitness', indv.fitness)
            ind.insert(3, 'best', int(indv == best))
            for i, param in enumerate(dataset_params):
                ind.insert(4 + i, param, [getattr(indv, param)] * len(ds.index))  #  multiply for when group-by causes copied rows

            # patch outdir
            ind['outdir'] = ind['outdir'].apply(lambda o: os.path.join(f"gen{indv.gen}", o))
            to_drop = [col for col in ['magnet_coords', 'magnet_angles', 'labels'] if col in ind]
            ind.drop(columns=to_drop, inplace=True)  # debug
            # fitness_componenets should be added last due to variable column number
            for i, comp in enumerate(indv.fitness_components):
                ind.insert(len(ind.columns), f"fitness_component{i}", comp)
            #dataset.index = dataset.index.append(ind)
            dataset.index = pd.concat([dataset.index, ind], ignore_index=True)
        if not dataset.params:
            dataset.params = ds.params


def save_snapshot(outdir, pop):
    with open(os.path.join(outdir, "snapshot.pkl"), "wb") as f:
        pkl.dump([repr(indv) for indv in pop], f)


def improvement_rate(mutant_pop, dataset, minimize_fitness=True):
    if len(mutant_pop) < 1:
        return -1
    ds = dataset.index.drop_duplicates(subset=['indv_id'])
    kid_fit = [indv.fitness for indv in mutant_pop]
    parent_fit = []

    for indv in mutant_pop:
        if indv.parent_ids[0] in ds["indv_id"].values:
            parent_fit.append(ds[ds["indv_id"] == indv.parent_ids[0]]["fitness"].values[0])
        else:
            parent_fit.append(np.nan)

    better = [kf <= pf if minimize_fitness else kf >= pf for kf, pf in zip(kid_fit, parent_fit) if not np.isnan(kf) and not np.isnan(pf)]
    if len(better) < 1:
        return -1
    return sum(better) / len(better)


def only_run_fitness_func(outdir, individual_class, evaluate_inner, evaluate_outer, minimize_fitness=True,
        *, individual_params={}, outer_eval_params={}, sweep_params=OrderedDict(), group_by=None, starting_pop=None,
        keep_id=False, dataset_params=None, **kwargs):

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

    pop = [individual_class.from_string(i, gen=0, keep_pheno=True) for i in starting_pop]
    if not keep_id:
        for i, indv in enumerate(pop):
            indv.id = i

    evaluate_inner(pop, 0, outdir, sweep_params=sweep_params, group_by=group_by, **kwargs)
    evaluate_outer(pop, basepath=outdir, gen=0, **outer_eval_params)

    index = pd.DataFrame()
    params = None  # to be added by update_superdataset
    info = {'command': ' '.join(map(shlex.quote, sys.argv)), }
    dataset = Dataset(index, params, info, basepath=outdir)

    update_superdataset(dataset, outdir, pop, 0, minimize_fitness, dataset_params=dataset_params)
    dataset.save()


def parse_starting_pop(starting_pop, individual_class):
    try:
        with open(starting_pop, "r") as f:
            starting_pop = f.read().splitlines()
    except Exception:
        pass
    pop = [individual_class.from_string(i, id=None, gen=0, keep_pheno=True) for i in starting_pop]
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
    super_index = read_csv(os.path.join(outdir, "index.csv"))
    max_id = np.concatenate((newest_index["indv_id"].values, super_index["indv_id"].values)).max()
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


def crossover(pop, cx_prob, parent_list=None):
    parent_list = parent_list if parent_list is not None else []
    kids_list = []
    for i, indv in enumerate(pop):  # TODO: replace with itertools combination or likewise
        if np.random.rand() < cx_prob:
            partner = np.random.choice(pop)  # can partner with itself, resulting in perfect copy
            cross_result = indv.crossover(partner)
            parent_list.extend([indv, partner])
            kids_list.extend(cross_result)
    return parent_list, kids_list


def random_search_main(outdir, individual_class, evaluate_inner, evaluate_outer, minimize_fitness, individual_params={},
         outer_eval_params={}, sweep_params=OrderedDict(), dependent_params={}, group_by=None, *, n_evals=100, n_batches=10, dataset_params=None, **kwargs):
    print("Initialising")
    # create superdataset
    index = pd.DataFrame()
    info = {'command': ' '.join(map(shlex.quote, sys.argv)), }
    dataset = Dataset(index, None, info, basepath=outdir)

    evals_per_batch = n_evals // n_batches
    remaining_evals = n_evals % n_batches
    for batch in range(n_batches):
        print(f"Batch {batch+1} / {n_batches}")
        pop = [individual_class(gen=batch, **individual_params) for _ in range(evals_per_batch)]
        if batch == 0:
            pop += [individual_class(gen=batch, **individual_params) for _ in range(remaining_evals)]

        evaluate_inner(pop, batch, outdir, sweep_params=sweep_params, group_by=group_by, dependent_params=dependent_params, **kwargs)
        evaluate_outer(pop, basepath=outdir, gen=batch, **outer_eval_params)

        update_superdataset(dataset, outdir, pop, batch, minimize_fitness, dataset_params=dataset_params)
        dataset.save()


def main(outdir, individual_class, evaluate_inner, evaluate_outer, minimize_fitness=True, *,
         pop_size=100, generation_num=100, mut_prob=0.2, cx_prob=0.3,
         mut_strength=1, reval_inner=False, elitism=False, individual_params={},
         outer_eval_params={}, evolved_params={}, sweep_params=OrderedDict(), dependent_params={},
         stop_at_fitness=None, group_by=None, select_randomise_order=False,
         starting_pop=None, continue_run=False, starting_gen=1, select="best", mutate_strategy=0, keep_parents=True,
         random_search=False, dataset_params=None, **kwargs):

    print("Initialising")
    main_check_args(individual_params, evolved_params, sweep_params, kwargs)

    assert os.path.isdir(outdir) or not continue_run, "can't continue run without existing outdir"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    setup_evolved_params(evolved_params, individual_class)

    if random_search:
        random_search_main(outdir, individual_class, evaluate_inner, evaluate_outer, minimize_fitness, individual_params, outer_eval_params, sweep_params, dependent_params, group_by, **kwargs)
        return

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
        info = {'command': ' '.join(map(shlex.quote, sys.argv)), }
        dataset = Dataset(index, None, info, basepath=outdir)

        update_superdataset(dataset, outdir, pop, 0, minimize_fitness, dataset_params)
        dataset.save()

    gen_times = []
    best = None
    for gen in range(starting_gen, generation_num + 1):
        print(f"starting gen {gen} of {generation_num}")
        if len(gen_times) > 0:
            tr = np.mean(gen_times[-max(10, len(gen_times) // 10):]) * (generation_num - gen)
            print(f"~{np.round(tr / 3600, 2)} hours remaining" if tr > 3600 else (
                f"~{np.round(tr / 60, 2)} minutes remaining" if tr > 60 else f"~{np.round(tr, 2)} seconds remaining"))
        time = datetime.now()

        parent_list = []  # track parents in case we want to remove them
        # Mutate!
        print("    Mutate")
        mut_kids = []
        for indv in pop:
            if np.random.rand() < mut_prob:
                mut_kids += indv.mutate(mut_strength)
                parent_list.append(indv)

        # Crossover!
        print("    Crossover")
        parent_list, crossover_kids = crossover(pop, cx_prob, parent_list)

        if not keep_parents:
            for parent in parent_list:
                try:
                    pop.remove(parent)
                except ValueError:
                    pass
            if elitism and best and best not in pop:
                pop.append(best)

        for indv in mut_kids + crossover_kids:
            indv.gen = gen

        # Eval
        print("    Evaluate")
        if reval_inner:  # do we re-evealuate all inner fitnesses?
            pop.extend(mut_kids + crossover_kids)
            list(map(individual_class.clear_fitness, pop))  # clear fitness and fitness componenets
            evaluate_inner(pop, gen, outdir, sweep_params=sweep_params, group_by=group_by, dependent_params=dependent_params, **kwargs)
        else:
            evaluate_inner(mut_kids + crossover_kids, gen, outdir, sweep_params=sweep_params, group_by=group_by, dependent_params=dependent_params, **kwargs)
            pop.extend(mut_kids + crossover_kids)
        evaluate_outer(pop, basepath=outdir, gen=gen, **outer_eval_params)

        update_superdataset(dataset, outdir, pop, gen, minimize_fitness, dataset_params)
        dataset.save()

        if mutate_strategy and len(mut_kids) > 0:
            improve_rate = improvement_rate(mut_kids, dataset, minimize_fitness)
            if improve_rate >= 0:
                new_mut_strength = (mut_strength + mutate_strategy) if improve_rate > 0.2 else (mut_strength - mutate_strategy)
                new_mut_strength = np.round(new_mut_strength, 3)
                if new_mut_strength > 0:
                    print(f"improvment rate: {improve_rate}, updating mut_strength {mut_strength} -> {new_mut_strength}")
                    mut_strength = new_mut_strength
                else:
                    print(f"improvment rate: {improve_rate}, mut_strength is already at minimum ({mut_strength})")
        # Select
        print("    Select")
        if select == "best":
            pop = fittest_select(pop, pop_size, minimize_fitness)
        elif select == "roulette":
            pop = roulette_select(pop, pop_size, elitism, minimize_fitness)
        elif select == "tournament":
            pop = tournament_select(pop, pop_size, elitism=elitism, minimize_fit=minimize_fitness)
        elif select == "multi_tourn":
            pop = multi_tournament_select(pop, pop_size, elitism=elitism, minimize_fit=minimize_fitness, randomise_order=select_randomise_order)
        else:
            raise ValueError(f"select '{select}' not recognised, choose from 'best', 'roulette', 'tournament' or 'multi_tourn'")

        assert len(pop) <= pop_size, f"pop size {len(pop)} > {pop_size}"

        best = save_stats(outdir, pop, minimize_fitness)
        print(f"best fitness: {best.fitness}")
        if len(best.fitness_components) > 1:
            print(f"    with fitness components: {best.fitness_components}")
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

