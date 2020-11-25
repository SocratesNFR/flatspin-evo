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

from flatspin.runner import run, run_dist, run_local
from flatspin.data import Dataset, is_archive_format
from flatspin.utils import get_default_params, import_class


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
    assert len(lines) % 3 == 0
    results = []
    for gen in range(0, len(lines), 3):
        d = {k.strip(): v.strip() for k, v in [keyval.split(":") for keyval in lines[gen].split(";")]}
        for k in d:
            if "," in d[k]:
                d[k] = float(d[k].strip("][").split(","))
            else:
                d[k] = float(d[k].strip("]["))
        d["bestIndv"] = lines[gen + 2].strip()
        results.append(d)
    return results


def save_stats(outdir, pop, minimize_fitness):
    fits = np.array(list(map(lambda indv: indv.fitness, pop)))
    fits = fits[np.isfinite(fits)]
    finite_pop = [pop[i] for i in np.where(np.isfinite(fits))[0]]
    out = []
    if len(finite_pop) > 0:
        best = min(finite_pop, key=lambda indv: indv.fitness) if minimize_fitness else max(finite_pop, key=lambda
            indv: indv.fitness)
    else:
        best = pop[0]
    out.extend(repr(best))
    out.append("\n")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with(open(os.path.join(outdir, "stats.txt"), "a+")) as f:
        f.writelines(out)
    return best


def top_of_the_pops(result, individual_class, interval=400, compress=True):
    frames = []
    titles = []
    prev_id = None
    gen_i = 0
    for indv in [gen["bestIndv"] for gen in result]:
        parsedIndv = individual_class.from_string(indv)
        if not compress or (prev_id is None or prev_id != parsedIndv.id):
            frames.append(list(map(lambda m: m.as_patch(), parsedIndv.pheno)))
            titles.append(f"id = {parsedIndv.id}; gen = {gen_i}; fit = {parsedIndv.fitness}")
            prev_id = parsedIndv.id
        gen_i += 1
    return individual_class.frames2animation(frames, interval, title=titles)


def update_superdataset(dataset, outdir, pop, gen, minimize_fitness=True):
    pop = list(filter(lambda indv: np.isfinite(indv.fitness), pop))

    if len(pop) < 1:
        return

    best = pop[0]
    if len(pop) > 1:
        fn = min if minimize_fitness else max
        best = fn(pop, key=lambda indv: indv.fitness)

    for indv in pop:
        ds = Dataset.read(os.path.join(outdir, f"gen{indv.gen}"))
        ds = ds.filter(indv_id=indv.id)
        ind = ds.index
        ind.insert(0, 'gen', gen)  # current generation
        ind.insert(2, 'fitness', indv.fitness)
        ind.insert(3, 'best', int(indv == best))

        # patch outdir
        ind['outdir'] = ind['outdir'].apply(lambda o: os.path.join(f"gen{indv.gen}", o))
        ind.drop(columns=['magnet_coords', 'magnet_angles'], inplace=True)  # debug
        # fitness_componenets should be added last due to variable column number
        for i, comp in enumerate(indv.fitness_components):
            ind.insert(len(ind.columns), f"fitness_component{i}", comp)
        dataset.index = dataset.index.append(ind)



        if not dataset.params:
            dataset.params = ds.params


def main(outdir, individual_class, evaluate_inner, evaluate_outer, minimize_fitness=True, *,
         pop_size=100, generation_num=100, mut_prob=0.2, cx_prob=0.3, mut_strength=1,
         elitism=False, individual_params={}, outer_eval_params={}, evolved_params={}, stop_at_fitness=None, **kwargs):
    print("Initialising")
    for evo_p in evolved_params:
        evolved_params[evo_p] = {"low": evolved_params[evo_p][0],
                                 "high": evolved_params[evo_p][1],
                                 "shape": (evolved_params[evo_p][2:] if len(evolved_params[evo_p]) > 2 else None)}
    individual_class.set_evolved_params(evolved_params)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    pop = [individual_class(**individual_params) for _ in range(pop_size)]
    pop = evaluate_outer(evaluate_inner(pop, 0, outdir, **kwargs), basepath=outdir, **outer_eval_params)
    gen_times = []

    index = pd.DataFrame()
    params = None  # to be added by update_superdataset
    info = {'command': ' '.join(map(shlex.quote, sys.argv)), }
    dataset = Dataset(index, params, info, basepath=outdir)

    update_superdataset(dataset, outdir, pop, 0, minimize_fitness)
    dataset.save()

    for gen in range(1, generation_num + 1):
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
        pop.extend(evaluate_inner(new_kids, gen, outdir, **kwargs))
        pop = evaluate_outer(pop, basepath=outdir, **outer_eval_params)

        # Select
        print("    Select")
        pop = roulette_select(pop, pop_size, elitism, minimize_fitness)
        # pop = fittestSelect(pop, popSize)
        assert len(pop) == pop_size

        update_superdataset(dataset, outdir, pop, gen, minimize_fitness)
        dataset.save()

        best = save_stats(outdir, pop, minimize_fitness)
        if stop_at_fitness is not None and np.isfinite(best.fitness) and (
                (minimize_fitness and best.fitness <= stop_at_fitness) or
                ((not minimize_fitness) and best.fitness >= stop_at_fitness)
        ):
            print(f"Halting early, fitness {best.fitness} achieved")
            print(stop_at_fitness is not None, minimize_fitness and best.fitness <= stop_at_fitness,
                  (not minimize_fitness) and best.fitness >= stop_at_fitness)
            return best
        gen_times.append((datetime.now() - time).total_seconds())
    # best.plot()
    return best
