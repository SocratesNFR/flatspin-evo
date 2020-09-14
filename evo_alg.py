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
from flatspin.tools.run import import_model
from flatspin.encoder import encoders as dipole_encoders
from flatspin.data import Dataset, is_archive_format
from flatspin.utils import get_default_params


def rainbow_colours(num):
    cmap = plt.get_cmap('gist_rainbow')
    return cmap(np.linspace(0, 1, num))


def roulette_select(pop, pop_size, elitism=False):
    fits = np.fromiter(map(lambda p: p.fitness, pop), count=len(pop), dtype=np.float64)
    #print(fits)
    if elitism:
        best_index = np.argmax(fits)
        best = pop.pop(best_index)  # remove best to be added in at end
        fits = np.delete(fits, best_index)
    assert not np.isnan(fits).any()
    fits = zscore(fits)
    if np.isnan(fits).any():
        fits = None
    else:
        fits += np.abs(np.min(fits)) + 1
        fits = fits / np.sum(fits)
        assert not np.isnan(fits).any()
    #print(fits)
    if elitism:
        new_pop = list(np.random.choice(pop, pop_size - 1, False, p=fits)) + [best]
    else:
        new_pop = list(np.random.choice(pop, pop_size, False, p=fits))

    return new_pop


def fittest_select(pop, pop_size):
    fits = np.fromiter(map(lambda p: p.fitness, pop), count=len(pop), dtype=np.float64)
    assert not np.isnan(fits).any()
    best_indicies = np.argpartition(fits, -pop_size)[-pop_size:]

    return [pop[i] for i in best_indicies]


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


def save_stats(outdir, pop):
    fits = list(map(lambda indv: indv.fitness, pop))
    out = [f"mean: {(np.mean(fits))}; max: {(np.max(fits))}", "\nBest indv in current pop:\n"]

    try:  # save fitness components if they exist
        fit_comps = list(map(lambda indv: indv.fitness_components, pop))
        out[0] += f"; cMean: {(list(np.mean(fit_comps, 0)))}; cMax: {(list(np.max(fit_comps, 0)))}"
    except AttributeError:
        pass

    best = max(pop, key=lambda indv: indv.fitness)
    out.extend(repr(best))
    out.append("\n")

    with(open(f"{outdir}\\stats.txt", "a+")) as f:
        f.writelines(out)
    return best


def top_of_the_pops(result, individual_class, interval=400):
    frames = []
    for indv in [gen["bestIndv"] for gen in result]:
        frames.append(list(map(lambda m: m.as_patch(), individual_class.from_string(indv).pheno)))
    return individual_class.frames2animation(frames, interval, True)


def evo_run(runs_params, shared_params, gen):
    """ modified from run_sweep.py main()"""
    model_name = shared_params.pop("model", "generated")
    model_class = import_model(model_name)
    encoder_name = shared_params.get("encoder", "sin")
    encoder = dipole_encoders[encoder_name]

    data_format = shared_params.get("format", "npz")

    params = get_default_params(run)
    params['encoder'] = encoder_name
    params.update(get_default_params(model_class))
    params.update(get_default_params(encoder))
    params.update(shared_params)

    info = {
        'model': f'{model_class.__module__}.{model_class.__name__}',
        'model_name': model_name,
        'data_format': data_format,
        'command': ' '.join(map(shlex.quote, sys.argv)),
    }

    ext = data_format if is_archive_format(data_format) else "out"

    outdir_tpl = "gen{:d}indv{:d}." + ext

    basepath = params["basepath"]
    if os.path.exists(basepath):
        # Refuse to overwrite an existing dataset
        raise FileExistsError(basepath)
    os.makedirs(basepath)

    index = []
    filenames = []
    # Generate queue
    for run_params in runs_params:
        newparams = copy.copy(params)
        newparams.update(run_params)

        outdir = outdir_tpl.format(gen, newparams["indv_id"])
        filenames.append(outdir)
        row = OrderedDict(run_params)
        row.update({'outdir': outdir})
        index.append(row)

    # Save dataset
    index = pd.DataFrame(index)
    dataset = Dataset(index, params, info, basepath)
    dataset.save()

    # Run!
    print("Starting sweep with {} runs".format(len(dataset)))
    run_type = shared_params.get("run", "local")
    if run_type == 'local':
        run_local(dataset)

    elif run_type == 'dist':
        run_dist(dataset)

    return


def main(outdir, individual_class, evaluate_inner, evaluate_outer, *,
         pop_size=100, generation_num=100, mut_prob=0.2, cx_prob=0.3,
         elitism=False, run="local", **kwargs):
    print("Initialising")
    pop = [individual_class() for _ in range(pop_size)]
    pop = evaluate_outer(evaluate_inner(pop, 0, outdir, run))
    gen_times = []
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
                new_kids.append(indv.mutate())
        # Crossover!
        print("    Crossover")
        for i, indv in enumerate(pop):  # TODO: replace with itertools combination or likewise
            if np.random.rand() < cx_prob:
                partner = np.random.choice(pop)  # can partner with itself, resulting in perfect copy
                new_kid = indv.crossover(partner)
                if new_kid is not None:  # do not append if crossover failed
                    new_kids.append(new_kid)

                    # Eval
        print("    Evaluate")
        pop.extend(evaluate_inner(new_kids, gen, outdir, run))
        pop = evaluate_outer(pop)

        # Select
        print("    Select")
        pop = roulette_select(pop, pop_size, elitism)
        # pop = fittestSelect(pop, popSize)
        assert len(pop) == pop_size

        best = save_stats(outdir, pop)
        gen_times.append((datetime.now() - time).total_seconds())
    #best.plot()
    return best
