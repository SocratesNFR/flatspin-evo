import numpy as np
from scipy.spatial.distance import hamming
import pandas as pd
from itertools import chain
import heapq
from datetime import datetime
from collections import OrderedDict, defaultdict
import shlex
import os
import sys
from warnings import warn
from flatspin.data import Dataset, read_csv
import pickle as pkl


def better_than(indv1, indv2, minimize_fitness=True, strict=True):

    if np.isnan(indv1.fitness):
        return False
    if np.isnan(indv2.fitness):
        return True

    if strict:
        if minimize_fitness:
            return indv1.fitness < indv2.fitness
        else:
            return indv1.fitness > indv2.fitness

    if minimize_fitness:
        return indv1.fitness <= indv2.fitness
    else:
        return indv1.fitness >= indv2.fitness

def get_best(indvs, n=None, minimize_fitness=True, return_nan=False):
    # if n is None, return the single best, otherwise return a list of the n best
    k = n if n is not None else 1

    valid = [
        indv for indv in indvs
        if not np.isnan(indv.fitness)
    ]

    fn = heapq.nsmallest if minimize_fitness else heapq.nlargest

    best = fn(
        k,
        valid,
        key=lambda indv: indv.fitness,
    )

    if return_nan and not best and indvs:
        best = indvs[:k]

    if n is None:
        return best[0] if best else None

    return best


def update_superdataset(dataset, outdir, pops, gen, minimize_fitness=True, dataset_params=None):
    pop = list(flat_n_concat(pops))
    if not pop:
        return
    dataset_params = dataset_params or []
    best = get_best(pop, minimize_fitness=minimize_fitness)
    pop_bests = {i: get_best([indv for indv in pop if indv.pop_id == i], minimize_fitness=minimize_fitness) 
             for i in range(len(pops))}

    ds = Dataset.read(os.path.join(outdir, f"gen{gen}"))
    ind = ds.index.copy()
    ind['outdir'] = ind['outdir'].apply(lambda o: os.path.join(f"gen{gen}", o))
    to_drop = [col for col in ['magnet_coords', 'magnet_angles', 'labels'] if col in ind]
    ind.drop(columns=to_drop, inplace=True)

    id_cols = [c for c in ind.columns if c.startswith("indv_id_")]

    for indv in pop:
        # find all rows involving this individual
        mask = (ind[id_cols] == indv.id).any(axis=1)
        rows = ind[mask].copy()
        if rows.empty:
            continue

        rows = rows.assign(
            indv_id=indv.id,
            gen=gen,
            born=indv.gen,
            pop_id=indv.pop_id,
            fitness=indv.fitness,
            best=int(indv == pop_bests[indv.pop_id]),
        )
        column_names = [f"fitness_component{i}" for i in range(len(indv.fitness_components))]
        rows = rows.assign(**dict(zip(column_names, indv.fitness_components)))

        dataset.index = pd.concat([dataset.index, rows], ignore_index=True)

    if not dataset.params:
        dataset.params = ds.params

    return best

def save_snapshot(outdir, pop, suffix=""):
    with open(os.path.join(outdir, f"snapshot{suffix}.pkl"), "wb") as f:
        pkl.dump([repr(indv) for indv in pop], f)


def setup_continue_run(outdir, individual_class, start_gen, n_populations=1):
    pops = []
    for i in range(n_populations):
        with open(os.path.join(outdir, f"snapshot_{i}.pkl"), "rb") as f:
            pops.append(list(map(lambda ind: individual_class.from_string(ind), pkl.load(f))))

    dataset = Dataset.read(outdir)

    assert os.path.isdir(
        os.path.join(outdir, f"gen{start_gen-1}")
    ), f"gen{start_gen-1} does not exist"

    if os.path.isdir(os.path.join(outdir, f"gen{start_gen}")):
        # rename last gen so not overwritten
        gen_string = f"gen{start_gen}"
        os.rename(
            os.path.join(outdir, gen_string), os.path.join(outdir, "old_" + gen_string)
        )
        gen_string = "old_" + gen_string
    else:
        gen_string = f"gen{start_gen-1}"

    # find the largest id in the dataset so we dont overwrite
    newest_index = read_csv(os.path.join(outdir, gen_string, "index.csv"))
    super_index = read_csv(os.path.join(outdir, "index.csv"))
    max_id = np.concatenate(
        (newest_index["indv_id"].values, super_index["indv_id"].values)
    ).max()
    individual_class.set_id_start(max_id + 1)

    return pops, dataset


def main_check_args(individual_params, pop_specific_params, n_pops, kwargs):
    check_args = np.unique(
        list(individual_params) + list(pop_specific_params) + list(kwargs), return_counts=True
    )
    check_args = [
        check_args[0][i] for i in range(len(check_args[0])) if check_args[1][i] > 1
    ]
    if check_args:
        raise RuntimeError(f"param '{check_args[0]}' appears in multiple param groups")

    for k, v in pop_specific_params.items():
        assert len(v) == n_pops, f"pop_specific_params['{k}'] has {len(v)} values but n_pops={n_pops}"




def crossover(pop, n_kids, minimize_fitness=True, use_rank=False):
    fitnesses = np.array([indv.fitness if not np.isnan(indv.fitness) else np.inf if minimize_fitness else -np.inf for indv in pop], dtype=float)

    if use_rank:
        order = np.argsort(fitnesses)
        if not minimize_fitness:
            order = order[::-1]
        weights = np.zeros(len(pop))
        weights[order] = np.arange(1, len(pop) + 1, dtype=float)
    else:
        weights = fitnesses.copy()
        if minimize_fitness:
            weights = weights.max() - weights

    if weights.sum() == 0:
        weights = np.ones(len(pop))
    weights /= weights.sum()

    kids = []
    for _ in range(n_kids):
        a_idx = np.random.choice(len(pop), p=weights)
        remaining_idx = [i for i in range(len(pop)) if i != a_idx]
        remaining_weights = weights[remaining_idx] / weights[remaining_idx].sum()
        b_idx = np.random.choice(remaining_idx, p=remaining_weights)
        kids.append(pop[a_idx].crossover(pop[b_idx])[0])
    return kids

def niche_select(pop, n, minimize_fitness=True, threshold=0.05):
    # group into niches by similarity
    niches = []
    for indv in sorted(pop, key=lambda i: i.fitness, reverse=not minimize_fitness):
        for niche in niches:
            if hamming(indv.novelty > 0, niche[0].novelty >  0) < threshold:
                niche.append(indv)
                break
        else:
            niches.append([indv])  # new niche

    # niches are already internally sorted by fitness since we iterate best-first
    # sort niches by their best member
    niches.sort(key=lambda n: n[0].fitness, reverse=not minimize_fitness)

    # round-robin pick best from each niche
    selected = []
    while len(selected) < n:
        for niche in niches:
            if niche:
                selected.append(niche.pop(0))
            if len(selected) == n:
                break

    return selected


def print_time(tr):
    print(
        f"~{np.round(tr / 3600, 2)} hours remaining"
        if tr > 3600
        else (
            f"~{np.round(tr / 60, 2)} minutes remaining"
            if tr > 60
            else f"~{np.round(tr, 2)} seconds remaining"
        )
    )


def choose(lst, size):
    if size == 0 or len(lst) == 0:
        return []

    indices = np.arange(len(lst))
    chosen = []

    if len(indices) < size:
        repeats = size // len(indices)
        chosen += list(indices) * repeats  # Add repeated full list cycles

    # Add remaining random choices
    chosen += np.random.choice(
        a=indices, size=size - len(chosen), replace=False
    ).tolist()

    return [lst[i] for i in chosen]

def flat_n_concat(*lists):
    """takes one or more 'list of lists' and flattens and concatenates them into a single shallow list"""
    return chain.from_iterable(chain(*lists))

def migrate_genome(indv, target_pop_id, gen, pop_specific_params={}):
    migrant = indv.copy()
    migrant.pop_id = target_pop_id
    migrant.gen = gen
    
    for k, v in pop_specific_params.items():
        setattr(migrant, k, v[target_pop_id])
    
    genome_params_len = len(migrant.genome_params)
    state = migrant.genome[genome_params_len:]
    state_2d = state.reshape(10, 5)
    state_2d = np.rot90(state_2d, 2)
    state_2d = np.roll(state_2d, 1, axis=0)
    migrant.genome[genome_params_len:] = state_2d.flatten()
    return migrant

def main(
    outdir,
    individual_class,
    evaluate_inner,
    evaluate_outer,
    minimize_fitness=True,
    *,
    pop_size=50,
    n_pops=2,
    generation_num=500,
    mut_prob=0.05,
    cx_ratio=2,
    mut_strength=1,
    outer_eval_params={},
    individual_params={},
    pop_specific_params={},  # dict of {param: [val_for_pop0, val_for_pop1, ...]} - used for individual params
    dependent_params={},
    elitism=0.1,
    continue_run=False,
    starting_gen=1,
    dataset_params=None,
    random_seed=0,
    niche_threshold=0.05,
    migration_prob=0.0,
    **kwargs,
):

    print("Initialising")
    main_check_args(individual_params, pop_specific_params, n_pops, kwargs)

    assert (
        os.path.isdir(outdir) or not continue_run
    ), "can't continue run without existing outdir"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    np.random.seed(random_seed)


    if continue_run:
        pops, dataset = setup_continue_run(outdir, individual_class, starting_gen, n_pops)
        for i, subpop in enumerate(pops):
            for indv in subpop:
                indv.pop_id = i

    else:
        pops = [
            [individual_class(**individual_params, **{k: v[i] for k, v in pop_specific_params.items()}) for _ in range(pop_size)]
            for i in range(n_pops)
        ]
        for i, subpop in enumerate(pops):
            for indv in subpop:
                indv.pop_id = i

        evaluate_inner(
            pops,
            0,
            outdir,
            dependent_params=dependent_params,
            **kwargs,
        )
        evaluate_outer(flat_n_concat(pops), basepath=outdir, gen=0, **outer_eval_params)
        # create superdataset
        index = pd.DataFrame()
        info = {
            "command": " ".join(map(shlex.quote, sys.argv)),
        }
        dataset = Dataset(index, None, info, basepath=outdir)

        update_superdataset(dataset, outdir, pops, 0, minimize_fitness, dataset_params)
        dataset.save()

    gen_times = []
    best = None
    for gen in range(starting_gen, generation_num + 1):
        print(f"starting gen {gen} of {generation_num}")
        individual_class.current_gen = gen
        if len(gen_times) > 0:
            tr = np.mean(gen_times[-10:]) * (generation_num - gen)
            print_time(tr)
        time = datetime.now()

        # Crossover!
        print("    Crossover")
        kids = [crossover(pop, pop_size * cx_ratio, minimize_fitness=minimize_fitness, use_rank=True) for pop in pops]

        # Mutate!
        print("    Mutate")
        for i, subkids in enumerate(kids):
            for kid in subkids:
                if np.random.rand() < mut_prob:
                    kid.mutate(mut_strength)
                kid.gen = gen
                kid.pop_id = i
                kid.refresh()
        # Eval
        print("    Evaluate")


        all_elites = [
            get_best(pops[i], int(pop_size * elitism), minimize_fitness, return_nan=False)
            for i in range(len(pops))
        ]

        for i, elites in enumerate(all_elites):
            for e in elites:
                e.refresh()
            pops[i] = kids[i] + elites

        for i in range(n_pops):
            if n_pops == 1 or np.random.rand() > migration_prob:
                continue
            donor_pop = np.random.choice([j for j in range(n_pops) if j != i])
            expat = np.random.choice(all_elites[donor_pop])
            pops[i].append(migrate_genome(expat, i, gen, pop_specific_params))


        evaluate_inner(
            pops,
            gen,
            outdir,
            dependent_params=dependent_params,
            **kwargs,
        )

        evaluate_outer(flat_n_concat(pops), basepath=outdir, gen=gen, **outer_eval_params)

        # Select
        pops[:] = [niche_select(pop, pop_size, minimize_fitness, threshold=niche_threshold) for pop in pops]


        best = update_superdataset(
            dataset, outdir, pops, gen, minimize_fitness, dataset_params
        )
        dataset.save()


        for i, subpop in enumerate(pops):
            pop_best = get_best(subpop, minimize_fitness=minimize_fitness)
            if pop_best is not None:
                print(f"  pop {i} best fitness: {pop_best.fitness}")

        born_this_gen = sum(indv.gen == gen for indv in flat_n_concat(pops))
        n_elites = int(pop_size * elitism) * n_pops
        n_dead_elites = n_elites - (pop_size * n_pops - born_this_gen)
        print(f"{n_dead_elites} elites were defeated this generation")

        for i, subpop in enumerate(pops):
            save_snapshot(outdir, subpop, suffix=f"_{i}")

        gen_times.append((datetime.now() - time).total_seconds())
    return best
