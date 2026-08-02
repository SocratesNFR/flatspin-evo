import numpy as np
import pandas as pd
from datetime import datetime
import shlex
import os
import sys
from flatspin.data import Dataset
import pickle as pkl
import nevergrad as ng
from copy import deepcopy


def update_superdataset(dataset, outdir, pop, gen, minimize_fitness=True, dataset_params=None):
    if not pop:
        return

    dataset_params = dataset_params or []
    best = min(pop, key=lambda indv: indv.fitness) if minimize_fitness else max(pop, key=lambda indv: indv.fitness)

    ds = Dataset.read(os.path.join(outdir, f"gen{gen}"))
    ind = ds.index.copy()
    ind['outdir'] = ind['outdir'].apply(lambda o: os.path.join(f"gen{gen}", o))
    to_drop = [col for col in ['magnet_coords', 'magnet_angles', 'labels'] if col in ind]
    ind.drop(columns=to_drop, inplace=True)

    id_cols = [c for c in ind.columns if c.startswith("indv_id")]

    for indv in pop:
        # find all rows involving this individual
        mask = (ind[id_cols] == indv.id).any(axis=1)
        rows = ind[mask].copy()
        if rows.empty:
            continue

        rows = rows.assign(
            indv_id=indv.id,
            gen=gen,
            fitness=indv.fitness,
            best=int(indv == best),
        )
        column_names = [f"fitness_component{i}" for i in range(len(indv.fitness_components))]
        rows = rows.assign(**dict(zip(column_names, indv.fitness_components)))

        if dataset_params:
            rows = rows.assign(**dict(zip(dataset_params, [getattr(indv, p) for p in dataset_params])))


        dataset.index = pd.concat([dataset.index, rows], ignore_index=True)

    if not dataset.params:
        dataset.params = ds.params

    return best

def save_snapshot(outdir, optimizer, suffix=""):
    with open(os.path.join(outdir, f"snapshot{suffix}.pkl"), "wb") as f:
        pkl.dump(optimizer, f)


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

def init_evolved_params(param_template, evolved_params):
    for p_name, (p_lower, p_upper) in evolved_params.items():
        param_template[p_name]= ng.p.Scalar((p_lower + p_upper) / 2).set_mutation(
            sigma=(p_upper - p_lower) / 6 # gives +/- 3 sigma range as ng requests
            ).set_bounds(p_lower, p_upper)

def main(
    outdir,
    individual_class,
    evaluate_inner,
    minimize_fitness=True,
    *,
    pop_size=50,
    generation_num=500,
    individual_params={},
    evolved_params={},
    dependent_params={},
    continue_run=False,
    starting_gen=0,
    dataset_params=None,
    random_seed=0,
    **kwargs,
):

    print("Initialising")

    assert not continue_run, "continue_run is not implemented"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    np.random.seed(random_seed)


    # create superdataset
    index = pd.DataFrame()
    info = {
        "command": " ".join(map(shlex.quote, sys.argv)),
    }
    dataset = Dataset(index, None, info, basepath=outdir)

    # update_superdataset(dataset, outdir, pops, 0, minimize_fitness, dataset_params)
    # dataset.save()
    param_template = individual_class.get_default_param_template(**individual_params)
    init_evolved_params(param_template, evolved_params)

    def print_optimizer_chain(opt, depth=0):
        print("  " * depth + f"{opt.name} ({type(opt).__name__})")
        if hasattr(opt, "optim"):
            print_optimizer_chain(opt.optim, depth + 1)
        if hasattr(opt, "optimizers"):  # portfolios hold a list
            for sub in opt.optimizers:
                print_optimizer_chain(sub, depth + 1)


    optimizer = ng.optimizers.NGOpt(
        parametrization=param_template, budget=pop_size * generation_num, num_workers=1)

    print_optimizer_chain(optimizer)
    gen_times = []
    best = None
    for gen in range(starting_gen, generation_num):
        print(f"starting gen {gen} of {generation_num}")
        if len(gen_times) > 0:
            tr = np.mean(gen_times[-10:]) * (generation_num - gen)
            print_time(tr)
        time = datetime.now()


        # batch = [ optimizer.ask() for _ in range(pop_size) ]

        try:
            batch = [optimizer.ask() for _ in range(pop_size)]
        except KeyError as e:
            # walk the chain to find the PSO-like optimizer with .population/._uid_queue
            def find_broken(opt):
                found = []
                if hasattr(opt, "population") and hasattr(opt, "_uid_queue"):
                    found.append(opt)
                if hasattr(opt, "optim"):
                    found += find_broken(opt.optim)
                if hasattr(opt, "optimizers"):
                    for sub in opt.optimizers:
                        found += find_broken(sub)
                return found

            for sub in find_broken(optimizer):
                pop_keys = set(sub.population.keys())
                queue_asked = set(getattr(sub._uid_queue, "asked", []))
                queue_all = set(getattr(sub._uid_queue, "order", []))  # attribute name may vary by version
                print(f"{sub.name}: population has {len(pop_keys)} entries")
                print(f"  missing uid {e.args[0]} in population: {e.args[0] not in pop_keys}")
                print(f"  uid in queue.asked: {e.args[0] in queue_asked}")
                print(f"  population keys sample: {list(pop_keys)[:5]}")
            raise


        batch_uids = [b.uid for b in batch]
        assert len(set(batch_uids)) == len(batch_uids), "duplicate uids in batch!"

        pop = [individual_class(params=deepcopy(params.value), **individual_params) for params in batch]

        if gen == generation_num - 1: # last generation, ask for the best recommendation too
            best = optimizer.provide_recommendation()
            pop.append(individual_class(params=deepcopy(best.value), **individual_params))


        print("    Evaluate")
        evaluate_inner(
            pop,
            gen,
            outdir,
            dependent_params=dependent_params,
            **kwargs,
        )
        for indv in pop:
            indv.fitness = sum(indv.fitness_components)
        for b, indv in zip(batch, pop[:len(batch)]): # skip the best recommendation if it was added
            assert b.uid in batch_uids, "batch uid not found in batch_uids"
            optimizer.tell(b, indv.fitness if minimize_fitness else -indv.fitness)

        best = update_superdataset(
            dataset, outdir, pop, gen, minimize_fitness, dataset_params
        )
        dataset.save()


        if best:
            print(f"Best fitness: {best.fitness}")

        save_snapshot(outdir, optimizer)

        gen_times.append((datetime.now() - time).total_seconds())
    return best
