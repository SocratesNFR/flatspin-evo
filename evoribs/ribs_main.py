import os
import shlex
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from ribs.archives import DNSArchive
from flatspin.data import Dataset
from MixedEmitter import MixedMutationEmitter


def update_superdataset(dataset, outdir, gen, ids, objectives, measures):
    """Appends one row per individual to the superdataset, combining the
    run parameters from gen{gen} with the objective and measures."""
    batch = Dataset.read(os.path.join(outdir, f"gen{gen}"))

    rows = []
    for indv_id, obj, meas in zip(ids, objectives, measures):
        ind = batch.filter(indv_id=indv_id).index.copy()
        ind["gen"] = gen
        ind["objective"] = obj
        for i, m in enumerate(meas):
            ind[f"measure{i}"] = m
        ind["outdir"] = ind["outdir"].apply(lambda o: os.path.join(f"gen{gen}", o))
        rows.append(ind)

    if not dataset.params:
        dataset.params = batch.params

    frames = ([dataset.index] if len(dataset.index) else []) + rows
    dataset.index = pd.concat(frames, ignore_index=True)


def main(evaluator, archive=None, total_itrs=100, batch_size=32, seed=None):
    """Runs the QD loop.

    `evaluator.evaluate` must return (objectives, measures, ids). Individuals
    with a non-finite objective or measure (failures) are not added to the
    archive or the superdataset.
    """
    layout = evaluator.layout
    outdir = evaluator.outdir

    if archive is None:
        archive = DNSArchive(
            solution_dim=layout.solution_dim,
            measure_dim=evaluator.measure_dim,
            capacity=200,
            k_neighbors=7,
        )

    emitter = MixedMutationEmitter(archive, layout=layout, batch_size=batch_size, seed=seed)

    info = {"command": " ".join(map(shlex.quote, sys.argv))}
    dataset = Dataset(pd.DataFrame(), None, info, basepath=outdir)

    for itr in trange(0, total_itrs, file=sys.stdout, desc="Generations"):
        solutions = emitter.ask()
        objectives, measures, ids = evaluator.evaluate(solutions)

        valid = np.isfinite(objectives) & np.isfinite(measures).all(axis=1)
        if valid.any():
            archive.add(solution=solutions[valid], objective=objectives[valid],
                        measures=measures[valid])
            update_superdataset(dataset, outdir, evaluator.last_gen,
                                ids[valid], objectives[valid], measures[valid])
            dataset.save()

        tqdm.write(f"itr {itr}: {valid.sum()}/{len(valid)} valid, archive size {len(archive)}")

    if len(archive) == 0:
        print("No valid solutions were found.")
        return archive

    elite_solutions = archive.data("solution")
    elite_objectives = archive.data("objective")
    best = np.argmax(elite_objectives)
    print("Best objective:", elite_objectives[best])
    print("Best solution:", elite_solutions[best])

    return archive