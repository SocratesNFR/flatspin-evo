import os
import shlex
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from ribs.archives import DNSArchive
from flatspin.data import Dataset
from MixedEmitter import MixedMutationEmitter

SNAPSHOT = "snapshot.npz"


def save_snapshot(outdir, archive, evaluator):
    """Writes the current population atomically (a killed run never leaves a half-written file)."""
    path = os.path.join(outdir, SNAPSHOT)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:  # a file object stops np.savez appending its own extension
        np.savez(f,
                 solution=archive.data("solution"), objective=archive.data("objective"),
                 measures=archive.data("measures"), indv_id=archive.data("indv_id"),
                 gen=evaluator._gen, next_id=evaluator._next_id)
    os.replace(tmp, path)


def load_snapshot(outdir, archive, evaluator):
    """Refills `archive` from the snapshot and restores the evaluator's counters."""
    path = os.path.join(outdir, SNAPSHOT)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nothing to resume from: {path} does not exist")
    snap = np.load(path)

    evaluator._gen = int(snap["gen"])          # next unused generation number
    evaluator._next_id = int(snap["next_id"])  # next unused indv_id
    if len(snap["solution"]):
        archive.add(solution=snap["solution"], objective=snap["objective"],
                    measures=snap["measures"], indv_id=snap["indv_id"])

    # A run killed mid-generation leaves a gen folder that was never added; move it aside.
    leftover = os.path.join(outdir, f"gen{evaluator._gen}")
    if os.path.exists(leftover):
        os.rename(leftover, f"{leftover}.abandoned{int(time.time())}")
    return len(snap["solution"])


def update_superdataset(dataset, outdir, gen, ids, objectives, measures, infos):
    """Appends one row per individual to the superdataset, combining the
    run parameters from gen{gen} with the objective and measures."""
    batch = Dataset.read(os.path.join(outdir, f"gen{gen}"))

    rows = []
    for indv_id, obj, meas, info in zip(ids, objectives, measures, infos):
        ind = batch.filter(indv_id=indv_id).index.copy()
        ind["gen"] = gen
        ind["objective"] = obj
        for i, m in enumerate(meas):
            ind[f"measure{i}"] = m
        for key, value in info.items():
            ind[key] = value
        ind["outdir"] = ind["outdir"].apply(lambda o: os.path.join(f"gen{gen}", o))
        rows.append(ind)

    if not dataset.params:
        dataset.params = batch.params

    frames = ([dataset.index] if len(dataset.index) else []) + rows
    dataset.index = pd.concat(frames, ignore_index=True)


def main(evaluator, archive=None, total_itrs=100, batch_size=32, seed=None,
         archive_capacity=200, k_neighbors=5, resume=False):
    """Runs the QD loop for `total_itrs` more generations (on a resume, that many
    additional ones). Failures (non-finite objective or measures) are not added."""
    layout = evaluator.layout
    outdir = evaluator.outdir

    if archive is None:
        archive = DNSArchive(
            solution_dim=layout.solution_dim,
            measure_dim=evaluator.measure_dim,
            capacity=archive_capacity,
            k_neighbors=k_neighbors,
            extra_fields={"indv_id": ((), np.int64)},
        )

    restored = load_snapshot(outdir, archive, evaluator) if resume else 0

    info = {"command": " ".join(map(shlex.quote, sys.argv))}
    if resume and restored:
        dataset = Dataset.read(outdir)
        # Drop rows from generations the snapshot doesn't cover, e.g. when the run was
        # killed after the superdataset was saved but before the snapshot.
        if "gen" in dataset.index.columns:
            dataset.index = dataset.index[dataset.index["gen"] < evaluator._gen].reset_index(drop=True)
    else:
        dataset = Dataset(pd.DataFrame(), None, info, basepath=outdir)

    emitter = MixedMutationEmitter(archive, layout=layout, batch_size=batch_size, seed=seed)
    if resume:
        tqdm.write(f"Resumed at generation {evaluator._gen} with {restored} individuals in the archive")

    for itr in trange(1, total_itrs + 1, file=sys.stdout, desc="Iterations"):
        solutions = emitter.ask()
        objectives, measures, ids, infos = evaluator.evaluate(solutions)

        valid = np.isfinite(objectives) & np.isfinite(measures).all(axis=1)
        if valid.any():
            archive.add(solution=solutions[valid], objective=objectives[valid],
                        measures=measures[valid], indv_id=ids[valid])
            update_superdataset(dataset, outdir, evaluator.last_gen, ids[valid],
                                objectives[valid], measures[valid],
                                [infos[i] for i in np.flatnonzero(valid)])
            dataset.save()

        save_snapshot(outdir, archive, evaluator)  # every generation, even if nothing was valid
        tqdm.write(f"itr {itr}: {valid.sum()}/{len(valid)} valid, archive size {len(archive)}")

    if len(archive) == 0:
        print("No valid solutions were found.")
        return archive

    elite_solutions = archive.data("solution")
    elite_objectives = archive.data("objective")
    best = np.argmax(elite_objectives)
    print("Best objective:", elite_objectives[best])
    print("Best indv_id:", archive.data("indv_id")[best])
    print("Best solution:", elite_solutions[best])
    return archive