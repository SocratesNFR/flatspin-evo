import inspect
import os
import shlex
import sys
import time
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd

from evaluator import Evaluator
from MixedEmitter import GenomeLayout
import evaluation_functions 

from flatspin.data import Dataset, is_archive_format
from flatspin.utils import get_default_params, import_class
from flatspin.runner import run, run_dist, run_local
from flatspin.cmdline import StoreKeyValue, eval_params


def is_not_ready(e: Exception) -> bool:
    """True if the exception just means the run's output isn't ready yet."""
    return (
        isinstance(e, FileNotFoundError)
        or (isinstance(e, AssertionError)
            and "No vector data found for quantity: mag" in str(e))
        or "Bad magic number" in str(e)
    )


class FlatspinEvaluator(Evaluator):
    """Runs a batch of flatspin simulations per call, then analyses each run.

    Args:
        shared_params: Constant simulator parameters (like -p "H=0.01"). May
            also hold "model", "encoder", "format" and "run" ("local"/"dist"),
            as in your evo_run.
        gene_params: Parameters taken from real genes (like -g "a=[0.04,0.06]"),
            as name -> (low, high). The gene is kept in [0, 1] internally and
            rescaled to this range.
        dependent_params: Parameters that depend on other parameters (like -d "H0=-H*2").
        n_binary: Number of binary genes.
        outdir: Directory for the datasets. Each batch goes in outdir/gen{N}.
            Use a fresh outdir per experiment, since existing datasets are
            never overwritten.
        max_jobs: Passed to run_dist.
        poll_interval: Seconds to sleep when a full pass over the pending runs
            finished none of them (dist runs only).

    Subclasses implement `analyse`, and `bits_to_params` if n_binary > 0.
    """

    def __init__(self, gene_params: dict, n_binary: int, outdir: str,
                 evaluation_func: str, evaluation_params: dict | None = None, dependent_params: dict | None = None,
                 max_jobs: int = 1000, poll_interval: float = 30.0, **shared_params):
        
        if evaluation_func not in evaluation_functions.EVALUATIONS:
            raise ValueError(
                f"Unknown evaluation function '{evaluation_func}'. "
                f"Available: {sorted(evaluation_functions.EVALUATIONS)}")
        
        self.evaluation_func = evaluation_functions.EVALUATIONS[evaluation_func]
        self.measure_dim = self.evaluation_func.measure_dim
        self.dependent_params = dict(dependent_params or {})

        # check evaluation_params are valid
        self.evaluation_params = dict(evaluation_params or {})
        accepted = inspect.signature(self.evaluation_func).parameters
        unknown = set(self.evaluation_params) - set(accepted)
        if unknown:
            raise ValueError(
                f"Unknown args for {evaluation_func}: {sorted(unknown)}. "
                f"It accepts: {[p for p in accepted if p != 'ds']}")


        overlap = (set(shared_params) | set(self.dependent_params)) & set(gene_params)
        if overlap:
            raise ValueError(f"Parameters defined more than once: {sorted(overlap)}")

        self.shared_params = shared_params
    
        self.gene_names = list(gene_params)
        bounds = np.array([gene_params[name] for name in self.gene_names], dtype=float)
        if bounds.shape != (len(self.gene_names), 2):
            raise ValueError("Each gene_params value must be a (low, high) pair")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("gene_params ranges need low < high")
        self._low, self._high = bounds[:, 0], bounds[:, 1]

        self.run_type = self.shared_params.get("run", "local")
        self.outdir = outdir
        self.max_jobs = max_jobs
        self.poll_interval = poll_interval
        self._gen = 0  # batch counter; each evaluate() call uses the next one
        self._next_id = 0  # next unused indv_id; keeps ids unique across batches

        super().__init__(GenomeLayout(n_real=len(self.gene_names), n_binary=n_binary))

    # --- genome -> run parameters ---------------------------------------

    def decode(self, genome: np.ndarray):
        """Returns (gene parameter dict, binary genes) for one genome."""
        real = genome[self.layout.real]  # assumes the emitter's default real_range (0, 1)
        scaled = self._low + real * (self._high - self._low)
        bits = genome[self.layout.binary].astype(np.int8)
        return dict(zip(self.gene_names, scaled.tolist())), bits

    def bits_to_params(self, bits: np.ndarray) -> dict:
        """Maps one genome's binary genes to simulator parameters.

        Override this if n_binary > 0. Return scalars or strings rather than
        arrays, since the values are saved in the dataset index."""
        if len(bits):
            raise NotImplementedError("Override bits_to_params to map binary genes to parameters")
        return {}

    # --- simulation (adapted from evo_run) --------------------------------

    def simulate(self, run_params: list, gen: int) -> str:
        """Builds a dataset with one run per entry in run_params and starts it.

        Local runs block until finished. Dist runs return after submitting.
        Returns the dataset's basepath."""
        shared = dict(self.shared_params)
        model_name = shared.pop("model", "CustomSpinIce")
        model_class = import_class(model_name, "flatspin.model")
        encoder_name = shared.get("encoder", "Sine")
        encoder_class = (import_class(encoder_name, "flatspin.encoder")
                         if isinstance(encoder_name, str) else encoder_name)
        data_format = shared.get("format", "npz")

        basepath = os.path.join(self.outdir, f"gen{gen}")
        if os.path.exists(basepath):
            raise FileExistsError(basepath)
        os.makedirs(basepath)

        params = get_default_params(run)
        params["encoder"] = f"{encoder_class.__module__}.{encoder_class.__name__}"
        params.update(get_default_params(model_class))
        params.update(get_default_params(encoder_class))
        params.update(shared)
        params["basepath"] = basepath

        info = {
            "model": f"{model_class.__module__}.{model_class.__name__}",
            "model_name": model_name,
            "data_format": data_format,
            "command": " ".join(map(shlex.quote, sys.argv)),
        }

        ext = data_format if is_archive_format(data_format) else "out"
        index = []
        for rp in run_params:
            row = OrderedDict(rp)
            dp = eval_params(self.dependent_params, rp)
            row.update(dp)
            row["outdir"] = f"gen{gen}indv{rp['indv_id']}.{ext}"

            index.append(row)

        dataset = Dataset(pd.DataFrame(index), params, info, basepath)
        dataset.save()

        if self.run_type == "local":
            run_local(dataset)  # blocks until finished
        elif self.run_type == "dist":
            run_dist(dataset, wait=False, max_jobs=self.max_jobs)  # returns after submitting
        else:
            raise ValueError(f"Unknown run type: {self.run_type}")
        return basepath

    # --- analysis ----------------------------------------------------------

    def evaluate_run(self, ds):
        """Runs the selected evaluation function on one run. Must be safe to call
        repeatedly, since it is retried until the output is ready."""
        return self.evaluation_func(ds, **self.evaluation_params)
    
    def collect(self, basepath: str) -> dict:
        """Analyses runs as their output becomes available.

        Returns a dict of indv_id -> (objective, measures). Runs whose output
        isn't ready are retried; any other exception is raised."""
        pending = list(Dataset.read(basepath))
        results = {}

        while pending:
            still_pending = []
            for ds in pending:
                indv_id = int(ds.index["indv_id"].unique()[0])
                try:
                    results[indv_id] = self.evaluate_run(ds)
                except Exception as e:
                    if self.run_type == "local" or not is_not_ready(e):
                        raise
                    still_pending.append(ds)  # output not ready yet, retry next pass

            progressed = len(still_pending) < len(pending)
            pending = still_pending
            if pending and not progressed:
                time.sleep(self.poll_interval)

        return results

    # --- batch evaluation --------------------------------------------------

    def evaluate(self, solutions):
        gen = self._gen
        self._gen += 1
        
        self.last_gen = gen

        ids = list(range(self._next_id, self._next_id + len(solutions)))
        self._next_id += len(solutions)

        run_params = []
        for indv_id, genome in zip(ids, solutions):
            gene_vals, bits = self.decode(genome)
            run_params.append({"indv_id": indv_id, **gene_vals, **self.bits_to_params(bits)})

        basepath = self.simulate(run_params, gen)
        results = self.collect(basepath)

        missing = set(ids) - set(results)
        if missing:
            raise RuntimeError(f"No results for individuals {sorted(missing)} in {basepath}")

        objectives = np.array([results[i][0] for i in ids], dtype=float)
        measures = np.array([results[i][1] for i in ids], dtype=float)
        return objectives, measures, np.array(ids)

    @classmethod
    def main(cls):
        import argparse
        import ribs_main

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("-o", "--output", metavar="DIR", required=True,
                            help="output directory for the datasets")
        parser.add_argument("-p", "--parameter", action=StoreKeyValue, default={},
                            help="param passed to flatspin and the evaluator (e.g. -p H=0.01)")
        parser.add_argument("-d", "--dependent_param", action=StoreKeyValue, default={},
                            help="param dependent on other params (e.g. -g H=[0.5,1] -d 'H0=-H*2')")
        parser.add_argument("-g", "--genome_param", action=StoreKeyValue, default={},
                            help="flatspin param controlled by a gene, e.g. -g a=[0.04,0.06]")
        parser.add_argument("-f", "--evaluation_func", required=True,
                    choices=sorted(evaluation_functions.EVALUATIONS),
                    help="name of the evaluation function in evaluation_functions.py")
        parser.add_argument("-E", "--evaluation_param", action=StoreKeyValue, default={},
                    help="keyword arg for the evaluation function, e.g. -E state_step=5")
        parser.add_argument("-n", "--itrs", type=int, default=100, help="number of QD iterations")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--seed", type=int, default=None)
        args = parser.parse_args()
        evaluator = cls(
            outdir=args.output,
            evaluation_func=args.evaluation_func,
            evaluation_params=eval_params(args.evaluation_param),
            dependent_params=args.dependent_param,
            gene_params=eval_params(args.genome_param),
            **eval_params(args.parameter),
        )
        ribs_main.main(evaluator, total_itrs=args.itrs, batch_size=args.batch_size, seed=args.seed)

if __name__ == "__main__": # put this in your subclass, dont run this one
    FlatspinEvaluator.main()