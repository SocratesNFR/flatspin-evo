from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
import os
import warnings
from copy import copy, deepcopy
import shlex
import sys
from collections import OrderedDict, deque
from time import sleep
import pandas as pd
from itertools import chain, count
import traceback

from flatspin.data import Dataset, read_table, load_output, is_archive_format, match_column, save_table
from flatspin.utils import get_default_params, import_class
from flatspin.runner import run, run_dist, run_local
from flatspin.sweep import sweep
from flatspin.cmdline import eval_params


numeric = Union[int, float, np.number]


class Base_Individual(ABC):
    id: int
    gen: int
    parent_ids: List[int]
    fitness: numeric
    fitness_components: List[numeric]
    fitness_info: List

    _evolved_params = {}
    _id_counter = count(0)

    def __init__(self, *, id=None, gen=0, fitness=None, fitness_components=None, fitness_info=None,
                parent_ids=None, evolved_params_values=None, remember_fitness=0, fitness_history=None,
                last_random_seed=0, **kwargs):

        self.id = id if id is not None else next(Base_Individual._id_counter)
        self.gen = gen  # generation of birth

        self.fitness = fitness

        self.fitness_components = fitness_components
        self.fitness_info = fitness_info

        if parent_ids is None:
            self.parent_ids = []
        else:
            self.parent_ids = parent_ids

        self.remember_fitness = remember_fitness
        if fitness_history is None and remember_fitness > 0:
            self.fitness_history = deque(maxlen=remember_fitness)
        else:
            self.fitness_history = fitness_history

        self.init_evolved_params(evolved_params_values)
        self.last_random_seed = last_random_seed


    def next_seed(self):
        self.last_random_seed += 1
        return self.last_random_seed

    @classmethod
    def set_evolved_params(cls, evolved_params):
        cls._evolved_params = evolved_params


    def init_evolved_params(self, evolved_params_values=None):

        self.evolved_params_values = (evolved_params_values if evolved_params_values else {})
        if any((ep not in self._evolved_params for ep in self.evolved_params_values)):
            warnings.warn(
                "Unexpected evolved parameter passed to Individual constructor, this will not be mutated correctly!"
            )
        for param in self._evolved_params:
            if self.evolved_params_values.get(param) is None:
                self.evolved_params_values[param] = np.random.uniform(
                    self._evolved_params[param]["low"],
                    self._evolved_params[param]["high"],
                    self._evolved_params[param].get("shape"),
                )


    def refresh(self):
        self.clear_fitness()


    def clear_fitness(self):
        self.fitness = None
        self.fitness_components = None
        self.fitness_info = None

        self.fitness_history = deque(maxlen=self.remember_fitness) if self.remember_fitness > 0 else None

    def copy(self, **override_kwargs):
        ignored_attrs = ['id', 'gen', 'last_random_seed']
        params = {}

        for k, v in vars(self).items():
            if k in ignored_attrs:
                continue
            if Base_Individual.is_mutable(v):
                params[k] = deepcopy(v)
            else:
                params[k] = v

        params.update(override_kwargs)
        return type(self)(**params)

    @staticmethod
    def is_mutable(value):
        return isinstance(value, (list, dict, set, np.ndarray))

    @staticmethod
    def random_range(min, max, shape=None):
        if shape is None:
            return min + (max - min) * np.random.rand()
        else:
            return min + (max - min) * np.random.rand(*shape)

    @abstractmethod
    def mutate(self, strength):
        """
        :param strength: the strength of the mutation
        :return: a list of 1 or more new individuals (return empty list if mutation fails or not implemented)
        """

    @abstractmethod
    def crossover(self, other):
        """
        :param other: the other individual to crossover with
        :return: a list of 1 or more new individuals (return empty list if crossover fails or not implemented)
        """

    @staticmethod
    def crossover_evo_params(parents):
        """return new dict of evo params from randomly choosing between params of each parent"""
        evo_params = deepcopy(parents[0].evolved_params_values)
        for param, rnd in zip(evo_params, np.random.random(len(evo_params))):
            if rnd > 0.5:
                evo_params[param] = deepcopy(parents[1].evolved_params_values[param])
        return evo_params

    @staticmethod
    def gauss_mutate(x, std, low=None, high=None):
        x = np.random.normal(x, std)
        if low is not None or high is not None:
            x = np.clip(x, low, high)
        return x

    @staticmethod
    def mutate_evo_param(clone, strength):
        param_name = np.random.choice(list(clone.evolved_params_values))
        mut_param_info = clone._evolved_params[param_name]

        new_val = clone.gauss_mutate(
            clone.evolved_params_values[param_name],
            strength * (mut_param_info["high"] - mut_param_info["low"]) / 20,
        )

        res_info = f"{param_name} changed {clone.evolved_params_values[param_name]} -> {new_val}"

        if new_val == clone.evolved_params_values[param_name]:
            # mutation failed, terminate clone!
            clone = None
        else:
            clone.evolved_params_values[param_name] = new_val

        return res_info

    @classmethod
    def from_string(cls, string, **overide_kwargs):
        array = np.array
        kwargs = eval(string)
        kwargs.update(overide_kwargs)

        return cls(**kwargs)

    def push_fitness_history(self, fitness):
        self.fitness_history.appendleft(fitness)


    @staticmethod
    def get_default_shared_params(outdir="", gen=None, select_param=None):
        default_params = {
            "model": "CustomSpinIce",
            "encoder": "AngleSine",
            "radians": True,
        }
        if gen is not None:
            outdir = os.path.join(outdir, f"gen{gen}")
        default_params["basepath"] = outdir

        if select_param is not None:
            return default_params[select_param]

        return default_params


    def fast_tessellate(self, shape=(5, 1), padding=0, centre=True, return_labels=False):
        pos = self.coords
        angles = self.angles
        cell_size = pos.ptp(axis=0) + padding

        res = np.tile(pos, (np.prod(shape), 1))
        offsets = np.indices(shape).T.reshape(-1, 2) * cell_size
        res += offsets.repeat(len(pos), axis=0)

        if centre:
            res -= (0.5 * cell_size[0] * (shape[0]), 0.5 * cell_size[1] * (shape[1]))

        angles = np.tile(angles, np.prod(shape))

        if return_labels:
            labels = np.indices((np.prod(shape), len(pos))).reshape(2, -1).T
            return res, angles, labels
        else:
            return res, angles


    @classmethod
    def flatspin_eval(cls, pop, run_params, score_func, gen, outdir, *, max_jobs=1000, dont_run=False, **shared_params):

        if not pop:
            return pop

        run_type = shared_params.get("run", "local")
        shared_params["basepath"] = os.path.join(outdir, f"gen{gen}")
        wait = run_type == "local"
        if run_params:
            cls.evo_run(run_params, shared_params, gen,
                        max_jobs=max_jobs, wait=wait,
                        dont_run=dont_run)

            dataset = Dataset.read(shared_params["basepath"])
            process_dataset_local(dataset, score_func, wait)

        # mark unevaluated individuals
        id_cols = [k for k in run_params[0] if k.startswith("indv_id_")]
        evaluated = set(id for rp in run_params for k in id_cols for id in [rp[k]])
        for indv in pop:
            if indv.id not in evaluated:
                indv.fitness_components = [np.nan]

        return pop


    @classmethod
    def evo_run(cls, runs_params, shared_params, gen, wait=False, max_jobs=1000, dont_run=False):
        """modified from run_sweep.py main()"""
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


            sub_run_name = newparams.get("sub_run_name", f"_i{i}")
            id_cols = sorted([k for k in newparams if k.startswith("indv_id_")])
            indv_str = "v".join(str(newparams[k]) for k in id_cols)
            outdir = f"gen{gen}indv{indv_str}{sub_run_name}.{ext}"
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
            run_local(dataset)

        elif run_type == "dist":
            run_dist(dataset, wait=wait, max_jobs=max_jobs)

        np.random.set_state(rs)
        return


def generate_script(template, outfile, **params):
    with open(template) as fp:
        tpl = fp.read()
    script = tpl.format(**params)
    with open(outfile, 'w') as fp:
        fp.write(script)


def make_job_script(dataset, group_by, job_script_template):
    # Construct a sensible name for the job script
    job_script_dir = dataset.basepath
    job_script_name = os.path.basename(job_script_template)
    job_script = os.path.join(job_script_dir, job_script_name)

    # Job template params
    job_params = {
        'job_script_dir': job_script_dir,
        'job_script_name': job_script_name,
        'basepath': dataset.basepath,
    }

    return generate_script(job_script_template, job_script, **job_params)


def process_dataset_dist(dataset, id2indv, fit_func, shared_params, group_by):
    raise NotImplementedError("Distributed processing not implemented yet")
    queue = dataset
    job_script = make_job_script(dataset, group_by)


def process_dataset_local(dataset, fit_func, wait):
    queue = dataset

    queue = list(queue)
    while queue:
        ds = queue.pop(0)
        with np.errstate():
            try:
                fit_func(ds)
            except Exception as e:
                handle_exception(e, queue, ds, wait)#group_by)


def get_ds_indv_id(ds):
    unique = ds.index["indv_id"].unique()
    assert len(unique) == 1, "Dataset contains multiple individuals: {}".format(unique)
    return unique[0]







def handle_exception(e, queue, ds, wait=True):
    if wait:
        raise e
    if (not isinstance(e, FileNotFoundError) and
        not (isinstance(e, AssertionError) and "No vector data found for quantity: mag" in str(e)) and
        not "Bad magic number" in str(e)):
        print(type(e), e)
        traceback.print_exc()
    queue.append(ds)  # queue.append((indv_id, ds))
    sleep(2)


def overwrite_default_params(default_params, params):
    if params is None:
        return default_params
    else:
        default_params = copy(default_params)
        default_params.update(params)
        return default_params


def make_parser():
    import argparse
    from flatspin.cmdline import StoreKeyValue
    from collections import OrderedDict
    parser = argparse.ArgumentParser(description=__doc__)

    # common
    parser.add_argument("-o", "--output", metavar="FILE", help=r"¯\_(ツ)_/¯")
    parser.add_argument("-l", "--log", metavar="FILE", default="evo.log", help=r"name of the log file to create")
    parser.add_argument("-p", "--parameter", action=StoreKeyValue, default={},
                        help="param passed to flatspin and inner evaluate fitness function",)

    parser.add_argument(
        "-i",
        "--individual_param",
        action=StoreKeyValue,
        default={},
        help="param passed to Individual constructor",
    )
    parser.add_argument(
        "-f",
        "--outer_eval_param",
        action=StoreKeyValue,
        default={},
        help="param past to outer evaluate fitness function",
    )
    parser.add_argument(
        "-d",
        "--dependent_param",
        action=StoreKeyValue,
        default={},
        help="use for flatspin param that is dependent on other params (e.g. -e H=[0.5,1] -d 'H0=-H*2')"
    )

    parser.add_argument(
        "--calculate-fit-only",
        action="store_true",
        help="use if you only want to run a fitness func once on some individuals (don't run EA)",
    )

    return parser
