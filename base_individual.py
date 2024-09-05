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
from itertools import count

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
                **kwargs):

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

    @staticmethod
    def get_default_run_params(pop, sweep_list=None, *, condition=None):
        sweep_list = sweep_list or [[0, 0, {}]]

        if not condition:
            def condition(indv):
                return len(indv.coords) > 0

        id2indv = {individual.id: individual for individual in [p for p in pop if condition(p)]}

        run_params = []
        for id, indv in id2indv.items():
            for i, j, rp in sweep_list:
                run_params.append(dict(rp, indv_id=id, magnet_coords=indv.coords, magnet_angles=indv.angles, sub_run_name=f"_{i}_{j}"))

        return run_params

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
    def flatspin_eval(cls, fit_func, pop, gen, outdir, *, run_params=None, shared_params=None, use_default_shared_params=True,
                    sweep_params=None, condition=None, group_by=None, max_jobs=1000,
                    repeat=1, repeat_spec=None, preprocessing=None, dont_run=False, dependent_params={}, **flatspin_kwargs):
        """
        fit_func is a function that takes a dataset and produces an iterable (or single value) of fitness components.
        if an Individual already has fitness components the value(s) will be appended
        (allows for multiple datasets per Individual)
        using group_by, it is possible to use mutliple datasets to determine the fitness of an individual
        """

        if len(pop) < 1:
            return pop

        default_shared = cls.get_default_shared_params(outdir, gen)
        if use_default_shared_params:
            shared_params = overwrite_default_params(default_shared, shared_params)
        elif shared_params is None:
            shared_params = {}
        shared_params.update(flatspin_kwargs)

        sweep_list = (list(sweep(sweep_params, repeat, repeat_spec, params=shared_params)) if sweep_params else [])

        if run_params is None:
            run_params = cls.get_default_run_params(pop, sweep_list, condition=condition)

        if preprocessing:
            run_params = preprocessing(run_params)

        run_type = shared_params.get("run", "local")
        if len(run_params) > 0:
            id2indv = {individual.id: individual for individual in pop}
            evolved_params = [
                id2indv[rp["indv_id"]].evolved_params_values for rp in run_params
            ]
            wait = run_type == "local" or group_by
            cls.evo_run(run_params, shared_params, gen, evolved_params, max_jobs=max_jobs, wait=wait, dont_run=dont_run, dependent_params=dependent_params)
            dataset = Dataset.read(shared_params["basepath"])

            process_dataset_local(dataset, id2indv, fit_func, shared_params, group_by)
            """
            if run_type == "local":
                process_dataset_local(dataset, id2indv, fit_func, shared_params, group_by)
            elif run_type == "dist":
                process_dataset_dist(dataset, id2indv, fit_func, shared_params, group_by)
            else:
                raise ValueError("Unknown run type: {}".format(run_type))
            """
        # individuals that have not been evaluated (malformed)
        evaluated = set([rp["indv_id"] for rp in run_params])
        for indv in [i for i in pop if i.id not in evaluated]:
            indv.fitness_components = [np.nan]
        return pop


    @classmethod
    def evo_run(cls, runs_params, shared_params, gen, evolved_params=None, wait=False, max_jobs=1000, dont_run=False, dependent_params={}):
        """modified from run_sweep.py main()"""
        if not evolved_params:
            evolved_params = []
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
            if evolved_params:
                # get any flatspin params in evolved_params and update run param with them
                run_params.update(
                    {k: v for k, v in evolved_params[i].items() if k in newparams}
                )
            if dependent_params:
                # get any dependent params in dependent_params and update run param with them
                dp = eval_params(dependent_params, {**run_params, **evolved_params[i]})
                run_params.update(dp)

            sub_run_name = newparams.get("sub_run_name", "x")
            outdir = outdir_tpl.format(gen, newparams["indv_id"]) + f"{sub_run_name}.{ext}"
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
            run_local(dataset, False)

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


def process_dataset_local(dataset, id2indv, fit_func, shared_params, group_by):
    queue = dataset
    if group_by:
        _, queue = zip(*dataset.groupby(group_by))
    queue = list(queue)
    while queue:
        ds = queue.pop(0)
        with np.errstate():
            indv_id = get_ds_indv_id(ds)
            try:
                fit_components = calculate_fitness(ds, fit_func)
                assign_fitness(id2indv, indv_id, fit_components)
            except Exception as e:
                handle_exception(e, queue, ds, False)#group_by)


def get_ds_indv_id(ds):
    unique = ds.index["indv_id"].unique()
    assert len(unique) == 1, "Dataset contains multiple individuals: {}".format(unique)
    return unique[0]




def calculate_fitness(ds, fit_func):
    fit_components = fit_func(ds)
    try:
        fit_components = list(fit_components)
    except (TypeError):
        fit_components = [fit_components]
    return fit_components


def assign_fitness(id2indv, indv_id, fit_components):
    indv = id2indv[indv_id]
    if indv.fitness_components is not None:
        indv.fitness_components += fit_components
    else:
        indv.fitness_components = fit_components


def handle_exception(e, queue, ds, wait=True):
    if wait:
        raise e
    if not isinstance(e, FileNotFoundError):
        print(type(e), e)
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
        "-s",
        "--sweep_param",
        action=StoreKeyValue,
        default=OrderedDict(),
        help="flatspin param to be swept on each Individual evaluation",
    )
    parser.add_argument(
        "-n",
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="repeat each flatspin run N times (default: %(default)s)",
    )
    parser.add_argument(
        "-ns",
        "--repeat-spec",
        action=StoreKeyValue,
        metavar="key=SPEC",
        help="repeat each flatspin run according to key=SPEC",
    )
    parser.add_argument(
        "-e",
        "--evolved_param",
        action=StoreKeyValue,
        default={},
        help="""param passed to flatspin and inner evaluate that is under evolutionary control, format: -e param_name=[low, high] or -e param_name=[low, high, shape*]
                                int only values not supported""",
    )
    parser.add_argument(
        "--evo-rotate",
        action="store_true",
        help='short hand for "-e initial_rotation=[0,2*np.pi]"',
    )
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
        "--group-by", nargs="*", help="group by parameter(s) for fitness evaluation"
    )
    parser.add_argument(
        "--calculate-fit-only",
        action="store_true",
        help="use if you only want to run a fitness func once on some individuals (don't run EA)",
    )

    return parser
