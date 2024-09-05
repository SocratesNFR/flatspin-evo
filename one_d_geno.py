# vim: tw=120
from joblib import Parallel
from tqdm.auto import tqdm
import logging
import numpy as np
import os

import evo_alg as ea
from base_individual import Base_Individual
import fitness_functions


class ProgressBar(tqdm):
    pass


class ParallelProgress(Parallel):
    def __init__(self, progress_bar, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_bar = progress_bar

    def print_progress(self):
        inc = self.n_completed_tasks - self._progress_bar.n
        self._progress_bar.update(inc)


class Individual(Base_Individual):

    def __init__(self, *, genome=None, min_len=1, max_len=1, **kwargs):

        super().__init__(**kwargs)

        self.min_len = min_len
        self.max_len = max_len

        self.genome = genome
        if genome is None:
            length = self.min_len
            if self.max_len > self.min_len:
                length = np.random.randint(self.min_len, self.max_len + 1)
            self.genome = random_range(0, 1, [length])


    def __repr__(self):
        # defines which attributes are ignored by repr
        ignore_attributes = []
        return repr({k: v for (k, v) in vars(self).items() if k not in ignore_attributes})



    def copy(self, **override_kwargs):
        ignored_attrs = ['id', 'gen']
        params = {k: v for k, v in vars(self).items() if k not in ignored_attrs}
        params.update(override_kwargs)
        if "genome" in params:
            params["genome"] = params["genome"].copy()

        return type(self)(**params)

    @staticmethod
    def get_default_shared_params(outdir="", gen=None, select_param=None):
        default_params = {
            "encoder": "AngleSine",
        }
        if select_param is not None:
            return default_params[select_param]
        if gen is not None:
            outdir = os.path.join(outdir, f"gen{gen}")
        default_params["basepath"] = outdir

        return default_params

    @staticmethod
    def get_default_run_params(pop, sweep_list=None, *, condition=None):
        sweep_list = sweep_list or [[0, 0, {}]]

        id2indv = {individual.id: individual for individual in [p for p in pop if condition is None or condition(p)]}

        run_params = []
        
        for id, indv in id2indv.items():
            for i, j, rp in sweep_list:
                run_params.append(dict(rp, indv_id=id, sub_run_name=f"_{i}_{j}", **indv.genome2run_params()))
        return run_params

    @classmethod
    def set_id_start(cls, start):
        cls._id_counter = count(start)

    def genome2run_params(self):
        """
        overide this with method to convert genome to run_params
        return a dictionary of run_params
        """
        return {}
    
    @property
    def coords(self) -> np.ndarray:
        return None

    @property
    def angles(self) -> np.ndarray:
        return None

    
    # ====================  Mutation and Crossover  ====================
    def mutate(self, strength=1):
        child = self.copy(parent_ids=[self.id])
        mutations = [Individual.point_mutate]


        weights = [1] * len(mutations)
        if len(self.evolved_params_values) > 0:
            mutations += [Individual.mutate_evo_param]
            # increase chance of selecting param-mutation by the num of evo params so they are picked evenly
            weights += [len(self.evolved_params_values)]
        mutation = np.random.choice(mutations, p=np.array(weights) / np.sum(weights))
        mutation(child, strength)
        child.refresh()
        return [child]

    @classmethod
    def point_mutate(cls, child, strength=1, floor=0, ceiling=1):
        strength /= 30
        indx = np.random.randint(0, len(child.genome))
        child.genome[indx] = np.random.normal(child.genome[indx], strength)
        child.genome[indx] = np.clip(child.genome[indx], floor, ceiling)

    
    
    def crossover(self, other):
        child1 = self.copy(parent_ids=[self.id, other.id])
        child2 = self.copy(parent_ids=[self.id, other.id])
        Individual.point_crossover(child1, child2)
        child1.refresh()
        child2.refresh()
        return [child1, child2]
    
    @classmethod
    def point_crossover(cls, child1, child2):
        if len (child1.genome) == 1 and len(child2.genome) == 1:
            return
        assert len(child1.genome) == len(child2.genome), "different genome lengths not implemented"
        indx = np.random.randint(1, len(child1.genome)-1)
        child1.genome[indx:], child2.genome[indx:] = child2.genome[indx:], child1.genome[indx:]

def random_range(min, max, shape=None):
    if shape is None:
        return min + (max - min) * np.random.rand()
    else:
        return min + (max - min) * np.random.rand(*shape)

def main(outdir=r"results\tileTest", inner="flips", outer="default", minimize_fitness=True, calculate_fit_only=False, **kwargs):
    known_fits = {

    }  # genotype-specific fitnesses

    inner = known_fits.get(inner, fitness_functions.known_fits.get(inner, inner))
    outer = known_fits.get(outer, fitness_functions.known_fits.get(outer, outer))

    if calculate_fit_only:
        return ea.only_run_fitness_func(outdir, Individual, inner, outer, minimize_fitness=minimize_fitness, **kwargs)
    else:
        return ea.main(outdir, Individual, inner, outer, minimize_fitness=minimize_fitness, **kwargs)




if __name__ == "__main__":
    if len(sys.argv) > 1:
        import argparse
        from flatspin.cmdline import StoreKeyValue, eval_params
        from base_individual import make_parser

        parser = make_parser()
        args = parser.parse_args()

        evolved_params = eval_params(args.evolved_param)


        outpath = os.path.join(os.path.curdir, args.output)
        logpath = os.path.join(outpath, args.log)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        logging.basicConfig(filename=logpath, level=logging.INFO)
        main(
            outdir=args.output,
            **eval_params(args.parameter),
            evolved_params=evolved_params,
            individual_params=eval_params(args.individual_param),
            outer_eval_params=eval_params(args.outer_eval_param),
            sweep_params=args.sweep_param,
            dependent_params=args.dependent_param,
            repeat=args.repeat,
            repeat_spec=args.repeat_spec,
            group_by=args.group_by,
            calculate_fit_only=args.calculate_fit_only,
        )
