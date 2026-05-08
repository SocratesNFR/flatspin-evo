# vim: tw=120
from joblib import Parallel
from tqdm.auto import tqdm
import logging
import numpy as np
import os
from collections import OrderedDict

import adversEa as ea
from base_individual import Base_Individual
import fitness_functions
import copy

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

    def __init__(self, *, genome=None, min_len=1, max_len=1, genome_params=None, **kwargs):

        super().__init__(**kwargs)

        self.min_len = min_len
        self.max_len = max_len

        self.genome_params = genome_params or {}
        assert self.min_len >= len(
            genome_params), "not enough genes for the genome_params"

        self.genome = genome
        if genome is None:
            length = self.min_len
            if self.max_len > self.min_len:
                length = np.random.randint(self.min_len, self.max_len + 1)
            self.genome = Individual.random_range(0, 1, [length])

    def __repr__(self):
        # defines which attributes are ignored by repr
        ignore_attributes = []
        return repr({k: v for (k, v) in vars(self).items() if k not in ignore_attributes})


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


    @classmethod
    def set_id_start(cls, start):
        cls._id_counter = count(start)

    def genome2run_params(self):
        """
        override this with method to convert genome to run_params
        return a dictionary of run_params
        """
        rp = {"indv_id": self.id}
        for i, (gp, val) in enumerate(self.genome_params.items()):
            rp[gp] = val[0] + (val[1] - val[0]) * self.genome[i]

        return rp

    @property
    def coords(self) -> np.ndarray:
        return None

    @property
    def angles(self) -> np.ndarray:
        return None

    # ====================  Mutation and Crossover  ====================

    def mutate(self, strength=1):
        child = self.copy(parent_ids=[self.id])
        # mutations = [Individual.point_mutate, Individual.full_mutate]
        mutations = [Individual.full_mutate]

        weights = [1] * len(mutations)
        if len(self.evolved_params_values) > 0:
            mutations += [Individual.mutate_evo_param]
            # increase chance of selecting param-mutation by the num of evo params so they are picked evenly
            weights += [len(self.evolved_params_values)]
        mutation = np.random.choice(
            mutations, p=np.array(weights) / np.sum(weights))
        mutation(child, strength)
        child.refresh()
        return [child]

    @classmethod
    def point_mutate(cls, child, strength=1, floor=0, ceiling=1):
        strength /= 30
        indx = np.random.randint(0, len(child.genome))
        child.genome[indx] = np.random.normal(child.genome[indx], strength)
        child.genome[indx] = np.clip(child.genome[indx], floor, ceiling)

    @classmethod
    def full_mutate(cls, child, strength=1, floor=0, ceiling=1):
        strength *=  0.01
        child.genome = np.random.normal(child.genome, strength)
        child.genome = np.clip(child.genome, floor, ceiling)

    def crossover(self, other):
        child = self.copy(parent_ids=[self.id, other.id])
        child.genome = self.line_crossover(self.genome, other.genome)
        return [child]

    @classmethod
    def point_crossover(cls, genome1, genome2):
        assert len(genome1) == len(
            genome2), "different genome lengths not implemented"

        if np.random.rand() < 0.5:
            genome1, genome2 = genome2, genome1 # shuffle which is first

        indx = np.random.randint(0, len(genome1))
        child_genome = np.concatenate((genome1[:indx], genome2[indx:]))
        return child_genome

    @classmethod
    def line_crossover(cls, genome1, genome2):
        assert len(genome1) == len(
            genome2), "different genome lengths not implemented"

        dist = np.random.rand()
        return dist * genome2 + (1 - dist) * genome1



def inner_main(outdir=r"results\tileTest", *,  individual_class=Individual, inner="flips", outer="default",
                minimize_fitness=True, **kwargs):
    known_fits = {

    }  # genotype-specific fitnesses

    inner = known_fits.get(
        inner, fitness_functions.known_fits.get(inner, inner))
    outer = known_fits.get(
        outer, fitness_functions.known_fits.get(outer, outer))


    return ea.main(outdir, individual_class, inner, outer, minimize_fitness=minimize_fitness, **kwargs)


def main(individual_class=Individual):
    from flatspin.cmdline import StoreKeyValue, eval_params
    from base_individual import make_parser

    parser = make_parser()
    parser.add_argument("-g", "--genome_param", action=StoreKeyValue, default={},
                        help="""a flatspin parameter to be controlled by a gene in the genome, format: -g param_name=[low, high] """)

    parser.add_argument("-pi", "--pop_specific_params", action=StoreKeyValue, default={},
                        help=""" inidividual parameters (like -i) but per population, format: -pi keyword=[param1, param2, ... paramN] where N is the number of populations.""")


    args = parser.parse_args()

    pop_specific_params = eval_params(args.pop_specific_params)
    genome_params = eval_params(args.genome_param)

    args.individual_param["genome_params"] = OrderedDict(genome_params)

    outpath = os.path.join(os.path.curdir, args.output)
    logpath = os.path.join(outpath, args.log)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    logging.basicConfig(filename=logpath, level=logging.INFO)
    inner_main(
        outdir=args.output,
        individual_class=individual_class,
        **eval_params(args.parameter),
        individual_params=eval_params(args.individual_param),
        outer_eval_params=eval_params(args.outer_eval_param),
        dependent_params=args.dependent_param,
        pop_specific_params=pop_specific_params,
    )


if __name__ == "__main__":
    main()
