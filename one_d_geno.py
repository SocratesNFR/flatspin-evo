# vim: tw=120
from joblib import Parallel
from tqdm.auto import tqdm
import logging
import numpy as np
import os
from collections import OrderedDict

import evo_alg as ea
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

    @staticmethod
    def get_default_run_params(pop, sweep_list=None, *, condition=None, outdir=None):
        sweep_list = sweep_list or [[0, 0, {}]]

        id2indv = {individual.id: individual for individual in [
            p for p in pop if condition is None or condition(p)]}

        run_params = []

        for id, indv in id2indv.items():
            for i, j, rp in sweep_list:
                run_params.append(
                    dict(rp, indv_id=id, sub_run_name=f"_{i}_{j}", **indv.genome2run_params()))
        return run_params

    @classmethod
    def set_id_start(cls, start):
        cls._id_counter = count(start)

    def genome2run_params(self):
        """
        override this with method to convert genome to run_params
        return a dictionary of run_params
        """
        rp = {}
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
        mutations = [Individual.point_mutate, Individual.full_mutate]
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
        child = self.line_crossover(other)
        return [child]

    def line_crossover(self, other):
        assert len(self.genome) == len(
            other.genome), "different genome lengths not implemented"

        child = self.copy(parent_ids=[self.id, other.id])

        dist = np.random.rand()
        child.genome = dist * other.genome + (1 - dist) * self.genome
        return child




def inner_main(outdir=r"results\tileTest", *,  individual_class=Individual, inner="flips", outer="default",
                minimize_fitness=True, calculate_fit_only=False, map_elite=False, robust_map_elite=False, **kwargs):
    known_fits = {

    }  # genotype-specific fitnesses

    inner = known_fits.get(
        inner, fitness_functions.known_fits.get(inner, inner))
    outer = known_fits.get(
        outer, fitness_functions.known_fits.get(outer, outer))

    if map_elite:
        import map_elites
        return map_elites.main(outdir, individual_class, inner, minimize_fitness, **kwargs)

    if robust_map_elite:
        import robust_map_elites
        return robust_map_elites.main(outdir, individual_class, inner, minimize_fitness, **kwargs)

    if calculate_fit_only:
        return ea.only_run_fitness_func(outdir, individual_class, inner, outer, minimize_fitness=minimize_fitness, **kwargs)
    
    return ea.main(outdir, individual_class, inner, outer, minimize_fitness=minimize_fitness, **kwargs)


def main(individual_class=Individual):
    from flatspin.cmdline import StoreKeyValue, eval_params
    from base_individual import make_parser

    parser = make_parser()
    parser.add_argument("-g", "--genome_param", action=StoreKeyValue, default={},
                        help="""a flatspin parameter to be controlled by a gene in the genome, format: -g param_name=[low, high] """)
    parser.add_argument("--map-elite", action="store_true",
                        help="use map-elites algorithm", default=False)
    parser.add_argument("--robust-map-elite", action="store_true",
                        help="use roubust map-elites algorithm", default=False)

    args = parser.parse_args()

    evolved_params = eval_params(args.evolved_param)
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
        evolved_params=evolved_params,
        individual_params=eval_params(args.individual_param),
        outer_eval_params=eval_params(args.outer_eval_param),
        sweep_params=args.sweep_param,
        dependent_params=args.dependent_param,
        repeat=args.repeat,
        repeat_spec=args.repeat_spec,
        group_by=args.group_by,
        calculate_fit_only=args.calculate_fit_only,
        map_elite=args.map_elite,
        robust_map_elite=args.robust_map_elite
    )


if __name__ == "__main__":
    main()
