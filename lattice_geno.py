import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import os
import sys
import logging
from copy import deepcopy

from flatspin.model import CustomSpinIce

from base_individual import Base_Individual
import fitness_functions
import evo_alg as ea


class Individual(Base_Individual):
    _id_counter = count(0)
    basis_min = 0.5
    basis_max = 1

    def __init__(self, *, basis0=None, basis1=None, id=None, gen=0, code=None, code_len=10, angle_array=None, pheno_bounds=(25, 25), pos=None, angle=None, **kwargs):
        self.gen = gen
        self.pheno_bounds = pheno_bounds

        self.id = next(self._id_counter) if id is None else id

        self.basis0 = basis0 if basis0 is not None else np.array((np.random.rand() * (Individual.basis_max - Individual.basis_min) + Individual.basis_min, 0))
        self.basis1 = basis1 if basis1 is not None else np.array((np.random.rand() * Individual.basis_max, np.random.rand() * (Individual.basis_max - Individual.basis_min) + Individual.basis_min))

        self.code = code if code is not None else np.random.rand(code_len)
        if angle_array is None:
            self.angle_array = [np.random.rand() * np.pi * 2 for _ in range(code_len)]
            dropout = [1] + [((np.random.rand() < 0.7) * 2 - 1) for _ in range(code_len - 1)]
            np.random.shuffle(dropout)
            self.angle_array = np.array([a * d for a, d in zip(self.angle_array, dropout)])
        else:
            self.angle_array = angle_array

        if pos is None or angle is None:
            self.pos, self.angle = self.geno2pheno()
        else:
            self.pos, self.angle = pos, angle

        self.fitness = None
        self.fitness_components = []
        self.fitness_info = []

        self.init_evolved_params(**kwargs)

    def is_in_bounds(self, point):
        return 0 <= point[0] <= self.pheno_bounds[0] and 0 <= point[1] <= self.pheno_bounds[1]

    def perm_from_sort(self):
        perm = np.argsort(self.code)
        inv_perm = [np.where(perm == i)[0][0] for i in range(len(perm))]
        return perm, inv_perm

    def geno2pheno(self, pheno_bounds=None):
        if pheno_bounds is None:
            pheno_bounds = self.pheno_bounds

        # pos = np.array([x * self.basis0 + y * self.basis1 for x in range(pheno_bounds[0]) for y in range(pheno_bounds[1])])
        # angle = np.array([self.code[0] * x * np.pi + self.code[1] * y * np.pi for x in range(pheno_bounds[0]) for y in range(pheno_bounds[1])])

        frontier = [(0.0, 0.0)]
        symbol_frontier = [0]

        pos = []
        symbols = []

        seen = set(frontier)
        angle = []
        i = 0
        perm, inv_perm = self.perm_from_sort()

        # find min magnitude
        min_magn = min([np.linalg.norm(self.basis0), np.linalg.norm(self.basis1)])

        # scale basis vectors
        scaled_basis0 = self.basis0 / min_magn
        scaled_basis1 = self.basis1 / min_magn

        while frontier:
            point = frontier.pop()
            symbol = symbol_frontier.pop()
            new_pos = [tuple(np.round((point[0] + sign * b[0], point[1] + sign * b[1]), 5)) for b in (self.basis0, self.basis1) for sign in (-1, 1)]

            # new_sym = [(perm if basis == 0 else inv_perm)[symbol if sign == 1 else -1 - symbol] for basis in (0, 1) for sign in (-1, 1)]
            # new_sym = [(perm if basis * sign > 0 else inv_perm)[symbol] for basis in (1, -1) for sign in (1, -1)]
            new_sym = [(perm if basis == 0 else inv_perm)[symbol] for basis in (0, 1) for sign in (-1, 1)]

            new_pos, new_sym = list(zip(*[(p, s) for p, s in zip(new_pos, new_sym) if self.is_in_bounds(p) and p not in seen])) or [[], []]

            frontier.extend(new_pos)
            symbol_frontier.extend(new_sym)

            seen.update(new_pos)
            pos.append(point)
            symbols.append(symbol)
            i += 1

        angle = [self.angle_array[s] for s in symbols]

        # remove negative angles
        pos, angle = list(zip(*[(p, a) for p, a in zip(pos, angle) if a >= 0])) or [[], []]

        return pos, angle

    def plot(self, **kwargs):
        pos, angle = self.pos, self.angle
        if not pos:
            return
        csi = CustomSpinIce(magnet_coords=pos, magnet_angles=angle, radians=True)
        csi.plot(**kwargs)

    @property
    def coords(self):
        return self.pos

    @property
    def angles(self):
        return self.angle

    # ======= Mutation helpers =======================================================
    @staticmethod
    def gaussian_mutation(values, sigma, min=None, max=None, ignore_negative=False):
        negatives = values * (values < 0) if ignore_negative else None
        values = np.random.normal(values, sigma)
        if min is not None:
            values = np.maximum(values, min)
        if max is not None:
            values = np.minimum(values, max)

        if negatives is not None:  # restore negatives
            values = values * (negatives == 0) + negatives

        return values

    @staticmethod
    def swap_mutation(collection):
        i, j = np.random.choice(len(collection), 2, replace=False)
        collection[i], collection[j] = collection[j], collection[i]

    @staticmethod
    def mutate_bases(child, strength):
        sigma = (Individual.basis_max - Individual.basis_min) * strength * 0.05
        child.basis0 = Individual.gaussian_mutation(child.basis0, sigma=sigma, min=(Individual.basis_min, 0), max=(Individual.basis_max, 0))
        child.basis1 = Individual.gaussian_mutation(child.basis1, sigma=sigma, min=(0, Individual.basis_min), max=Individual.basis_max)

    @staticmethod
    def mutate_code(child, strength):
        if np.random.rand() < 0.5:
            sigma = strength * 0.05
            child.code = Individual.gaussian_mutation(child.code, sigma=sigma, min=0, max=1)
        else:
            Individual.swap_mutation(child.code)

    @staticmethod
    def mutate_angle_array(child, strength):
        rand = np.random.rand()
        if rand < 0.4:
            sigma = strength * 0.05
            child.angle_array = Individual.gaussian_mutation(child.angle_array, sigma=sigma, min=0, max=np.pi * 2, ignore_negative=True)
        elif rand < 0.8:
            Individual.swap_mutation(child.angle_array)
        else:
            indexs = list(range(len(child.angle_array)))
            positive = np.nonzero(child.angle_array >= 0)[0]
            if len(positive) == 1:
                indexs.remove(positive[0])
            if len(indexs) == 0:
                return
            i = np.random.choice(indexs)
            child.angle_array[i] = -1 if child.angle_array[i] >= 0 else np.random.rand() * np.pi * 2

# ======= Crossover helpers =======================================================

    @staticmethod
    def crossover_bases(child, parent2):
        child.basis0 = (child.basis0, parent2.basis0)[np.random.randint(2)]
        child.basis1 = (child.basis1, parent2.basis1)[np.random.randint(2)]

    @staticmethod
    def crossover_code_and_angle_array(child, parent2):
        for i in range(len(child.code)):
            if np.random.rand() < 0.5:
                child.code[i] = parent2.code[i]
                child.angle_array[i] = parent2.angle_array[i]
# ===================================================================================

    def mutate(self, strength=1):
        child = self.copy()
        mutations = [Individual.mutate_bases, Individual.mutate_code, Individual.mutate_angle_array]
        mutation = np.random.choice(mutations)
        mutation(child, strength)
        return [child]

    def crossover(self, other):
        child = self.copy()
        Individual.crossover_bases(child, other)
        Individual.crossover_code_and_angle_array(child, other)
        return [child]

    def from_string(string, **overide_kwargs):
        array = np.array
        kwargs = eval(string)
        kwargs.update(overide_kwargs)

        return Individual(**kwargs)

    def __repr__(self):
        print(self.__dict__)
        ignored_attrs = ['pos', 'angle']
        return repr({k: v for k, v in vars(self).items() if k not in ignored_attrs})




    def copy(self, **override_kwargs):
        ignored_attrs = ['pos', 'angle', 'id', 'gen']
        params = {k: v for k, v in vars(self).items() if k not in ignored_attrs}
        params.update(override_kwargs)

        return Individual(**params)

    @staticmethod
    def get_default_shared_params(outdir="", gen=None, select_param=None):
        default_params = {
            "model": "CustomSpinIce",
            "encoder": "AngleSine",
            "radians": True,
            "neighbor_distance": 10,
        }
        if select_param is not None:
            return default_params[select_param]
        if gen is not None:
            outdir = os.path.join(outdir, f"gen{gen}")
        default_params["basepath"] = outdir

        return default_params


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
        if args.evo_rotate:
            evolved_params["initial_rotation"] = [0, 2 * np.pi]

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
    else:
        plt.style.use("dark_background")

        indvs = []
        while len(indvs) < 25:
            ind = Individual(code_len=7, pheno_bounds=(15, 15))
            if ind.pos:
                indvs.append(ind)
        for i, indv in enumerate(indvs):

            plt.subplot(5, 5, i + 1)
            indv.plot()
            plt.title(f"ID: {indv.id}")
        plt.show()

        inp = input("choose individual to animate: ")
        indv = [i for i in indvs if i.id == int(inp)][0]
        pos, angle = indv.pos, indv.angle

        def step(i):
            plt.cla()
            csi = CustomSpinIce(magnet_coords=pos[:i + 1], magnet_angles=angle[:i + 1], radians=True)
            csi.plot()

        from matplotlib.animation import FuncAnimation
        fig = plt.figure()
        ani = FuncAnimation(fig, step, frames=len(pos), interval=0.1)
        plt.show()
        ani.save("lattice_geno.gif", fps=100)
