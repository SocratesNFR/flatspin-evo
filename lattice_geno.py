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
    basis_min = 0.3
    basis_max = 1.3
    min_angle_offset = np.deg2rad(10)

    def __init__(self, *, basis0=None, basis1=None, id=None, gen=0, angle_tile=None, angle_tile_shape=(3, 3), max_holes=None, angle_table=None,
                 num_angles=None, lattice_shape=(10, 10), pos=None, angle=None, parent_ids=None, **kwargs):
        self.gen = gen
        self.lattice_shape = lattice_shape

        if parent_ids is None:
            self.parent_ids = []
        else:
            self.parent_ids = parent_ids

        self.id = next(self._id_counter) if id is None else id

        self.basis0 = basis0 if basis0 is not None else np.array((np.random.rand() * (Individual.basis_max - Individual.basis_min) + Individual.basis_min, np.random.rand() * np.pi))
        self.basis1 = basis1 if basis1 is not None else np.array((np.random.rand() * (Individual.basis_max - Individual.basis_min) + Individual.basis_min, np.random.rand() * (np.pi - 2 * Individual.min_angle_offset) + Individual.min_angle_offset))

        if angle_tile is not None:
            self.angle_tile_shape = angle_tile.shape
        else:
            self.angle_tile_shape = angle_tile_shape

        self.num_angles = num_angles if num_angles is not None else np.prod(angle_tile_shape)
        self.angle_tile = angle_tile if angle_tile is not None else np.random.randint(0, self.num_angles, size=self.angle_tile_shape)
        if max_holes is None:
            self.max_holes = self.num_angles - 1
        else:
            self.max_holes = max_holes

        if angle_table is None:
            self.angle_table = [np.random.rand() * np.pi * 2 for _ in range(self.num_angles)]
            dropout = [((np.random.rand() < 0.7) * 2 - 1) for _ in range(self.max_holes)] + [1] * (self.num_angles - self.max_holes)
            np.random.shuffle(dropout)
            self.angle_table = np.array([a * d for a, d in zip(self.angle_table, dropout)])
        else:
            self.angle_table = angle_table

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

    def geno2pheno(self, lattice_shape=None):
        if lattice_shape is None:
            lattice_shape = self.lattice_shape

        # calculate basis from magnitude and angle
        basis0 = (self.basis0[0] * np.array((np.cos(self.basis0[1]), np.sin(self.basis0[1]))))
        b1_angle = self.basis1[1] + self.basis0[1]
        basis1 = (self.basis1[0] * np.array((np.cos(b1_angle), np.sin(b1_angle))))
        # find min magnitude
        min_magn = min([np.linalg.norm(basis0), np.linalg.norm(basis1)])

        # scale basis vectors
        scaled_basis0 = basis0 / min_magn
        scaled_basis1 = basis1 / min_magn

        grid = np.array(np.meshgrid(np.arange(lattice_shape[0]), np.arange(lattice_shape[1]))).T.reshape(-1, 2)
        pos = np.dot(grid, np.array([scaled_basis0, scaled_basis1]))
        angle = self.angle_table[self.angle_tile[np.mod(grid[:, 0], self.angle_tile_shape[0]), np.mod(grid[:, 1], self.angle_tile_shape[1])]]

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
    def gaussian_mutation(values, std, low=None, high=None, ignore_negative=False):
        negatives = values * (values < 0) if ignore_negative else None
        values = np.random.normal(values, std)
        if low is not None:
            values = np.maximum(values, low)
        if high is not None:
            values = np.minimum(values, high)

        if negatives is not None:  # restore negatives
            values = values * (negatives == 0) + negatives

        return values

    @staticmethod
    def swap_mutation(collection):
        i, j = np.random.choice(len(collection), 2, replace=False)
        collection[i], collection[j] = collection[j], collection[i]

    @staticmethod
    def mutate_bases(child, strength):
        std = (Individual.basis_max - Individual.basis_min) * strength * 0.05
        child.basis0[0] = Individual.gaussian_mutation(child.basis0[0], std, low=Individual.basis_min, high=Individual.basis_max)
        child.basis0[1] = Individual.gaussian_mutation(child.basis0[1], std, low=0, high=np.pi)

        child.basis1[0] = Individual.gaussian_mutation(child.basis1[0], std, low=Individual.basis_min, high=Individual.basis_max)
        child.basis1[1] = Individual.gaussian_mutation(child.basis1[1], std, low=Individual.min_angle_offset, high=np.pi - Individual.min_angle_offset)
        return child

    @staticmethod
    def mutate_angle_tile(child, strength):
        #np.random.randint(0, self.num_angles, size=self.angle_tile_shape)
        i = np.random.randint(0, np.prod(child.angle_tile_shape))
        if np.random.rand() < 0.5:
            child.angle_tile.flat[i] = np.random.randint(0, child.num_angles)
        else:
            j = np.random.choice([k for k in range(np.prod(child.angle_tile_shape)) if k != i])
            child.angle_tile.flat[i], child.angle_tile.flat[j] = child.angle_tile.flat[j], child.angle_tile.flat[i]


    @staticmethod
    def mutate_angle_table(child, strength):
        rand = np.random.rand()
        if rand < 0.4:
            std = strength * 0.05
            child.angle_table = Individual.gaussian_mutation(child.angle_table, std=std, low=0, high=np.pi * 2, ignore_negative=True)
        elif rand < 0.8 or child.max_holes == 0:
            Individual.swap_mutation(child.angle_table)
        else:
            indexs = list(range(len(child.angle_table)))
            negative = np.nonzero(child.angle_table < 0)[0]

            i = np.random.choice(indexs) if len(negative) < child.max_holes else np.random.choice(negative)  # choose negative angle if at max holes
            child.angle_table[i] = -1 if child.angle_table[i] >= 0 else np.random.rand() * np.pi * 2

# ======= Crossover helpers =======================================================

    @staticmethod
    def crossover_bases(child1, child2, parent2):
        if np.random.rand() < 0.5:
            child1.basis0 = parent2.basis0
        else:
            child2.basis0 = parent2.basis0
        if np.random.rand() < 0.5:
            child1.basis1 = parent2.basis1
        else:
            child2.basis1 = parent2.basis1

    @staticmethod
    def crossover_code_and_angle_table(child1, child2, parent2):
        for i in range(len(child1.code)):
            if np.random.rand() < 0.5:
                child1.code[i] = parent2.code[i]
                child1.angle_table[i] = parent2.angle_table[i]
            else:
                child2.code[i] = parent2.code[i]
                child2.angle_table[i] = parent2.angle_table[i]

    @staticmethod
    def crossover_evo_params(child1, child2, parent2):
        for param, rnd in zip(parent2._evolved_params, np.random.random(len(parent2._evolved_params))):
            if rnd > 0.5:
                child1.evolved_params_values[param] = deepcopy(parent2.evolved_params_values[param])
            else:
                child2.evolved_params_values[param] = deepcopy(parent2.evolved_params_values[param])
# ===================================================================================

    def mutate(self, strength=1):
        child = self.copy(parent_ids=[self.id])
        mutations = [Individual.mutate_bases, Individual.mutate_angle_tile, Individual.mutate_angle_table]
        weights = [1] * len(mutations)
        if len(self.evolved_params_values) > 0:
            mutations += [Individual.mutate_evo_param]
            # increase chance of selecting param-mutation by the num of evo params so they are picked evenly
            weights += [len(self.evolved_params_values)]
        mutation = np.random.choice(mutations, p=np.array(weights) / np.sum(weights))
        mutation(child, strength)
        return [child]

    def crossover(self, other):
        child1 = self.copy(parent_ids=[self.id, other.id])
        child2 = self.copy(parent_ids=[self.id, other.id])
        Individual.crossover_bases(child1, child2, other)
        Individual.crossover_code_and_angle_table(child1, child2, other)
        Individual.crossover_evo_params(child1, child2, other)
        return [child1, child2]

    def from_string(string, **overide_kwargs):
        array = np.array
        kwargs = eval(string)
        kwargs.update(overide_kwargs)

        return Individual(**kwargs)

    def __repr__(self):
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
            ind = Individual(code_len=10, pheno_bounds=(15, 15))
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
