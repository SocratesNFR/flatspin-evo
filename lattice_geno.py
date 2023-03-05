import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import os
import sys
import logging
from copy import deepcopy

from flatspin.model import TileLatticeSpinIce

from base_individual import Base_Individual
import fitness_functions
import evo_alg as ea


class Individual(Base_Individual):
    _id_counter = count(0)
    basis_min = 0.5
    basis_max = 1.1
    min_angle_offset = np.deg2rad(45)

    def __init__(self, *, lattice_shape=(10, 10), basis0_len=None, basis0_angle=None, basis1_len=None, basis1_angle=None, id=None, gen=0,
                 angle_tile_map=None, angle_tile_shape=(3, 3), angle_tile_max_dim=None, angle_table=None, num_angles=None,
                 hole_tile=None, hole_tile_shape=None, hole_tile_max_dim=None, min_holes=None, max_holes=None, parent_ids=None, min_magnets=None, **kwargs):
        self.gen = gen
        self._lattice_shape = lattice_shape
        self.min_magnets = min_magnets

        if parent_ids is None:
            self.parent_ids = []
        else:
            self.parent_ids = parent_ids

        self.id = next(self._id_counter) if id is None else id

        self.basis0_len = basis0_len if basis0_len is not None else random_range(Individual.basis_min, Individual.basis_max)
        self.basis0_angle = basis0_angle if basis0_angle is not None else random_range(0, np.pi)
        self.basis1_len = basis1_len if basis1_len is not None else random_range(Individual.basis_min, Individual.basis_max)
        self.basis1_angle = basis1_angle if basis1_angle is not None else random_range(Individual.min_angle_offset, np.pi - Individual.min_angle_offset)

        if angle_tile_map is not None:
            angle_tile_map = np.array(angle_tile_map) if not isinstance(angle_tile_map, np.ndarray) else angle_tile_map
            angle_tile_shape = angle_tile_map.shape
        else:
            if angle_tile_max_dim is not None:
                angle_tile_shape = (np.random.randint(1, angle_tile_max_dim + 1), np.random.randint(1, angle_tile_max_dim + 1))
        self.angle_tile_shape = angle_tile_shape
        self.angle_tile_max_dim = angle_tile_max_dim

        self.num_angles = num_angles
        self.angle_tile_map = angle_tile_map if angle_tile_map is not None else (
            np.random.randint(0, self.num_angles or np.prod(self.angle_tile_shape), size=self.angle_tile_shape)
        )

        self.angle_table = angle_table if angle_table is not None else random_range(0, 2 * np.pi, shape=(self.num_angles or np.prod(self.angle_tile_shape),))

        if hole_tile is not None:
            hole_tile = np.array(hole_tile) if not isinstance(hole_tile, np.ndarray) else hole_tile
            self.hole_tile_shape = hole_tile.shape
            self.hole_tile = hole_tile
        else:
            if hole_tile_max_dim is not None:
                hole_tile_shape = (np.random.randint(1, hole_tile_max_dim + 1), np.random.randint(1, hole_tile_max_dim + 1))
            self.hole_tile_shape = hole_tile_shape if hole_tile_shape is not None else self.angle_tile_shape
        self.hole_tile_max_dim = hole_tile_max_dim
        self._max_holes = max_holes

        self._min_holes = min_holes

        if hole_tile is None:
            num_holes = np.random.randint(self.min_holes, (self.max_holes) + 1)
            self.hole_tile = np.random.permutation(np.concatenate((np.zeros(num_holes), np.ones(np.prod(self.hole_tile_shape) - num_holes)))).reshape(self.hole_tile_shape)

        self.hole_tile = self.hole_tile.astype(int)
        self.fitness = None
        self.fitness_components = []
        self.fitness_info = []
        self._as_asi = None

        self.init_evolved_params(**kwargs)

    def is_in_bounds(self, point):
        return 0 <= point[0] <= self.pheno_bounds[0] and 0 <= point[1] <= self.pheno_bounds[1]

    def plot(self, **kwargs):
        self.as_ASI.plot(**kwargs)

    def num_magnets(self, lattice_shape=None):
        if lattice_shape is None:
            lattice_shape = self._lattice_shape

        div = (lattice_shape[0] // self.hole_tile_shape[0], lattice_shape[1] // self.hole_tile_shape[1])
        remainder = (lattice_shape[0] % self.hole_tile_shape[0], lattice_shape[1] % self.hole_tile_shape[1])

        total = np.sum(self.hole_tile) * div[0] * div[1]
        total += np.sum(self.hole_tile[:remainder[0], :] * div[1])
        total += np.sum(self.hole_tile[:, :remainder[1]] * div[0])

        return total

    @property
    def lattice_shape(self):
        """Returns the lattice shape, increasing it if necessary to satisfy the minimum number of magnets."""  
        if self.min_magnets is None:
            return self._lattice_shape

        num_mags = self.num_magnets()
        if num_mags >= self.min_magnets:
            return self._lattice_shape

        even_shape = (self._lattice_shape[0] - (self._lattice_shape[0] % self.hole_tile_shape[0]),
                     self._lattice_shape[1] - (self._lattice_shape[1] % self.hole_tile_shape[1])) # make hole_tile fit exactly into lattice_shape
        b = np.sum(even_shape)
        c = self.num_magnets(even_shape) - self.min_magnets
        base_increase = int(np.ceil((-b + np.sqrt(b * b - 4 * c)) / 2))
        bi_x = base_increase * self.hole_tile_shape[0]
        bi_y = base_increase * self.hole_tile_shape[1]
        increase = 0
        while self.num_magnets((self._lattice_shape[0] + bi_x + increase, self._lattice_shape[1] + bi_y + increase)) < self.min_magnets:
            increase += 1
            assert increase <= np.max(self.hole_tile.shape) + 1, f"Increase {increase} + {base_increase} + {np.max(self.hole_tile.shape)} is too large for hole_tile_shape {self.hole_tile.shape}"
        return (self._lattice_shape[0] + increase, self._lattice_shape[1] + increase)


    @property
    def basis0(self):
        return (self.basis0_len * np.array((np.cos(self.basis0_angle), np.sin(self.basis0_angle))))

    @property
    def basis1(self):
        b1_angle = self.basis1_angle + self.basis0_angle
        return (self.basis1_len * np.array((np.cos(b1_angle), np.sin(b1_angle))))

    @property
    def angle_tile(self):
        return self.angle_table[self.angle_tile_map]

    @property
    def min_holes(self):
        if self._min_holes is None:
            return 0
        if 0 < self._min_holes < 1:  # Fraction of holes
            return int(min(np.round(self._min_holes * np.prod(self.hole_tile_shape)), np.prod(self.hole_tile_shape) - 1))

        return int(self._min_holes)
    @property
    def max_holes(self):
        if self._max_holes is None:
            return int(np.prod(self.hole_tile_shape) - 1)

        if 0 < self._max_holes < 1: # Fraction of holes
            return int(min(np.round(self._max_holes * np.prod(self.hole_tile_shape)), np.prod(self.hole_tile_shape) - 1))

        return int(self._max_holes)

    @property
    def as_ASI(self):
        if self._as_asi is None:
            self._as_asi = TileLatticeSpinIce(basis0=self.basis0, basis1=self.basis1, angle_tile=self.angle_tile, hole_tile=self.hole_tile, radians=True, size=self.lattice_shape)
        return self._as_asi

    @property
    def coords(self):
        return self.as_ASI.pos

    @property
    def angles(self):
        return self.as_ASI.angle

    def reset(self):
        self.fitness = None
        self.fitness_components = []
        self.fitness_info = []
        self._as_asi = None

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
        if np.ndim(collection) < 2:
            i, j = np.random.choice(len(collection), 2, replace=False)
            collection[i], collection[j] = collection[j], collection[i]
        else:
            flat = np.array(collection).flatten()
            i, j = np.random.choice(len(flat), 2, replace=False)
            flat[i], flat[j] = flat[j], flat[i]
            collection = flat.reshape(collection.shape)
        return collection

    @staticmethod
    def mutate_bases(child, strength):
        mag_std = (Individual.basis_max - Individual.basis_min) * strength * 0.01
        angle_std = (2 * np.pi) * strength * 0.01
        child.basis0_len = Individual.gaussian_mutation(child.basis0_len, mag_std, low=Individual.basis_min, high=Individual.basis_max)
        child.basis0_angle = Individual.gaussian_mutation(child.basis0_angle, angle_std, low=0, high=2 * np.pi)

        child.basis1[0] = Individual.gaussian_mutation(child.basis1[0], mag_std, low=Individual.basis_min, high=Individual.basis_max)
        child.basis1[1] = Individual.gaussian_mutation(child.basis1[1], angle_std, low=Individual.min_angle_offset, high=np.pi - Individual.min_angle_offset)
        return child

    @ staticmethod
    def mutate_angle_table(child, strength):
        std = strength * 0.05
        child.angle_table = Individual.gaussian_mutation(child.angle_table, std=std, low=0, high=np.pi * 2)

    @ staticmethod
    def mutate_hole_tile(child, strength):
        if child.hole_tile_max_dim not in (None, 1) and np.random.rand() < 0.25:
            child._mutate_hole_tile_shape(strength)
        else:
            chance = min(strength * 0.05, 0.5)
            flat = child.hole_tile.flatten()
            holes = np.nonzero(flat == 0)[0]
            not_holes = np.nonzero(flat)[0]

            flippable_0 = np.random.choice(holes, len(holes) - child.min_holes, replace=False)
            flippable_1 = np.random.choice(not_holes, child.max_holes - len(holes), replace=False)
            flippable = np.concatenate((flippable_0, flippable_1))
            np.random.shuffle(flippable)
            do_flip = np.concatenate((np.ones(1), np.random.rand(len(flippable) - 1) < chance)).astype(bool)
            flat[flippable[do_flip]] = ~flat[flippable[do_flip]].astype(bool)
            child.hole_tile = flat.reshape(child.hole_tile.shape)
            assert child.min_holes <= np.sum(child.hole_tile == 0) <= child.max_holes, "Holes mutated out of bounds"

    def _mutate_hole_tile_shape(self, strength):
        assert self.hole_tile_max_dim not in (None, 1), "Cannot mutate hole tile shape if max dim is 1"
        dim = np.random.randint(0, 2)
        shape = list(self.hole_tile.shape)
        if shape[dim] == 1 or (np.random.rand() < 0.5 and shape[dim] <= self.hole_tile_max_dim):
            # make bigger in dim
            shape[dim] += 1
            new_tile = np.random.rand(*shape) < 0.5
            new_tile[:self.hole_tile.shape[0], :self.hole_tile.shape[1]] = self.hole_tile
        else:
            # make smaller in dim
            shape[dim] -= 1
            new_tile = self.hole_tile[:shape[0], :shape[1]]

        self.hole_tile = new_tile
        self.hole_tile_shape = new_tile.shape
        self._fix_holes()

    def _fix_holes(self):
        """Ensure that the number of holes is within the bounds"""
        num_holes = np.sum(self.hole_tile == 0)
        if num_holes < self.min_holes:
            self.hole_tile[np.random.choice(np.nonzero(self.hole_tile)[0], self.min_holes - num_holes, replace=False)] = 0
        elif num_holes > self.max_holes:
            self.hole_tile[np.random.choice(np.nonzero(self.hole_tile == 0)[0], num_holes - self.max_holes, replace=False)] = 1
        self.hole_tile = self.hole_tile.astype(int)

    @staticmethod
    def mutate_angle_tile_map(child, strength):
        if child.angle_tile_max_dim not in (None, 1) and np.random.rand() < 0.333:  # mutate shape
            child._mutate_angle_tile_shape(strength)
        elif np.random.rand() < 0.5 and np.prod(child.angle_tile_shape) > 1:  # swap angles
            child.angle_tile_map = Individual.swap_mutation(child.angle_tile_map)
        else:  # point mutation
            child.angle_tile_map.flat[np.random.randint(0, len(child.angle_tile_map.flat))] = np.random.randint(0, child.num_angles or np.prod(child.angle_tile_shape))

    def _mutate_angle_tile_shape(self, strength):
        assert self.angle_tile_max_dim not in (None, 1), "Cannot mutate shape if max dim is 1"
        dim = np.random.randint(0, 2)
        shape = list(self.angle_tile_shape)
        if shape[dim] == 1 or (np.random.rand() < 0.5 and shape[dim] <= self.angle_tile_max_dim):
            # make bigger in dim
            shape[dim] += 1
            new_map = np.random.randint(0, self.num_angles or np.prod(shape), size=shape)
            new_map[:self.angle_tile_shape[0], :self.angle_tile_shape[1]] = self.angle_tile_map
        else:
            # make smaller in dim
            shape[dim] -= 1
            new_map = self.angle_tile_map[:shape[0], :shape[1]]
        self.angle_tile_shape = tuple(shape)
        self.angle_tile_map = new_map



# ======= Crossover helpers =======================================================

    @staticmethod
    def crossover_bases(child1, child2, parent2):
        if np.random.rand() < 0.5:
            child1.basis0_len = parent2.basis0_len
            child1.basis0_angle = parent2.basis0_angle
        else:
            child2.basis0_len = parent2.basis0_len
            child2.basis0_angle = parent2.basis0_angle
        if np.random.rand() < 0.5:
            child1.basis1_len = parent2.basis1_len
            child1.basis1_angle = parent2.basis1_angle
        else:
            child2.basis1_len = parent2.basis1_len
            child2.basis1_angle = parent2.basis1_angle

    @staticmethod
    def crossover_angle_table_and_map(child1, child2, parent2):
        """Crossover angle table and angle tile map. if both parents tile_map/table are same shape, then children will have same shape."""
        for child in (child1, child2):
            child.angle_tile_map = Individual.crossover_arrays(child.angle_tile_map, parent2.angle_tile_map)
            child.angle_tile_shape = child.angle_tile_map.shape
            child.angle_table = Individual.crossover_arrays_1d(child.angle_table, parent2.angle_table,
                size=child.num_angles if child.num_angles is not None else np.prod(child.angle_tile_shape))

    @staticmethod
    def crossover_hole_tile(child1, child2, parent2):
        """Crossover hole tile. if both parents hole tile are same shape, then children will have same shape."""
        for child in (child1, child2):
            child.hole_tile = Individual.crossover_arrays(child.hole_tile, parent2.hole_tile)
            child.hole_tile_shape = child.hole_tile.shape
            child._fix_holes()

    @staticmethod
    def crossover_evo_params(child1, child2, parent2):
        for param, rnd in zip(parent2._evolved_params, np.random.random(len(parent2._evolved_params))):
            if rnd > 0.5:
                child1.evolved_params_values[param] = deepcopy(parent2.evolved_params_values[param])
            else:
                child2.evolved_params_values[param] = deepcopy(parent2.evolved_params_values[param])

    @staticmethod
    def crossover_arrays(arr1, arr2):
        """Crossover two arrays, can be different shapes but same dim, new shape is random between the two"""
        assert arr1.ndim == arr2.ndim, "Arrays must have same number of dimensions"
        if arr1.ndim == 1:
            return Individual.crossover_arrays_1d(arr1, arr2)
        assert arr1.ndim == 2, "Arrays must be 1d or 2d"
        shape = (Individual.randint_between(arr1.shape[0], arr2.shape[0]), Individual.randint_between(arr1.shape[1], arr2.shape[1]))
        rand = np.random.rand(*shape)
        arr = np.zeros_like(rand).astype(arr1.dtype)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i >= arr1.shape[0] or j >= arr1.shape[1]:    
                    arr[i, j] = arr2[i, j]
                elif i >= arr2.shape[0] or j >= arr2.shape[1]:
                    arr[i, j] = arr1[i, j]
                else:
                    arr[i, j] = arr1[i, j] if rand[i, j] > 0.5 else arr2[i, j]
        return arr

    @staticmethod
    def crossover_arrays_1d(arr1, arr2, size=None):
        """Crossover two 1d arrays, new shape is random between the two if sizee is None.
        If size is not None, the new array will be of that size, if extra values is needed,
        it will be filled with random samples from arr1 and arr2"""
        if size is None:
            size = Individual.randint_between(len(arr1), len(arr2))
        arr = np.zeros(size).astype(arr1.dtype)
        longest = arr1 if len(arr1) > len(arr2) else arr2
        shortest = arr1 if len(arr1) < len(arr2) else arr2
        arr[:len(shortest)] = np.where(np.random.rand(len(shortest)) > 0.5, shortest, longest[:len(shortest)])
        arr[len(shortest):len(longest)] = longest[len(shortest):len(longest)]
        if len(arr) > len(longest):
            diff = len(arr) - len(longest)
            arr[len(longest):] = np.random.choice(np.hstack((longest, shortest)), diff)
        return arr

    @staticmethod
    def randint_between(a, b, inclusive=True):
        if a == b:
            return a
        if a > b:
            a, b = b, a
        if inclusive:
            return np.random.randint(a, b + 1)
        return np.random.randint(a + 1, b)
# ===================================================================================

    def mutate(self, strength=1):
        child = self.copy(parent_ids=[self.id])
        mutations = [Individual.mutate_bases, Individual.mutate_angle_table, Individual.mutate_angle_tile_map]
        if self.max_holes != self.min_holes:
            mutations.append(Individual.mutate_hole_tile)

        weights = [1] * len(mutations)
        if len(self.evolved_params_values) > 0:
            mutations += [Individual.mutate_evo_param]
            # increase chance of selecting param-mutation by the num of evo params so they are picked evenly
            weights += [len(self.evolved_params_values)]
        mutation = np.random.choice(mutations, p=np.array(weights) / np.sum(weights))
        mutation(child, strength)
        child.reset()
        return [child]

    def crossover(self, other):
        child1 = self.copy(parent_ids=[self.id, other.id])
        child2 = self.copy(parent_ids=[self.id, other.id])
        Individual.crossover_bases(child1, child2, other)
        Individual.crossover_angle_table_and_map(child1, child2, other)
        Individual.crossover_hole_tile(child1, child2, other)
        Individual.crossover_evo_params(child1, child2, other)
        child1.reset()
        child2.reset()
        return [child1, child2]

    def from_string(string, keep_pheno=False, **overide_kwargs):
        array = np.array
        kwargs = eval(string)
        kwargs.update(overide_kwargs)

        return Individual(**kwargs)

    def __repr__(self):
        ignored_attrs = ['pos', 'angle']
        return repr({k: v for k, v in vars(self).items() if k not in ignored_attrs})

    def copy(self, **override_kwargs):
        ignored_attrs = ['pos', 'angle', 'id', 'gen']
        rename_attrs = {'_lattice_size': 'lattice_size', '_max_holes': 'max_holes', '_min_holes': 'min_holes'}
        params = {k: v for k, v in vars(self).items() if k not in ignored_attrs}
        params.update(override_kwargs)
        for old_name, new_name in rename_attrs.items():
            if old_name in params:
                params[new_name] = params.pop(old_name)

        return Individual(**params)

    @staticmethod
    def get_default_shared_params(outdir="", gen=None, select_param=None):
        default_params = {
            "model": "TileLatticeSpinIce",
            "encoder": "AngleSine",
            "radians": True,
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
                run_params.append(dict(rp, indv_id=id, basis0=indv.basis0, basis1=indv.basis1, angle_tile=indv.angle_tile, hole_tile=indv.hole_tile,
                size=indv.lattice_shape, sub_run_name=f"_{i}_{j}"))
        return run_params


def main(outdir=r"results\tileTest", inner="flips", outer="default", minimize_fitness=True, calculate_fit_only=False, **kwargs):
    known_fits = {

    }  # genotype-specific fitnesses

    inner = known_fits.get(inner, fitness_functions.known_fits.get(inner, inner))
    outer = known_fits.get(outer, fitness_functions.known_fits.get(outer, outer))

    if calculate_fit_only:
        return ea.only_run_fitness_func(outdir, Individual, inner, outer, minimize_fitness=minimize_fitness, **kwargs)
    else:
        return ea.main(outdir, Individual, inner, outer, minimize_fitness=minimize_fitness, **kwargs)


def random_range(min, max, shape=None):
    if shape is None:
        return min + (max - min) * np.random.rand()
    else:
        return min + (max - min) * np.random.rand(*shape)


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
