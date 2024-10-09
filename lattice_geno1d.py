import numpy as np
import logging
import os
import sys

import one_d_geno
import evo_alg as ea
import fitness_functions
from flatspin.model import TileLatticeSpinIce

class Individual(one_d_geno.Individual):
    """Genome = [basis0_len, basis0_angle, basis1_len, basis1_angle_offset, angles, angle_table, hole_tile]"""
    basis_min = 0.5
    basis_max = 1.1
    min_angle_offset = np.deg2rad(45)

    def __init__(self, *, lattice_shape=(10, 10), angle_tile_shape=(2,2), num_angles=2,
                 hole_tile_shape=(2,2), min_holes=0, max_holes=0, min_magnets=None, **kwargs):
        self.lattice_shape = lattice_shape
        self.angle_tile_shape = angle_tile_shape
        self.num_angles = num_angles
        self.hole_tile_shape = hole_tile_shape
        self.min_holes = min_holes
        self.max_holes = max_holes
        assert self.min_holes <= self.max_holes
        assert self.max_holes < np.prod(self.hole_tile_shape)
        self.min_magnets = min_magnets

        self.g_idx = self.make_genome_index()
        length = self.g_idx["end"]
        kwargs["max_len"] = length
        kwargs["min_len"] = length
        super().__init__(**kwargs)


    def make_genome_index(self):
        index = dict(basis0_len=0, basis0_angle=1, basis1_len=2, basis1_angle_offset=3)
        i = len(index) # 4
        index["angles"] = i
        i+= self.num_angles
        index["angle_table"] = i
        i+= np.prod(self.angle_tile_shape)
        index["hole_tile"] = i
        i+= np.prod(self.hole_tile_shape)
        index["end"] = i

        return index


    def genome2run_params(self):
        """Return the flatspin parameters to simulate the phenotype"""
        rp= dict(basis0=self.basis0, basis1=self.basis1, angle_tile=self.angle_tile, hole_tile=self.hole_tile,
                size=self.flatspin_size)

        return rp

    @property
    def as_ASI(self):
        return TileLatticeSpinIce(basis0=self.basis0, basis1=self.basis1, angle_tile=self.angle_tile, hole_tile=self.hole_tile,
                size=self.flatspin_size, radians=True)


    def plot(self, **kwargs):
        self.as_ASI.plot(**kwargs)

    def num_magnets(self, lattice_shape=None):
        if lattice_shape is None:
            lattice_shape = self.lattice_shape

        div = (lattice_shape[0] // self.hole_tile_shape[0], lattice_shape[1] // self.hole_tile_shape[1])
        remainder = (lattice_shape[0] % self.hole_tile_shape[0], lattice_shape[1] % self.hole_tile_shape[1])

        total = np.sum(self.hole_tile) * div[0] * div[1]
        total += np.sum(self.hole_tile[:remainder[0], :] * div[1])
        total += np.sum(self.hole_tile[:, :remainder[1]] * div[0])
        total += np.sum(self.hole_tile[:remainder[0], :remainder[1]])

        return total

    @property
    def flatspin_size(self):
        """Returns the lattice shape, increasing it if necessary to satisfy the minimum number of magnets."""
        if self.min_magnets is None:
            return self.lattice_shape

        num_mags = self.num_magnets()
        if num_mags >= self.min_magnets:
            return self.lattice_shape

        even_shape = (self.lattice_shape[0] - (self.lattice_shape[0] % self.hole_tile_shape[0]),
                     self.lattice_shape[1] - (self.lattice_shape[1] % self.hole_tile_shape[1])) # make hole_tile fit exactly into lattice_shape
        b = np.sum(even_shape)
        c = self.num_magnets(even_shape) - self.min_magnets
        base_increase = int(np.ceil((-b + np.sqrt(b * b - 4 * c)) / 2))
        bi_x = base_increase * self.hole_tile_shape[0]
        bi_y = base_increase * self.hole_tile_shape[1]
        increase = 0
        while self.num_magnets((self.lattice_shape[0] + bi_x + increase, self.lattice_shape[1] + bi_y + increase)) < self.min_magnets:
            increase += 1
            assert increase <= np.max(self.hole_tile.shape) + 1, f"Increase {increase} + {base_increase} + {np.max(self.hole_tile.shape)} is too large for hole_tile_shape {self.hole_tile.shape}"
        return (self.lattice_shape[0] + bi_x + increase, self.lattice_shape[1] + bi_y + increase)


    @property
    def basis0_len(self):
        return unit2range(self.genome[self.g_idx["basis0_len"]], Individual.basis_min, Individual.basis_max)

    @property
    def basis0_angle(self):
        return unit2range(self.genome[self.g_idx["basis0_angle"]], 0, 2 * np.pi)

    @property
    def basis0(self):
        return (self.basis0_len * np.array((np.cos(self.basis0_angle), np.sin(self.basis0_angle))))


    @property
    def basis1_len(self):
        return unit2range(self.genome[self.g_idx["basis1_len"]], Individual.basis_min, Individual.basis_max)

    @property
    def basis1_angle_offset(self):
        return unit2range(self.genome[self.g_idx["basis1_angle_offset"]], Individual.min_angle_offset, np.pi - Individual.min_angle_offset)

    @property
    def basis1(self):
        b1_angle = self.basis1_angle_offset + self.basis0_angle
        return (self.basis1_len * np.array((np.cos(b1_angle), np.sin(b1_angle))))

    @property
    def angles(self):
        angs = self.genome[self.g_idx["angles"] : self.g_idx["angle_table"]]
        return unit2range(np.array(angs), 0, 2 * np.pi)

    @property
    def angle_table(self):
        table = self.genome[self.g_idx["angle_table"] : self.g_idx["hole_tile"]]
        table = unit2intRange(table, 0, self.num_angles-1)
        return np.reshape(table, self.angle_tile_shape)

    @property
    def angle_tile(self):
        return self.angles[self.angle_table]

    @property
    def hole_tile(self):
        """handles min/max hole constraints"""
        holes = np.array(self.genome[self.g_idx["hole_tile"] :])

        # round values less that 0.5 to holes
        hole_idx = np.where(holes < 0.5)
        not_hole_idx = np.where(holes >= 0.5)

        # add/subtract holes to meet constraints ordered by their value (closest to becoming a hole/ not hole)
        if len(hole_idx) > self.max_holes:
            hole_idx = sorted(hole_idx, key=lambda i: holes[i])[:self.max_holes]

        elif len(hole_idx) < self.min_holes:
            hole_deficit =self.min_holes - len(hole_idx)
            hole_idx += sorted(not_hole_idx, key=lambda i: holes[i])[:hole_deficit]

        holes = np.ones_like(holes)
        holes[hole_idx] = 0
        hole_tile = np.reshape(holes, self.hole_tile_shape)

        return hole_tile



def unit2range(x, low, high):
    """convert x in [0,1] to [low,high]"""
    return low + x * (high - low)

def unit2intRange(x, low, high):
    """convert x in [0,1] to [low,high] and round to int"""
    x = unit2range(x, low, high+1)
    return np.clip(np.trunc(x).astype(int), low, high)

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

