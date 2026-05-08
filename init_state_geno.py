import numpy as np
import logging
import os
import sys
import base64
import zlib

import one_d_geno
import evo_alg as ea
import fitness_functions
from flatspin.data import save_table
from os import path


class Individual(one_d_geno.Individual):
    current_gen = 0
    def __init__(self, *, index_map=None, spin_count=1, fixed_val=None, gen2size=None, **kwargs):
        assert index_map, "index_map is required"
        assert spin_count >= len(
            index_map), f"spin_count {spin_count} must be >= len(index_map) {len(index_map)}"

        geno_size = len(index_map) + len(kwargs.get("genome_params", {}))

        min_len, max_len = kwargs.pop(
            "min_len", geno_size), kwargs.pop("max_len", geno_size)
        assert min_len == geno_size and max_len == geno_size, "genome length must be equal to spin_count + len(genome_params)"

        self.index_map = index_map
        self.spin_count = spin_count

        self.fixed_val = fixed_val
        self.gen2size = gen2size # [[0,10,20,30,40,50,60,70,80], [4,9,16,25,36,49,64,81,100]]
        if self.gen2size:
            assert isinstance(self.gen2size, (list, tuple)) and len(self.gen2size) == 2
            assert len(self.gen2size[0]) == len(self.gen2size[1]), "Mismatch in gen2size dimensions"


        super().__init__(min_len=min_len, max_len=max_len, **kwargs)


        self.fix()


    def genome2run_params(self, outdir, encode_and_save=True):
        rp = super().genome2run_params()
        init_state = np.zeros(self.spin_count) - 1

        state_genome = self.genome[len(self.genome_params): ] # the first genes are used for the genome params
        bin_genome = np.greater(state_genome, 0.5).astype(int)
        init_state[self.index_map] += 2 * bin_genome

        if encode_and_save:
            rp["init"] = self.encode_and_save_init(init_state, outdir)
        else:
                rp["init"] = init_state

        return rp

    @classmethod
    def encode_and_save_init(cls, init_state, outdir):
        bin_state = (init_state > 0).astype(int)
        zip_code = binstring2b64("".join(bin_state.astype(str)))
        dir = path.join(outdir, "init")
        fn = path.join(dir, f"init[{zip_code}].csv")
        if not os.path.exists(fn):
            if not os.path.exists(dir):
                os.makedirs(dir)
            save_table(init_state, fn)
        return fn


    def fix(self):
        if self.fixed_val is None or self.gen2size is None:
            return
        allowed_length = self.gen2size[1][np.searchsorted(self.gen2size[0], Individual.current_gen, side="right")-1]
        highest_index = len(self.genome_params) + allowed_length

        if highest_index >= len(self.genome):
            return

        self.genome[highest_index:] = self.fixed_val

    @staticmethod
    def get_default_run_params(pop, sweep_list=None, *, condition=None, outdir=None):
        sweep_list = sweep_list or [[0, 0, {}]]

        id2indv = {individual.id: individual for individual in [
            p for p in pop if condition is None or condition(p)]}

        run_params = []

        for id, indv in id2indv.items():
            for i, j, rp in sweep_list:
                run_params.append(
                    dict(rp, indv_id=id, sub_run_name=f"_{i}_{j}", **indv.genome2run_params(outdir)))
        return run_params

    def mutate(self, strength=1):

        [child] = super().mutate(strength=strength)
        child.fix()
        return [child]

    def crossover(self, other):

        gen1_params, gen1_state = self.genome[:len(self.genome_params)], self.genome[len(self.genome_params):]
        gen2_params, gen2_state = other.genome[:len(other.genome_params)], other.genome[len(other.genome_params):]

        genome = np.concatenate((self.line_crossover(gen1_params, gen2_params), self.point_crossover(gen1_state, gen2_state)))
        child = self.copy(parent_ids=[self.id, other.id])
        child.genome = genome
        child.fix()
        return [child]

def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')


def binstring2b64(bs):
    return base64.urlsafe_b64encode(zlib.compress(bitstring_to_bytes(bs)))


def b642binstring(str64, length=8):
    return format(int.from_bytes(zlib.decompress(base64.urlsafe_b64decode(str64))), f"0{length}b")


if __name__ == "__main__":
    one_d_geno.main(Individual)
