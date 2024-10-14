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
    def __init__(self, *, index_map=None, spin_count=1, **kwargs):
        self.index_map = index_map or []
        self.spin_count = spin_count
        super().__init__(**kwargs)

    def genome2run_params(self, outdir):
        rp = {}
        init_state = np.zeros(self.spin_count) - 1
        bin_genome = np.greater(self.genome, 0.5).astype(int)
        init_state[self.index_map] += 2 * bin_genome

        zip_code = "".join(bin_genome.astype(str))
        zip_code = binstring2b64(zip_code)

        dir = path.join(outdir, "init")
        fn = path.join(dir, f"init[{zip_code}].csv")
        rp["init"] = fn

        if not os.path.exists(fn):
            if not os.path.exists(dir):
                os.makedirs(dir)
            save_table(init_state, fn)

        return rp

    @staticmethod
    def get_default_run_params(pop, sweep_list=None, *, condition=None, outdir=None):
        sweep_list = sweep_list or [[0, 0, {}]]

        id2indv = {individual.id: individual for individual in [p for p in pop if condition is None or condition(p)]}

        run_params = []

        for id, indv in id2indv.items():
            for i, j, rp in sweep_list:
                run_params.append(dict(rp, indv_id=id, sub_run_name=f"_{i}_{j}", **indv.genome2run_params(outdir)))
        return run_params


def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')

def binstring2b64(bs):
    return base64.urlsafe_b64encode(zlib.compress(bitstring_to_bytes(bs)))

def b642binstring(str64,length=8):
    return format(int.from_bytes(zlib.decompress(base64.urlsafe_b64decode(str64))), f"0{length}b")


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

