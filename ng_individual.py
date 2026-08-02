from collections import OrderedDict

import os
import nevergrad as ng
from base_individual import Base_Individual
import fitness_functions
import never_grad_ea as ea


class NG_Individual(Base_Individual):

    def __init__(self, params,  **kwargs):
        super().__init__(**kwargs)

        self.params = params
        self.special_params = []

    def genome2run_params(self, outdir):
        rp = {"indv_id": self.id}
        for param_name, param_value in self.params.items():
            if param_name not in self.special_params:
                rp[param_name] = param_value

        return rp

    @staticmethod
    def get_default_param_template(**kwargs):
        return ng.p.Dict()



def main(individual_class=NG_Individual):
    from flatspin.cmdline import StoreKeyValue, eval_params
    from base_individual import make_parser

    parser = make_parser()
    parser.add_argument("-e", "--evo_param", action=StoreKeyValue, default={},
                        help="""a flatspin parameter (float) to be tuned, format: -g param_name=[low, high] """)



    args = parser.parse_args()
    evo_params = eval_params(args.evo_param)

    outpath = os.path.join(os.path.curdir, args.output)
    logpath = os.path.join(outpath, args.log)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    inner_main(
        outdir=args.output,
        individual_class=individual_class,
        **eval_params(args.parameter),
        individual_params=eval_params(args.individual_param),
        outer_eval_params=eval_params(args.outer_eval_param),
        dependent_params=args.dependent_param,
        evolved_params=evo_params,
    )

def inner_main(outdir=r"results\ng_test", *,  individual_class=NG_Individual, inner="flips",
                minimize_fitness=True, **kwargs):
    known_fits = {

    }  # genotype-specific fitnesses

    inner = known_fits.get(
        inner, fitness_functions.known_fits.get(inner, inner))


    return ea.main(outdir, individual_class, inner, minimize_fitness=minimize_fitness, **kwargs)


if __name__ == "__main__":
    main()