import base64
import zlib
import nevergrad as ng
import numpy as np
import os

from ng_individual import NG_Individual, main as ng_main
from flatspin.data import save_table

class NG_Init_And_Clock_Individual(NG_Individual):

    def __init__(self, params, spin_count=100, index_map=None, **kwargs):
        super().__init__(params, **kwargs)
        self.spin_count = spin_count
        self.index_map = index_map
        assert self.index_map is not None, "index_map must be provided for NG_Init_And_Clock_Individual"
        self.special_params += ["init", "field_order"]

    @staticmethod
    def get_default_param_template(index_map=(), **kwargs):
        return ng.p.Dict(
            init=ng.p.Tuple(*[
                ng.p.Choice([False, True])
                for _ in range(len(index_map))
            ]),

            field_order=ng.p.Array(shape=(8,))
                .set_mutation(sigma=(1 - 0) / 6)
                .set_bounds(0, 1),
        )


    def genome2run_params(self, outdir):
        rp = super().genome2run_params(outdir)

        if "init" in self.params:
            self.run_param_init_state(outdir, rp)

        if "field_order" in self.params:
            self.run_param_field_order(outdir, rp)
        return rp

    def run_param_field_order(self, outdir, rp):
        fields = list("AaBbCcDd")
        values = self.params["field_order"]
        fields = [fields[i] for i in np.argsort(values) if values[i] <= 0.75]
        fields += ["X"] * (8 - len(fields)) # fill the rest with Xs
        rp["cycle"] = "".join(fields)
        """
        define pulses like this:
        -d "pulses={'A':(HA, aA), 'B':(HB, aB), 'C':(HC, aC), 'D':(HD, aD),
                    'a':(Ha, aa), 'b':(Hb, ab), 'c':(Hc, ac), 'd':( Hd, ad),
                    'X':(0, 0)}"
        """

    def run_param_init_state(self, outdir, rp):
        init_state = np.zeros(self.spin_count) - 1
        bin_genome = np.array(self.params["init"], dtype=int)
        init_state[self.index_map] += 2 * bin_genome # add 2 because init_state is all -1

        rp["init"] = encode_and_save_init(init_state, bin_genome, outdir)

def encode_and_save_init(init_state, bin_state, outdir):
        zip_code = binstring2b64("".join(bin_state.astype(str)))
        dir = os.path.join(outdir, "init")
        fn = os.path.join(dir, f"init[{zip_code}].csv")
        if not os.path.exists(fn):
            if not os.path.exists(dir):
                os.makedirs(dir)
            save_table(init_state, fn)
        return fn

def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')


def binstring2b64(bs):
    return base64.urlsafe_b64encode(zlib.compress(bitstring_to_bytes(bs)))


def b642binstring(str64, length=8):
    return format(int.from_bytes(zlib.decompress(base64.urlsafe_b64decode(str64))), f"0{length}b")

if __name__ == "__main__":
    ng_main(individual_class=NG_Init_And_Clock_Individual)