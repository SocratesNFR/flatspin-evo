import base64
import os
import zlib

import numpy as np

from flatspin.data import save_table

from flatspinEvaluator import FlatspinEvaluator

# Copy from your existing project (module path not verified):
# save_table


def binstring2b64(bs: str) -> str:
    """Compresses a bitstring ("0101...") into a short URL-safe string."""
    raw = int(bs, 2).to_bytes((len(bs) + 7) // 8, byteorder="big")
    return base64.urlsafe_b64encode(zlib.compress(raw)).decode("ascii")


def b642binstring(code: str, length: int) -> str:
    """Inverse of binstring2b64. `length` is needed to restore leading zeros."""
    raw = zlib.decompress(base64.urlsafe_b64decode(code))
    return format(int.from_bytes(raw, byteorder="big"), f"0{length}b")


class InitStateEvaluator(FlatspinEvaluator):
    """FlatspinEvaluator whose binary genes set the initial spin state.

    Each bit sets one spin (1 -> +1, 0 -> -1). Spins not covered by `index_map`
    stay at -1. The state is written to a CSV (named after the compressed
    bitstring, so identical genomes share one file) and its path is passed to
    the simulator as the "init" parameter.

    Args:
        gene_params, outdir, ...: see FlatspinEvaluator.
        spin_count: Total number of spins in the model.
        index_map: Spin index controlled by each binary gene. Defaults to
            range(spin_count), i.e. one bit per spin.
        **kwargs: Passed to FlatspinEvaluator (max_jobs, poll_interval).

    Subclasses still implement `analyse`.
    """

    def __init__(self, gene_params: dict, outdir: str, spin_count: int,
             index_map, **kwargs):
        
        if "init" in kwargs or "init" in gene_params:
            raise ValueError("'init' is set by InitStateEvaluator; remove it from the parameters")

        index_map = np.asarray(index_map, dtype=int)
        if len(np.unique(index_map)) != len(index_map):
            raise ValueError("index_map contains duplicate spin indices")
        if index_map.min() < 0 or index_map.max() >= spin_count:
            raise ValueError("index_map entries must be in [0, spin_count)")

        self.spin_count = spin_count
        self.index_map = index_map
        super().__init__(gene_params=gene_params, n_binary=len(index_map),
                        outdir=outdir, **kwargs)

        self.init_dir = os.path.abspath(os.path.join(outdir, "init"))

    def bits_to_params(self, bits: np.ndarray) -> dict:
        init_state = np.full(self.spin_count, -1.0)
        init_state[self.index_map] = 2 * bits - 1  # 0 -> -1, 1 -> +1
        return {"init": self.save_init(init_state, bits)}

    def save_init(self, init_state: np.ndarray, bits: np.ndarray) -> str:
        """Writes the init state to disk (once per distinct genome) and returns its path."""
        code = binstring2b64("".join(map(str, bits.tolist())))
        fn = os.path.join(self.init_dir, f"init_{code}.csv")
        if not os.path.exists(fn):
            os.makedirs(self.init_dir, exist_ok=True)
            save_table(init_state, fn)
        return fn

if __name__ == "__main__": # put this in your subclass, dont run this one
    InitStateEvaluator.main()