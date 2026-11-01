from dataclasses import dataclass

import numpy as np

from ribs.emitters import EmitterBase

N_STRATEGY = 2  # [log_sigma, log_flip_prob]


@dataclass(frozen=True)
class GenomeLayout:
    """Genome layout: [real genes | binary genes | log_sigma, log_flip_prob]."""

    n_real: int
    n_binary: int

    @property
    def solution_dim(self):
        return self.n_real + self.n_binary + N_STRATEGY

    @property
    def real(self):
        return slice(0, self.n_real)

    @property
    def binary(self):
        return slice(self.n_real, self.n_real + self.n_binary)

    @property
    def strategy(self):
        return slice(self.n_real + self.n_binary, self.solution_dim)


class MixedMutationEmitter(EmitterBase):
    """Self-adaptive emitter: Gaussian mutation on real genes, bit flips on
    binary genes, with mutation rates stored in the genome itself.

    Args:
        archive: Archive to sample elites from.
        layout: GenomeLayout describing the genome structure.
        batch_size: Number of solutions per ask().
        tau: Std of the log-normal update applied to the strategy genes.
        sigma_range: (min, max) allowed Gaussian sigma.
        flip_prob_range: (min, max) allowed bit-flip probability. Defaults to
            (0.1 / n_binary, 0.5).
        seed: Random seed.
        real_range: (min, max) bounds for the real genes. Each can be a scalar
            (applied to all real genes) or an array of length n_real.
            Defaults to (0, 1).
        crossover_prob: Probability that a child is produced by crossover of
            two elites. Otherwise it is a copy of a single elite. Set to 0 to
            disable crossover. Defaults to 0.5.
    """

    def __init__(self, archive, *, layout, batch_size=32, tau=0.2,
                crossover_prob=0.5, real_range=(0.0, 1.0),
                sigma_range=(1e-3, 1.0), flip_prob_range=None, seed=None):
        # Compute ranges first so they can be passed to the base class.
        lo, hi = real_range
        real_lo = np.broadcast_to(np.asarray(lo, dtype=float), (layout.n_real,)).copy()
        real_hi = np.broadcast_to(np.asarray(hi, dtype=float), (layout.n_real,)).copy()
        if np.any(real_lo >= real_hi):
            raise ValueError("real_range min must be less than max for every gene")

        if flip_prob_range is None:
            flip_prob_range = (0.1 / layout.n_binary, 0.5)
        strategy_lo = np.log([sigma_range[0], flip_prob_range[0]])
        strategy_hi = np.log([sigma_range[1], flip_prob_range[1]])

        lower_bounds = np.concatenate([real_lo, np.zeros(layout.n_binary), strategy_lo])
        upper_bounds = np.concatenate([real_hi, np.ones(layout.n_binary), strategy_hi])

        super().__init__(
            archive,
            solution_dim=layout.solution_dim,
            bounds=None,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        self._layout = layout
        self._batch_size = batch_size
        self._tau = tau
        self._crossover_prob = crossover_prob
        self._real_lo = real_lo
        self._real_hi = real_hi
        self._strategy_lo = strategy_lo
        self._strategy_hi = strategy_hi
        self._rng = np.random.default_rng(seed)
   
    def _random_genomes(self, n):
        lay = self._layout
        genomes = np.empty((n, lay.solution_dim))
        genomes[:, lay.real] = self._rng.uniform(
            self._real_lo, self._real_hi, size=(n, lay.n_real))
        genomes[:, lay.binary] = self._rng.integers(0, 2, size=(n, lay.n_binary))
        genomes[:, lay.strategy] = self._rng.uniform(
            self._strategy_lo, self._strategy_hi, size=(n, N_STRATEGY))
        return genomes

    # --- mutation operators (each modifies `genomes` in place) ----------

    def _mutate_gaussian(self, genomes):
        lay = self._layout
        sigma = np.exp(genomes[:, lay.strategy][:, 0])[:, None]  # (n, 1)
        noise = self._rng.normal(size=(len(genomes), lay.n_real))
        genomes[:, lay.real] = np.clip(
            genomes[:, lay.real] + sigma * noise, self._real_lo, self._real_hi)

    def _mutate_strategy(self, genomes):
        """Log-normal update of log_sigma and log_flip_prob."""
        lay = self._layout
        noise = self._rng.normal(scale=self._tau, size=(len(genomes), N_STRATEGY))
        genomes[:, lay.strategy] = np.clip(
            genomes[:, lay.strategy] + noise, self._strategy_lo, self._strategy_hi)



    def _mutate_bitflip(self, genomes):
        lay = self._layout
        flip_prob = np.exp(genomes[:, lay.strategy][:, 1])[:, None]  # (n, 1)
        flips = self._rng.random((len(genomes), lay.n_binary)) < flip_prob
        bits = genomes[:, lay.binary]
        genomes[:, lay.binary] = np.where(flips, 1 - bits, bits)

    # --- crossover operator ---------------------------------------------

    def _crossover(self, parents_a, parents_b):
        """Returns children; parents_a is modified in place and returned."""
        lay = self._layout
        n = len(parents_a)
        do_cross = self._rng.random(n) < self._crossover_prob  # (n,)

        # Uniform crossover on binary genes.
        take_b = self._rng.random((n, lay.n_binary)) < 0.5
        take_b &= do_cross[:, None]
        parents_a[:, lay.binary] = np.where(
            take_b, parents_b[:, lay.binary], parents_a[:, lay.binary])

        # Arithmetic blend on real genes (per-gene weight).
        w = self._rng.random((n, lay.n_real))
        w = np.where(do_cross[:, None], w, 1.0)  # w=1 keeps parent A unchanged
        parents_a[:, lay.real] = (
            w * parents_a[:, lay.real] + (1 - w) * parents_b[:, lay.real])

        # Average strategy genes (in log space).
        mean_strategy = 0.5 * (parents_a[:, lay.strategy] + parents_b[:, lay.strategy])
        parents_a[:, lay.strategy] = np.where(
            do_cross[:, None], mean_strategy, parents_a[:, lay.strategy])

        return parents_a

    def ask(self):
        if self.archive.empty:
            return self._random_genomes(self._batch_size)

        n = self._batch_size
        parents_a = self.archive.sample_elites(n)["solution"].copy()
        parents_b = self.archive.sample_elites(n)["solution"].copy()

        genomes = self._crossover(parents_a, parents_b)
        self._mutate_strategy(genomes)  # strategy first, so children use their new rates
        self._mutate_gaussian(genomes)
        self._mutate_bitflip(genomes)
        return genomes