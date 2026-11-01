from abc import ABC, abstractmethod

import numpy as np

from MixedEmitter import GenomeLayout


class Evaluator(ABC):
    """Base class for evaluators.

    An evaluator scores a batch of genomes. What `evaluate` returns (objectives,
    measures, constraints, ...) is up to the subclass; it just needs to match
    what the code consuming it expects (e.g. the archive you pass to `main`).

    Genome layout: [real genes | binary genes | strategy genes]. The strategy
    genes are mutation rates internal to the emitter and should not influence
    the evaluation.

    Args:
        layout: GenomeLayout describing the genome structure.
    """

    def __init__(self, layout: GenomeLayout):
        self.layout = layout

    def real_genes(self, solutions: np.ndarray) -> np.ndarray:
        """(batch, n_real) float array."""
        return solutions[:, self.layout.real]

    def binary_genes(self, solutions: np.ndarray) -> np.ndarray:
        """(batch, n_binary) int8 array of 0/1."""
        return solutions[:, self.layout.binary].astype(np.int8)

    @abstractmethod
    def evaluate(self, solutions: np.ndarray):
        """Evaluates a (batch, layout.solution_dim) array of genomes."""