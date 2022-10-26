from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
import os
import warnings

numeric = Union[int, float, np.number]


class Base_Individual(ABC):
    id: int
    gen: int
    fitness: numeric
    fitness_components: List[numeric]
    fitness_info: List

    _evolved_params = {}

    @classmethod
    def set_evolved_params(cls, evolved_params):
        cls._evolved_params = evolved_params

    def init_evolved_params(self, evolved_params_values=None, **kwargs):

        self.evolved_params_values = (evolved_params_values if evolved_params_values else {})
        if any((ep not in self._evolved_params for ep in self.evolved_params_values)):
            warnings.warn(
                "Unexpected evolved parameter passed to Individual constructor, this will not be mutated correctly!"
            )
        for param in self._evolved_params:
            if self.evolved_params_values.get(param) is None:
                self.evolved_params_values[param] = np.random.uniform(
                    self._evolved_params[param]["low"],
                    self._evolved_params[param]["high"],
                    self._evolved_params[param].get("shape"),
                )

    @property
    @abstractmethod
    def coords(self) -> np.ndarray:
        """
        :return: the coordinates of the individual's magnets
        """

    @property
    @abstractmethod
    def angles(self) -> np.ndarray:
        """
        :return: the angles of the individual's magnets
        """

    @abstractmethod
    def mutate(self, strength):
        """
        :param strength: the strength of the mutation
        :return: a list of 1 or more new individuals (return empty list if mutation fails or not implemented)
        """

    @abstractmethod
    def crossover(self, other):
        """
        :param other: the other individual to crossover with
        :return: a list of 1 or more new individuals (return empty list if crossover fails or not implemented)
        """

    @abstractmethod
    def from_string(string):
        """
        :param string: a string representation of the individual
        :return: an individual from the string
        """

    @staticmethod
    def get_default_shared_params(outdir="", gen=None, select_param=None):
        default_params = {
            "model": "CustomSpinIce",
            "encoder": "AngleSine",
            "radians": True,
        }
        if gen is not None:
            outdir = os.path.join(outdir, f"gen{gen}")
        default_params["basepath"] = outdir

        if select_param is not None:
            return default_params[select_param]

        return default_params

    @staticmethod
    def get_default_run_params(pop, sweep_list=None, *, condition=None):
        sweep_list = sweep_list or [[0, 0, {}]]

        if not condition:
            def condition(indv):
                len(indv.coords) > 0

        id2indv = {individual.id: individual for individual in [p for p in pop if condition(p)]}

        run_params = []
        for id, indv in id2indv.items():
            for i, j, rp in sweep_list:
                run_params.append(dict(rp, indv_id=id, magnet_coords=indv.coords, magnet_angles=indv.angles, sub_run_name=f"_{i}_{j}"))

        return run_params

    def fast_tessellate(self, shape=(5, 1), padding=0, centre=True, return_labels=False):
        pos = self.coords
        angles = self.angles
        cell_size = pos.ptp(axis=0) + padding

        res = np.tile(pos, (np.prod(shape), 1))
        offsets = np.indices(shape).T.reshape(-1, 2) * cell_size
        res += offsets.repeat(len(pos), axis=0)

        if centre:
            res -= (0.5 * cell_size[0] * (shape[0]), 0.5 * cell_size[1] * (shape[1]))

        angles = np.tile(angles, np.prod(shape))

        if return_labels:
            labels = np.indices((np.prod(shape), len(pos))).reshape(2, -1).T
            return res, angles, labels
        else:
            return res, angles


def make_parser():
    import argparse
    from flatspin.cmdline import StoreKeyValue
    from collections import OrderedDict
    parser = argparse.ArgumentParser(description=__doc__)

    # common
    parser.add_argument("-o", "--output", metavar="FILE", help=r"¯\_(ツ)_/¯")
    parser.add_argument("-l", "--log", metavar="FILE", default="evo.log", help=r"name of the log file to create")
    parser.add_argument("-p", "--parameter", action=StoreKeyValue, default={},
                        help="param passed to flatspin and inner evaluate fitness function",)
    parser.add_argument(
        "-s",
        "--sweep_param",
        action=StoreKeyValue,
        default=OrderedDict(),
        help="flatspin param to be swept on each Individual evaluation",
    )
    parser.add_argument(
        "-n",
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="repeat each flatspin run N times (default: %(default)s)",
    )
    parser.add_argument(
        "-ns",
        "--repeat-spec",
        action=StoreKeyValue,
        metavar="key=SPEC",
        help="repeat each flatspin run according to key=SPEC",
    )
    parser.add_argument(
        "-e",
        "--evolved_param",
        action=StoreKeyValue,
        default={},
        help="""param passed to flatspin and inner evaluate that is under evolutionary control, format: -e param_name=[low, high] or -e param_name=[low, high, shape*]
                                int only values not supported""",
    )
    parser.add_argument(
        "--evo-rotate",
        action="store_true",
        help='short hand for "-e initial_rotation=[0,2*np.pi]"',
    )
    parser.add_argument(
        "-i",
        "--individual_param",
        action=StoreKeyValue,
        default={},
        help="param passed to Individual constructor",
    )
    parser.add_argument(
        "-f",
        "--outer_eval_param",
        action=StoreKeyValue,
        default={},
        help="param past to outer evaluate fitness function",
    )
    parser.add_argument(
        "-d",
        "--dependent_param",
        action=StoreKeyValue,
        default={},
        help="use for flatspin param that is dependent on other params (e.g. -e H=[0.5,1] -d 'H0=-H*2')"
    )
    parser.add_argument(
        "--group-by", nargs="*", help="group by parameter(s) for fitness evaluation"
    )
    parser.add_argument(
        "--calculate-fit-only",
        action="store_true",
        help="use if you only want to run a fitness func once on some individuals (don't run EA)",
    )

    return parser
