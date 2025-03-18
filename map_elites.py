import numpy as np
import pandas as pd

from datetime import datetime
from collections import OrderedDict
import shlex
import os
import sys
from warnings import warn
from flatspin.data import Dataset, read_csv
import pickle as pkl


class EliteMap:
    def __init__(self, shape, bounds=None, minimize_fitness=False, indvs=None):
        self.shape = np.array(shape)
        self.map = CoordMap(self.shape)

        if bounds is None:
            bounds = np.array([(0, 1)] * len(shape)).T  # Default auto bounds
            self._auto_bounds = np.ones(len(shape), dtype=bool)  # All dimensions auto-expand
        else:
            processed_bounds = []
            auto_bounds = []

            for b in bounds:
                if b is None:
                    processed_bounds.append((0, 1))  # Default range for auto bounds
                    auto_bounds.append(True)
                else:
                    assert len(b) == 2 and b[0] < b[1], "Each bound must be (min, max) with min < max"
                    processed_bounds.append(b)
                    auto_bounds.append(False)

            self.bounds = np.array(processed_bounds).T
            self._auto_bounds = np.array(auto_bounds, dtype=bool)

        assert np.shape(self.bounds) == (2, len(shape)), "bounds must be shape (2, len(shape))"

        self._minimize_fitness = minimize_fitness
        self.add(indvs)

    def coords_in_map(self, coords):
        coords = np.array(coords)

        # Clip coordinates to be within bounds
        clipped_coords = np.clip(coords, self.bounds[0], self.bounds[1])

        # Normalize and scale to the shape
        normed_coords = (clipped_coords - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        shape_coords = np.floor(normed_coords * (self.shape - 1)).astype(int)

        # Return the final shape coordinates
        return tuple(shape_coords)

    def better_than(self, indv1, indv2):

        if np.isnan(indv1.fitness):
            return False
        if np.isnan(indv2.fitness):
            return True

        if self._minimize_fitness:
            return indv1.fitness < indv2.fitness
        else:
            return indv1.fitness > indv2.fitness

    def add(self, indvs):
        if not indvs:
            return

        if self._auto_bounds.any():
            self.expand_bounds(indvs)

        for indv in indvs:
            coords = self.coords_in_map(indv.novelty)
            if coords != None and (
                self.map[coords] is None or self.better_than(
                    indv, self.map[coords])
            ):
                self.map[coords] = indv

    def expand_bounds(self, indvs):
        if not indvs or not np.any(self._auto_bounds):
            return

        fitnesses = np.array([indv.novelty for indv in indvs])
        indv_bounds = np.array([fitnesses.min(axis=0), fitnesses.max(axis=0)])

        bounds = self.bounds.copy()
        bounds[0][self._auto_bounds] = np.minimum(bounds[0][self._auto_bounds], indv_bounds[0][self._auto_bounds])
        bounds[1][self._auto_bounds] = np.maximum(bounds[1][self._auto_bounds], indv_bounds[1][self._auto_bounds])

        if np.all(bounds == self.bounds):  # No change
            return

        self.bounds = bounds
        self.remap()

    def remap(self):
        indvs = list(self.map.values())
        self.map.clear()
        self.add(indvs)


class CoordMap:
    def __init__(self, shape):
        self.shape = np.array(shape)  # (height, width)
        self.data = {}  # Internal dictionary to store values
        self._cache_values = None

    def __setitem__(self, coords, value):
        """Sets a value in the map at the given coordinate if it's valid."""
        coords = self.validate_coords(coords)
        self.data[coords] = value
        self._cache_values = None

    def __getitem__(self, coords):
        """Gets a value from the map at the given coordinate if it's valid."""
        coords = self.validate_coords(coords)
        return self.data.get(coords, None)  # Return None if key doesn't exist

    def __contains__(self, coords):
        """Checks if a coordinate exists in the map."""
        index = self._coords_to_index(coords)
        return index in self.data if index is not None else False

    def __repr__(self):
        """Returns a string representation of the stored data."""
        return f"CoordMap(data={self.data})"

    def validate_coords(self, coords):
        """Validates that the given coordinates are within the map's bounds."""
        coords = np.array(coords)
        if not (np.all(coords >= 0) and np.all(coords < self.shape)):
            raise ValueError("Coordinates are out of bounds.")

        if len(coords) != len(self.shape):
            raise ValueError(
                "Coordinates have the wrong number of dimensions.")

        if coords.dtype != int:
            raise ValueError("Coordinates must be integers.")

        return tuple(coords)

    def keys(self):
        """Returns the keys of the internal dictionary."""
        return self.data.keys()

    def values(self):
        """Returns the values of the internal dictionary."""
        return self.data.values()

    def list_values(self):
        if self._cache_values is None:
            self._cache_values = list(self.data.values())
        return self._cache_values

    def items(self):
        """Returns the items of the internal dictionary."""
        return self.data.items()

    def clear(self):
        self.data.clear()
        self._cache_values = None


def update_superdataset(
    dataset, outdir, elite_map, gen, minimize_fitness=True, dataset_params=None
):
    # pop = list(filter(lambda indv: np.isfinite(indv.fitness), pop))

    if len(elite_map.map.values()) < 1:
        return
    dataset_params = dataset_params or []
    best = elite_map.map.list_values()[0]
    if len(elite_map.map.values()) > 1:
        fn = min if minimize_fitness else max
        best = fn(elite_map.map.list_values(), key=lambda indv: indv.fitness)

    for coords, indv in elite_map.map.items():
        ind = dataset.index
        if "indv_id" in ind.columns and indv.id in ind["indv_id"].values:
            copy_row = (
                ind[ind["indv_id"] == indv.id].iloc[:1].copy()
            )  # copy the row, use :1 range to keep as dataframe
            copy_row["gen"] = gen
            copy_row["fitness"] = indv.fitness
            copy_row["best"] = int(indv == best)
            # dataset.index = ind.append(copy_row, ignore_index=True)
            dataset.index = pd.concat(
                [dataset.index, copy_row], ignore_index=True)
        else:
            ds = Dataset.read(os.path.join(outdir, f"gen{indv.gen}"))
            ds = ds.filter(indv_id=indv.id)
            ind = ds.index
            ind = ind.assign(
                gen=gen,
                fitness=indv.fitness,
                coords=str(coords),
                best=int(indv == best),
                born=gen
            )
            for param in dataset_params:
                ind[param] = [getattr(indv, param)] * len(ds.index)  # multiply for when group-by causes copied rows

            # patch outdir
            ind["outdir"] = ind["outdir"].apply(
                lambda o: os.path.join(f"gen{indv.gen}", o)
            )
            to_drop = [
                col
                for col in ["magnet_coords", "magnet_angles", "labels"]
                if col in ind
            ]
            ind.drop(columns=to_drop, inplace=True)  # debug

            # novelty measures should be added last due to variable column number
            column_names = indv.novelty_labels if hasattr(indv, "novelty_labels") else [
                f"novelty_measures{i}" for i in range(len(indv.novelty))]
            novelty_data = {name: val for name, val in zip(column_names, indv.novelty)}
            ind = ind.assign(**novelty_data)  # Efficient batch assignment

            dataset.index = pd.concat([dataset.index, ind], ignore_index=True)
        if not dataset.params:
            dataset.params = ds.params


def save_snapshot(outdir, pop):
    with open(os.path.join(outdir, "snapshot.pkl"), "wb") as f:
        pkl.dump([repr(indv) for indv in pop], f)


def setup_continue_run(outdir, individual_class, start_gen):
    with open(os.path.join(outdir, "snapshot.pkl"), "rb") as f:
        pop = list(map(lambda i: individual_class.from_string(i), pkl.load(f)))

    dataset = Dataset.read(outdir)

    assert os.path.isdir(
        os.path.join(outdir, f"gen{start_gen-1}")
    ), f"gen{start_gen-1} does not exist"

    if os.path.isdir(os.path.join(outdir, f"gen{start_gen}")):
        # rename last gen so not overwritten
        gen_string = f"gen{start_gen}"
        os.rename(
            os.path.join(outdir, gen_string), os.path.join(
                outdir, "old_" + gen_string)
        )
        gen_string = "old_" + gen_string
    else:
        gen_string = f"gen{start_gen-1}"

    # find the largest id in the dataset so we dont overwrite
    newest_index = read_csv(os.path.join(outdir, gen_string, "index.csv"))
    super_index = read_csv(os.path.join(outdir, "index.csv"))
    max_id = np.concatenate(
        (newest_index["indv_id"].values, super_index["indv_id"].values)
    ).max()
    individual_class.set_id_start(max_id + 1)

    return pop, dataset


def main_check_args(individual_params, evolved_params, sweep_params, kwargs):
    check_args = np.unique(
        list(evolved_params) + list(kwargs) + list(sweep_params), return_counts=True
    )
    check_args = [
        check_args[0][i] for i in range(len(check_args[0])) if check_args[1][i] > 1
    ]
    if check_args:
        raise RuntimeError(
            f"param '{check_args[0]}' appears in multiple param groups")


def setup_evolved_params(evolved_params, individual_class):
    for evo_param in evolved_params:
        evolved_params[evo_param] = {
            "low": evolved_params[evo_param][0],
            "high": evolved_params[evo_param][1],
            "shape": (
                evolved_params[evo_param][2:]
                if len(evolved_params[evo_param]) > 2
                else None
            ),
        }
    individual_class.set_evolved_params(evolved_params)


def crossover(pop, n_kids, kids_per_pair=1):
    assert kids_per_pair == 1, "only 1 kid per pair supported"

    kids_list = []
    parents = choose(pop, size=int(n_kids / kids_per_pair * 2))
    for i in range(0, n_kids, 2):
        indv = parents[i]
        partner = parents[i + 1]
        cross_result = indv.crossover(partner)
        kids_list.append(cross_result[0])

    return kids_list


def print_time(tr):
    print(
        f"~{np.round(tr / 3600, 2)} hours remaining"
        if tr > 3600
        else (
            f"~{np.round(tr / 60, 2)} minutes remaining"
            if tr > 60
            else f"~{np.round(tr, 2)} seconds remaining"
        )
    )


def choose(lst, size):
    if size == 0 or len(lst) == 0:
        return []

    indices = np.arange(len(lst))
    chosen = []

    if len(indices) < size:
        repeats = size // len(indices)
        chosen += list(indices) * repeats  # Add repeated full list cycles

    # Add remaining random choices
    chosen += np.random.choice(a=indices, size=size -
                               len(chosen), replace=False).tolist()

    return [lst[i] for i in chosen]


def main(
    outdir,
    individual_class,
    evaluate_inner,
    minimize_fitness=True,
    *,
    pop_size=100,
    generation_num=100,
    mut_prob=0.5,
    cx_prob=0.25,
    mut_strength=1,
    individual_params={},
    evolved_params={},
    sweep_params=OrderedDict(),
    dependent_params={},
    group_by=None,
    continue_run=False,
    starting_gen=1,
    dataset_params=None,
    map_shape=(2, 2),
    map_bounds=None,
    **kwargs,
):

    print("Initialising")
    main_check_args(individual_params, evolved_params, sweep_params, kwargs)

    assert (
        os.path.isdir(outdir) or not continue_run
    ), "can't continue run without existing outdir"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    setup_evolved_params(evolved_params, individual_class)

    eliteMap = EliteMap(shape=map_shape, bounds=map_bounds,
                        minimize_fitness=minimize_fitness)

    if continue_run:
        init_pop, dataset = setup_continue_run(
            outdir, individual_class, starting_gen)
        eliteMap.add(init_pop)

    else:
        init_pop = [individual_class(**individual_params)
                    for _ in range(pop_size)]

        evaluate_inner(
            init_pop,
            0,
            outdir,
            sweep_params=sweep_params,
            group_by=group_by,
            dependent_params=dependent_params,
            **kwargs,
        )
        eliteMap.add(init_pop)

        # create superdataset
        index = pd.DataFrame()
        info = {
            "command": " ".join(map(shlex.quote, sys.argv)),
        }
        dataset = Dataset(index, None, info, basepath=outdir)

        update_superdataset(dataset, outdir, eliteMap, 0, dataset_params)
        dataset.save()

    gen_times = []
    best = None
    for gen in range(starting_gen, generation_num + 1):
        print(f"starting gen {gen} of {generation_num}")
        if len(gen_times) > 0:
            tr = np.mean(gen_times[-10:]) * (generation_num - gen)
            print_time(tr)
        time = datetime.now()

        # Mutate!
        print("    Mutate")
        mut_kids = []

        mut_parents = choose(eliteMap.map.list_values(),
                             size=int(pop_size * mut_prob))
        for parent in mut_parents:
            mut_kids += parent.mutate(mut_strength)

        # Crossover!
        print("    Crossover")
        crossover_kids = crossover(
            eliteMap.map.list_values(), int(pop_size * cx_prob))

        kids = mut_kids + crossover_kids
        for indv in kids:
            indv.gen = gen

        # Eval
        print("    Evaluate")

        evaluate_inner(
            kids,
            gen,
            outdir,
            sweep_params=sweep_params,
            group_by=group_by,
            dependent_params=dependent_params,
            **kwargs,
        )

        for kid in kids:  # if using one of old fit fun
            if hasattr(kid, "fitness"):
                continue
            kid.fitness = kid.fitness_components[0]
            kid.novelty = kid.fitness_components[1:]

        eliteMap.add(kids)

        update_superdataset(
            dataset, outdir, eliteMap, gen, minimize_fitness, dataset_params
        )
        dataset.save()

        better = min if minimize_fitness else max
        best = better(
            filter(lambda indv: not np.isnan(indv.fitness), eliteMap.map.list_values()),
            key=lambda indv: indv.fitness,
            default=None  # Handle case where all values are NaN
        )

        print(f"best fitness: {best.fitness if best is not None else 'NaN'}")
        print(f"  with novelty measures: {best.novelty}\n")
        print(f"{len(dataset.index[dataset.index['born'] == gen])} new individuals added to map")

        save_snapshot(outdir, eliteMap.map.list_values())

        gen_times.append((datetime.now() - time).total_seconds())
    return best
