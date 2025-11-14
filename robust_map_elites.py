from itertools import islice
import numpy as np
import pandas as pd
import scipy
from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict, defaultdict, deque, namedtuple, Counter
import shlex
import os
import sys
from warnings import warn
from flatspin.data import Dataset, read_csv
import pickle as pkl

from map_elites import (main_check_args, setup_evolved_params, setup_continue_run, print_time,
                        choose, crossover, save_snapshot, EliteMap)

def clip(x, minn, maxx):
    return max(min(x, maxx), minn)

class RobustEliteMap(EliteMap):
    def __init__(self, init_archive_size=3, max_archive_size=10, **kwargs):
        super().__init__(**kwargs)

        self._init_archive_size = init_archive_size
        self._max_archive_size = max_archive_size
        self._fitness_archive = FitnessArchive(max_size=max_archive_size)
        self._rejects_queue = deque(maxlen=self._target_capacity)
        assert not np.any(self._auto_bounds), "RobustEliteMap no support for autobounds"

    def population(self, fill=False):
        pop = self.map.list_values()
        if fill and len(pop) < self._target_capacity:
            needed = self._target_capacity - len(pop)
            return pop + list(islice(self._rejects_queue, needed))

        return pop

    def clean_up(self):
        """remove indivudals for the fitness archive that are no longer needed"""
        valid_ids = set(indv.id for indv in self.population() + list(self._rejects_queue))
        archive_ids = set(self._fitness_archive._archive.keys())

        ids_to_remove = archive_ids - valid_ids

        for id in ids_to_remove:
            del self._fitness_archive._archive[id]

    def expand_bounds(self, indvs):
        raise NotImplementedError("no bound expansion possible in RoubustEliteMap")

    def split_elites(self, indvs):
        """split a list of indvs to elites and not elites"""
        elite_ids = self.map.id_set()
        elites = [i for i in indvs if i.id in elite_ids]
        others = [i for i in indvs if i.id not in elite_ids]

        return elites, others

    def update(self, indvs):
        self._fitness_archive.update_many(indvs, self)
        updated_elites, challengers = self.split_elites(indvs)

        # if any elites nolonger on right place, remove them (hold onto them to let them challenge others)
        for coord, elite in list(self.map.items()):
            if self._fitness_archive[elite.id].mode_coord != coord:
                challengers.append(elite)
                self.map.pop(coord)

        for challenger in challengers:
            self.try_add(challenger)

        self.clean_up()


    def try_add(self, challenger):
        """see if challenger can be added to map, and if so => add it"""
        coords = self._fitness_archive[challenger.id].mode_coord
        elite = self.map[coords]

        if elite == None or self.beat_elite(challenger, elite):
            self.map[coords] = challenger
            if elite != None:
                self._rejects_queue.append(elite)
        else:
            self._rejects_queue.append(challenger)

    def beat_elite(self, challenger, elite):
        assert elite != challenger, "self challenge"
        elite_record = self._fitness_archive[elite.id]
        ch_record = self._fitness_archive[challenger.id]

        # if elite mean fitness is nan, challenger auto-win
        if np.isnan(elite_record.mean_fitness):
            return True

        # if challenger mean fitness is nan or challenger has less evals than elite, elite wins
        if np.isnan(ch_record.mean_fitness) or len(ch_record.fitness) < len(elite_record.fitness):
            return False

        # if unequal mode_count, the higher mode_count wins
        if ch_record.mode_count != elite_record.mode_count:
            return ch_record.mode_count > elite_record.mode_count

        return ch_record.mean_fitness > elite_record.mean_fitness

    def needed_runs(self, indvs):
        """Work out how many more times each individual should be evaluated.

        Challengers run until they match the eval count the elite will have
        after its next evaluation (len(elite.fitness) + 1).
        Each elite that is challenged runs once more."""
        repeat_dict = {}
        elite_ids = set()
        to_be_evaluated = []

        self._fitness_archive.update_many(indvs, self) # required because this is called after an evaluate inner but before an elite_map.update()
        for indv in indvs:
            coords = self._fitness_archive[indv.id].mode_coord
            elite = self.map[coords]
            if elite is None: # no elite to challenge, so no further runs needed
                continue

            if elite.id not in elite_ids:
                elite_ids.add(elite.id)
                to_be_evaluated.append(elite)
            my_evals = len(self._fitness_archive[indv.id].fitness)
            diff = min(len(self._fitness_archive[elite.id].fitness) + 1, self._max_archive_size) - my_evals
            diff = max(diff, 0)
            repeat_dict[indv.id] = clip(diff, 0, self._max_archive_size - my_evals)
            if diff > 0:
                to_be_evaluated.append(indv)

        for el_id in elite_ids:
            repeat_dict[el_id] = 1

        return to_be_evaluated, repeat_dict

    def get_best(self, indvs=None, minimize_fitness=None, return_nan=False):
        if indvs is None:
            indvs = self.population()
        if minimize_fitness == None:
            minimize_fitness = self._minimize_fitness
        better = min if minimize_fitness else max
        indv2fit = lambda indv: self._fitness_archive[indv.id].mean_fitness
        best = better(
            filter(lambda indv: not np.isnan(indv2fit(indv)), indvs),
            key=lambda indv: indv2fit(indv),
            default=None,  # Handle case where all values are NaN
        )
        if return_nan:
            return best or indvs[0] # all nan so return first
        return best


def tuple_mode(l, return_indices=False):
    """finds most common tuple and return the tuple and its count"""
    mode, count =  Counter(l).most_common(1)[0]

    if not return_indices:
        return mode, count

    indices = [i for i, t in enumerate(l) if t == mode]

    return mode, count, indices


@dataclass
class Record:
    fitness: deque
    coords: deque
    mean_fitness: float = None
    mode_coord: float = None
    mode_count: int = None


def make_record(max_size: int) -> Record:
    """Factory for a new Record with independent deques."""
    return Record(
        fitness=deque(maxlen=max_size),
        coords=deque(maxlen=max_size),
        mean_fitness=None,
        mode_coord=None,
        mode_count=None
    )

class FitnessArchive():
    def __init__(self, max_size):
        self._max_size = max_size
        self._archive : dict[int, Record] = {}

    def __getitem__(self, id):
        """Gets a value from the map at the given coordinate if it's valid."""
        return self._archive.get(id)

    def update_many(self, pop, elitemap):
        for indv in pop:
            self.update(indv, elitemap)

    def update(self, indv, elitemap):
        """push fitness & novelty to archive and recompute measures"""
        fits = indv.fitness
        novelty_list = indv.novelty

        if not is_list(fits):
            fits = [fits]
        if not is_list(novelty_list):
            novelty_list = [novelty_list]

        id = indv.id
        if id not in self._archive:
            self._archive[id] = make_record(self._max_size)
        coords_list = [elitemap.coords_in_map(novelty) for novelty in novelty_list]
        record = self._archive[id]
        record.fitness.extend(fits)
        record.coords.extend(coords_list)

        record.mode_coord, record.mode_count, indices = tuple_mode(record.coords, return_indices=True)
        record.mean_fitness = np.mean([record.fitness[i] for i in indices])
        # do something with novelty labels so they can be seen in the update_superdataset, but dont stack up in the indv
        indv.novelty_labels = indv.novelty_labels[0] if is_list(indv.novelty_labels) else indv.novelty_labels
        indv.refresh()
        del indv.novelty


def is_list(o):
    return isinstance(o, list)


def update_superdataset(
    dataset, outdir, elite_map, gen, minimize_fitness=True, dataset_params=None
):
    # pop = list(filter(lambda indv: np.isfinite(indv.fitness), pop))
    pop = elite_map.population()
    if len(pop) < 1:
        return
    dataset_params = dataset_params or []

    subgen = ""
    if type(gen) is str:
        gen, *subgen = gen.split("_", 1)
        gen = int(gen)
        subgen = subgen[0] if subgen else ""

    for coords, indv in elite_map.map.items():
        ind = dataset.index
        if "indv_id" in ind.columns and indv.id in ind["indv_id"].values:
            copy_row = (
                ind[ind["indv_id"] == indv.id].iloc[:1].copy()
            )  # copy the row, use :1 range to keep as dataframe
            copy_row["gen"] = gen
            copy_row["fitness"] = elite_map._fitness_archive[indv.id].mean_fitness
            copy_row["best"] = 1 # TODO: more meaningfull best
            # dataset.index = ind.append(copy_row, ignore_index=True)
            dataset.index = pd.concat([dataset.index, copy_row], ignore_index=True)
        else:
            gen_path = f"gen{indv.gen}_{subgen}" if subgen else f"gen{indv.gen}"
            ds = Dataset.read(os.path.join(outdir, gen_path))
            ds = ds.filter(indv_id=indv.id)
            ind = ds.index
            ind = ind.assign(
                gen=gen,
                fitness=elite_map._fitness_archive[indv.id].mean_fitness,
                coords=str(coords),
                best=1, # TODO: more meaningfull best
                born=gen,
            )
            for param in dataset_params:
                ind[param] = [getattr(indv, param)] * len(
                    ds.index
                )  # multiply for when group-by causes copied rows

            # patch outdir
            ind["outdir"] = ind["outdir"].apply(
                lambda o: os.path.join(f"gen{indv.gen}", o)
            )
            to_drop = [
                col
                for col in ["magnet_coords", "magnet_angles", "labels", "pulses"]
                if col in ind
            ]
            ind.drop(columns=to_drop, inplace=True)  # debug

            # novelty measures should be added last due to variable column number
            novelty = elite_map._fitness_archive[indv.id].mode_coord
            lbls = list(indv.novelty_labels)
            column_names = (
                lbls if hasattr(indv, "novelty_labels")
                else [f"novelty_measures{i}" for i in range(len(novelty))]
            )
            novelty_data = {name: val for name, val in zip(column_names, novelty)}
            ind = ind.assign(**novelty_data)  # Efficient batch assignment
            dataset.index = pd.concat([dataset.index, ind], ignore_index=True)
        if not dataset.params:
            dataset.params = ds.params


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
    map_target_capacity=None,
    random_seed=0,
    strict_compare=True,
    init_archive_size=3,
    max_archive_size=10,
    **kwargs,
):

    print("Initialising")
    main_check_args(individual_params, evolved_params, sweep_params, kwargs)

    assert (
        os.path.isdir(outdir) or not continue_run
    ), "can't continue run without existing outdir"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    np.random.seed(random_seed)

    setup_evolved_params(evolved_params, individual_class)

    elite_map = RobustEliteMap(
        shape=map_shape, bounds=map_bounds, minimize_fitness=minimize_fitness,
        target_capacity=map_target_capacity, strict=strict_compare, max_archive_size=max_archive_size
    )


    if continue_run:
        raise NotImplementedError()
        # probably need to store the fitness archive in the snapshot
        init_pop, dataset = setup_continue_run(outdir, individual_class, starting_gen)
        elite_map.add(init_pop)
        fitness_archive.update_many(init_pop, elite_map)

    else:
        init_pop = [individual_class(**individual_params) for _ in range(pop_size)]


        repeat_dict = {indv.id:init_archive_size for indv in init_pop}
        evaluate_inner(
            init_pop,
            0,
            outdir,
            sweep_params=sweep_params,
            group_by=group_by,
            dependent_params=dependent_params,
            repeat_dict=repeat_dict,
            **kwargs,
        )

        elite_map.update(init_pop)

        # create superdataset
        index = pd.DataFrame()
        info = {
            "command": " ".join(map(shlex.quote, sys.argv)),
        }
        dataset = Dataset(index, None, info, basepath=outdir)

        update_superdataset(dataset, outdir, elite_map, 0, minimize_fitness, dataset_params)
        elite_map.log(outdir, 0)
        dataset.save()

    gen_times = []
    best = None
    for gen in range(starting_gen, generation_num + 1):
        print(f"starting gen {gen} of {generation_num}")
        individual_class.current_gen = gen
        if len(gen_times) > 0:
            tr = np.mean(gen_times[-10:]) * (generation_num - gen)
            print_time(tr)
        time = datetime.now()

        # Mutate!
        print("    Mutate")
        mut_kids = []

        parent_pool = elite_map.population(fill=True)
        # replace nan parents with new random
        parent_pool = [
            (
                parent
                if not np.isnan(elite_map._fitness_archive[parent.id].mean_fitness)
                else individual_class(gen=gen, **individual_params)
            )
            for parent in parent_pool
        ]

        if len(parent_pool) < pop_size:
            parent_pool += [individual_class(gen=gen, **individual_params) for _ in range(pop_size - len(parent_pool))]

        mut_parents = choose(parent_pool, size=int(pop_size * mut_prob))
        for parent in mut_parents:
            mut_kids += parent.mutate(mut_strength)

        # Crossover!
        print("    Crossover")
        crossover_kids = crossover(parent_pool, int(pop_size * cx_prob))

        kids = mut_kids + crossover_kids
        for indv in kids:
            indv.gen = gen

        # Eval
        print("    Evaluate p1")
        repeat_dict = {indv.id:init_archive_size for indv in kids}
        evaluate_inner(
            kids,
            gen,
            outdir,
            sweep_params=sweep_params,
            group_by=group_by,
            dependent_params=dependent_params,
            repeat_dict=repeat_dict,
            **kwargs,
        )

        # check how many reruns for kids, and elites
        print("    Evaluate p2")
        evaluees, repeat_dict = elite_map.needed_runs(kids)
        evaluate_inner(
            evaluees,
            f"{gen}_b",
            outdir,
            sweep_params=sweep_params,
            group_by=group_by,
            dependent_params=dependent_params,
            repeat_dict=repeat_dict,
            **kwargs,
        )

        elite_map.update(evaluees)

        update_superdataset(
            dataset, outdir, elite_map, gen, minimize_fitness, dataset_params
        )
        dataset.save()
        elite_map.log(outdir, gen)

        best = elite_map.get_best()

        if best is not None:
            print(f"best fitness: {elite_map._fitness_archive[best.id].mean_fitness}")
            print(f"  with novelty coords: {elite_map._fitness_archive[best.id].mode_coord}\n")

        print(
            f"{len(dataset.index[dataset.index['born'] == gen])} new individuals added to map"
        )

        save_snapshot(outdir, elite_map.population())

        gen_times.append((datetime.now() - time).total_seconds())
    return best
