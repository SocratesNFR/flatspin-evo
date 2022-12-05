# vim: tw=120
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import logging
from matplotlib.animation import FuncAnimation, writers
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
import warnings

from flatspin import plotting
from shapely.geometry import box, MultiPolygon
from shapely.affinity import rotate, translate
from shapely.prepared import prep
from itertools import count, permutations
from functools import cached_property
from copy import deepcopy, copy
from collections import Sequence, OrderedDict
from time import sleep

import evo_alg as ea
from base_individual import Base_Individual
import fitness_functions

from PIL import Image

from flatspin.data import Dataset, read_table, load_output, is_archive_format, match_column, save_table
from flatspin.grid import Grid
from flatspin.utils import get_default_params, import_class
from flatspin.runner import run, run_dist, run_local
from flatspin.sweep import sweep
import os
import pandas as pd
import shlex
import sys


class ProgressBar(tqdm):
    pass


class ParallelProgress(Parallel):
    def __init__(self, progress_bar, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_bar = progress_bar

    def print_progress(self):
        inc = self.n_completed_tasks - self._progress_bar.n
        self._progress_bar.update(inc)


class Individual(Base_Individual):
    _id_counter = count(0)

    def __init__(self, *, max_tiles=1, tile_size=600, mag_w=220, mag_h=80, max_symbol=1, pheno_size=40,
                 pheno_bounds=None, age=0, id=None, gen=0, fitness=None, fitness_components=None, fitness_info=None,
                 tiles=None, init_pheno=True, fixed_geom=False, parent_ids=None, **kwargs):

        self.id = id if id is not None else next(Individual._id_counter)
        self.gen = gen  # generation of birth
        self.max_tiles = max_tiles
        self.tile_size = tile_size
        self.mag_w = mag_w
        self.mag_h = mag_h
        self.age = age
        self.max_symbol = max_symbol
        self.pheno_size = pheno_size
        self.pheno_bounds = pheno_bounds
        self.fixed_geom = fixed_geom
        if self.pheno_bounds and np.isscalar(pheno_bounds):
            self.pheno_bounds = [pheno_bounds] * 2
        self.fitness = fitness
        self.fitness_components = fitness_components
        self.fitness_info = fitness_info

        if parent_ids is None:
            self.parent_ids = []
        else:
            self.parent_ids = parent_ids

        self.init_evolved_params(**kwargs)
        if not self.fixed_geom:
            if mag_h > mag_w:
                raise Warning(
                    "conversion to flatspin assumes magnet height < magnet width!"
                )
            if tiles is not None and 1 <= len(tiles):
                if len(tiles) > self.max_tiles:
                    raise ValueError(
                        "Individual has more tiles than the value of 'max_tiles'"
                    )
                self.tiles = tiles
            else:
                self.tiles = [
                    Tile(
                        mag_w=mag_w,
                        mag_h=mag_h,
                        tile_size=tile_size,
                        max_symbol=max_symbol,
                    )
                    for _ in range(np.random.randint(1, max_tiles + 1))
                ]
            if init_pheno:
                self.pheno = self.geno2pheno(geom_size=self.pheno_size)

    def refresh(self):
        if not self.fixed_geom:
            self.pheno = self.geno2pheno(geom_size=self.pheno_size)
        self.clear_fitness()

    def clear_fitness(self):
        self.fitness = None
        self.fitness_components = None
        self.fitness_info = None

    def __repr__(self):
        # defines which attributes are ignored by repr
        ignore_attributes = ("pheno", "anime")
        return repr({k: v for (k, v) in vars(self).items() if k not in ignore_attributes})

    @property
    def coords(self):
        return np.array([mag.pos for mag in self.pheno]) if self.pheno else None

    @property
    def angles(self):
        return np.array([mag.angle for mag in self.pheno]) if self.pheno else None

    def copy(self, refresh=True, **overide_kwargs):
        # defines which attributes are used when copying
        # somre params are ignored as need to be deep copied
        # individual must be refreshed after cloning for to build correct phenotype
        # (this can be set to False if the refresh will be done manually later)
        ignore_attributes = (
            "gen",
            "evolved_params_values",
            "pheno",
            "id",
            "init_pheno",
        )
        params = {k: v for (k, v) in vars(self).items() if k not in ignore_attributes}
        params.update(overide_kwargs)
        new_indv = Individual(init_pheno=False, **params)
        new_indv.evolved_params_values = deepcopy(self.evolved_params_values)
        # copy attributes that are referenced to unlink
        if not new_indv.fixed_geom:
            new_indv.tiles = [
                Tile(magnets=[mag.copy() for mag in tile]) for tile in new_indv.tiles
            ]
        if refresh:
            new_indv.refresh()
        return new_indv

    @staticmethod
    def from_string(s, keep_pheno=False, **overides):
        array = np.array
        inf = np.inf
        params = eval(s)
        params.update(overides)

        if not params.get("fixed_geom", False):
            # Instanciate Magnets from result of repr
            params["tiles"] = [Tile(magnets=[Magnet(**mag) for mag in tile]) for tile in params["tiles"]]
        indv = Individual(init_pheno=not keep_pheno, **params)
        if keep_pheno:
            indv.pheno = [Magnet(**mag) for mag in params["pheno"]]
        return indv

    def geno2pheno(self, geom_size=40, animate=False, no_change_terminator=1, **animation_kwargs):
        frontier, frozen = [], []
        iter_count = 0
        frames = []
        since_change = 0
        max_len = 0
        # add origin magnet in the first tile to the frontier
        frontier.append(self.tiles[0][0].copy(created=iter_count))
        frontier[0].symbol.fill(0)
        if "initial_rotation" in self.evolved_params_values:
            frontier[0].i_rotate(
                self.evolved_params_values["initial_rotation"], "centroid"
            )
        if animate:
            frames.append(list(map(lambda m: m.as_patch(), frontier + frozen)))

        while (len(frontier) + len(frozen) < geom_size and len(frontier) > 0 and since_change < no_change_terminator):
            if animate:
                frames.append(list(map(lambda m: m.as_patch(), frontier + frozen)))
            iter_count += 1
            new_front = []
            for magnet in frontier:
                for tile in self.tiles:
                    new_mags = tile.apply_to_mag(magnet, iter_count)

                    for new_mag in new_mags:
                        # check doesnt intersect any magnets and if there are bounds is within them
                        if not new_mag.is_intersecting(frontier + frozen + new_front) \
                            and not (self.pheno_bounds
                                     and not (-self.pheno_bounds[0] < new_mag.pos[0] < self.pheno_bounds[0]
                                              and -self.pheno_bounds[1] < new_mag.pos[1] < self.pheno_bounds[1])):
                            new_front.append(new_mag)

            frozen.extend(frontier)
            frontier = new_front
            if max_len < len(frontier) + len(frozen):
                since_change = 0
                max_len = len(frontier) + len(frozen)
            else:
                since_change += 1

        if animate:
            self.anime = Individual.frames2animation(frames, **animation_kwargs)
            # plt.show()
            # self.anime.save("anime.mp4")

        return (centre_magnets(frozen + frontier)[:geom_size]if np.isfinite(geom_size) else
                centre_magnets(frozen + frontier))

    @staticmethod
    def frames2animation(frames, interval=400, title=False, ax_color="k", color_method="rainbow", figsize=(8, 6), axis_off=False, constant_zoom=True, repeat_final=0):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        bounds = np.vstack(
            list(map(lambda m: m.xy, [mag for frame in frames for mag in frame]))
        )
        xlim = bounds[:, 0].min(), bounds[:, 0].max()
        ylim = bounds[:, 1].min(), bounds[:, 1].max()
        if title:
            title = list(title)
        if repeat_final:
            frames = frames + frames[-1:] * repeat_final

        def step(i, maxi, xlim, ylim, title):
            ax.cla()
            for poly in frames[i]:
                ax.add_patch(poly)
            if title:
                ax.set_title(title[i] if len(title) > i else title[-1])
            if constant_zoom:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            if axis_off:
                plt.axis('off')
                ax.add_artist(ax.patch)
                ax.patch.set_zorder(-1)
                fig.subplots_adjust(left=None, bottom=None, right=None, wspace=None, hspace=None)
            if not constant_zoom:
                ax.autoscale()
            fig.canvas.draw()

        if color_method == "rainbow":
            colours = ea.rainbow_colours(len(frames))
            for frame in frames:
                for poly in frame:
                    poly.set_color(colours[poly.iterCreated % len(colours)])
        elif color_method == "current":
            for i, frame in enumerate(frames):
                current_iter = max(frame, key=lambda m: m.iterCreated).iterCreated
                for poly in frame:
                    if poly.iterCreated == current_iter:
                        poly.set_color("w")
                    else:
                        poly.set_color("r")
        ax.set_facecolor(ax_color)
        return FuncAnimation(fig, step, frames=len(frames), fargs=(len(frames) - 1, xlim, ylim, title),
                             blit=False, interval=interval)

    @staticmethod
    def print_mags(mags, facecolor=None, edgecolor=None, **kwargs):
        # plt.figure()
        for mag in mags:
            patch = mag.as_patch()
            if facecolor:
                patch.set_facecolor(facecolor)
            if edgecolor:
                patch.set_edgecolor(edgecolor)
            plt.gca().add_patch(patch)
            plt.gca().set_aspect(1)
            plt.autoscale()

    def mutate(self, strength=1):
        """mutate an Individual to produce children, return any children  as a list"""
        clone = self.copy(refresh=False, parent_ids=[self.id])
        mut_types = []  # valid mutations for individual

        if not clone.fixed_geom:
            mut_types += [
                Individual.mutate_magnet_pos,
                Individual.mutate_magnet_angle,
                Individual.mutate_symbol,
            ]
            if clone.max_tiles > 1:  # cannot add/remove tiles when max tile is 1
                if (len(clone.tiles) < clone.max_tiles):  # only add new tiles when less than max
                    mut_types += [
                        Individual.mutate_clone_tile,
                        Individual.mutate_add_rand_tile,
                    ]
                if len(clone.tiles) > 1:  # only remove tile when more than 1
                    mut_types += [Individual.mutate_delete_tile]

        weights = [1] * len(mut_types)  # uniform weights
        if len(self.evolved_params_values) > 0:
            mut_types += [Individual.mutate_evo_param]
            # increase chance of selecting param-mutation by the num of evo params so they are picked evenly
            weights += [len(self.evolved_params_values)]

        assert len(mut_types) > 0, "no mutations valid for this individual"
        weights = np.array(weights) / np.sum(weights)  # normalise weights
        mutation = np.random.choice(mut_types, p=weights)
        mut_info = mutation(clone, strength)

        if clone:
            clone.age = 0
            clone.refresh()
            logging.info(f"ind {clone.id} created from {mutation.__name__} on ind {self.id} info: {mut_info}")
        else:
            logging.info(f"Failed mutation: {mutation.__name__} on ind {self.id} info: {mut_info}")
        return [clone]

    def random_tiles(self, num=1, replace=False):
        return [self.tiles[i] for i in list(np.random.choice(range(len(self.tiles)), size=num, replace=replace))]

    def crossover(self, other):
        """crossover 2 individuls, return any new children as a list"""
        assert not (self.fixed_geom and len(self.evolved_params_values) < 1), "no free values to crossover"
        parents = [self, other]
        np.random.shuffle(parents)
        if not self.fixed_geom:
            # if both parents have 1 tile, cross over the angle and positions of the magnets
            if len(self.tiles) == len(other.tiles) == 1:
                cx = Individual.crossover_single_tile
            else:
                cx = Individual.crossover_tiles
            child = cx(parents)
            cx_name = cx.__name__
        else:
            child = parents[0].copy()
            cx_name = "fixed_copy"

        evo_params_info = ""
        if child:
            if len(self.evolved_params_values) > 0:
                child.evolved_params_values = Individual.crossover_evo_params(parents)
                evo_params_info = f"and {Individual.crossover_evo_params.__name__} "
        if child:
            child.age = 0
            child.refresh()
            logging.info(
                f"ind {child.id} created from {cx_name} {evo_params_info}with parents {[parents[0].id, parents[1].id]}"
            )
        else:
            logging.info(
                f"Failed crossover: {cx_name} {evo_params_info}with parents {[parents[0].id, parents[1].id]}"
            )
        child.parent_ids = [self.id, other.id]
        return [child] if child else []

    @staticmethod
    def crossover_single_tile(parents):
        """is affected by order of parents (shuffle before calling)"""
        child = parents[0].copy(refresh=False)
        for i in range(1, len(child.tiles[0])):  # probably just [1]
            angles = (parents[0].tiles[0][i].angle, parents[1].tiles[0][i].angle)
            poss = (parents[0].tiles[0][i].pos, parents[1].tiles[0][i].pos)

            new_angle = np.random.rand() * np.abs(angles[0] - angles[1]) + min(angles)
            new_pos = np.random.rand() * np.abs(poss[0] - poss[1]) + np.min(
                poss, axis=0
            )

            rot = new_angle - child.tiles[0][i].angle
            child.tiles[0][i].i_rotate(rot, "centroid")

            translate = new_pos - child.tiles[0][i].pos
            child.tiles[0][i].i_translate(*translate)

        if Magnet.any_intersecting(child.tiles[0]):  # crossover failed
            child = None
        return child

    @staticmethod
    def crossover_tiles(parents):
        num_tiles = (len(parents[0].tiles), len(parents[1].tiles))
        num_tiles = min(
            parents[0].max_tiles, np.random.randint(min(num_tiles), max(num_tiles) + 1)
        )

        from_first_parent = np.random.randint(0, num_tiles + 1)
        tiles = parents[0].random_tiles(
            num=min(from_first_parent, len(parents[0].tiles))
        )
        tiles += parents[1].random_tiles(
            num=min(num_tiles - from_first_parent, len(parents[1].tiles))
        )

        tiles = [tile.copy() for tile in tiles]
        child = parents[0].copy(refresh=False)
        child.tiles = tiles
        return child

    @staticmethod
    def mutate_magnet_pos(clone, strength):
        # pick a tile at random, pick a magnet excluding the first magnet
        # and move it (only position of magnet center is confined by tile_size)
        tile = clone.tiles[np.random.randint(0, len(clone.tiles))]
        x = 1 + np.random.randint(len(tile[1:]))
        copy_mag = tile[x].copy()
        distance = (
            Individual.gauss_mutate(
                copy_mag.pos, strength * clone.tile_size / 200, 0, clone.tile_size
            )
            - copy_mag.pos
        )
        old_pos = copy_mag.pos.tolist()
        copy_mag.i_translate(*distance)

        if not copy_mag.is_intersecting(tile[:x] + tile[x + 1:]):
            # only mutate if does not cause overlap
            tile.locked = False
            tile[x] = copy_mag
            tile.locked = True
        else:
            # mutation failed, terminate clone!
            clone = None
        return f"moved mag {old_pos} -> {copy_mag.pos.tolist()}"

    @staticmethod
    def mutate_magnet_angle(clone, strength):
        # pick a tile at random, pick a magnet excluding the first magnet and rotate it (about centroid)
        tile = clone.tiles[np.random.randint(0, len(clone.tiles))]
        x = 1 + np.random.randint(len(tile[1:]))
        copy_mag = tile[x].copy()
        old_angle = copy_mag.angle
        rotation = np.random.normal(0, strength * (2 * np.pi) / 200)
        copy_mag.i_rotate(rotation, "centroid")

        if not copy_mag.is_intersecting(tile[:x] + tile[x + 1:]):
            # only mutate if does not cause overlap
            tile.locked = False
            tile[x] = copy_mag
            tile.locked = True
        else:
            # mutation failed, terminate clone!
            clone = None
        return f"rotated mag {old_angle} -> {copy_mag.angle}"

    @staticmethod
    def mutate_symbol(clone, strength):
        tile = clone.random_tiles()[0]
        magnet = np.random.choice(tile)
        old_sym = magnet.symbol.copy()
        magnet.symbol = np.random.randint(clone.max_symbol, size=2)
        if all(old_sym == magnet.symbol):
            # mutation failed, terminate clone!
            clone = None

        return f"symbol changed {old_sym.tolist()} -> {magnet.symbol.tolist()}"

    @staticmethod
    def mutate_clone_tile(clone, strength):
        clone.tiles.append(clone.random_tiles()[0].copy())
        return f"cloned 1 tile"

    @staticmethod
    def mutate_delete_tile(clone, strength):
        clone.tiles.remove(clone.random_tiles()[0])
        return f"deleted 1 tile"

    @staticmethod
    def mutate_add_rand_tile(clone, strength):
        clone.tiles.append(
            Tile(
                mag_w=clone.mag_w,
                mag_h=clone.mag_h,
                tile_size=clone.tile_size,
                max_symbol=clone.max_symbol,
            )
        )
        return f"added 1 random tile"

    @staticmethod
    def known_spinice(name, min_dist=None, **kwargs):
        min_dist = (min_dist,) * 2 if min_dist is not None and np.isscalar(min_dist) else min_dist
        if name == "square":
            if min_dist is None:
                min_dist = (10, 10)
            ind = Individual(init_pheno=False, max_tiles=1, **kwargs)
            t = ind.tiles[0]
            t[1].i_set_rot(0)
            t[1].i_set_pos(t[0].pos[0] + t[1].mag_w / 2 + t[1].mag_h / 2 + t[1].padding + min_dist[0],
                           t[0].pos[1] + t[1].mag_w / 2 + t[1].mag_h / 2 + t[1].padding + min_dist[1])
        elif name == "ising":
            if min_dist is None:
                min_dist = (10, 10)
            ind = Individual(init_pheno=False, max_tiles=2, **kwargs)
            t1 = ind.tiles[0]
            t1[1].i_set_rot(t1[0].angle)
            t1[1].i_set_pos(t1[0].pos[0],
                            t1[0].pos[1] + t1[1].mag_w + t1[1].padding + min_dist[1])
            t2 = t1.copy()
            t2[1].i_set_pos(t2[0].pos[0] + t2[0].mag_h + t2[0].padding + min_dist[0],
                            t2[0].pos[1])
            ind.tiles = [t1, t2]

        elif name == "pinwheel":
            if min_dist is None:
                min_dist = (50, 50)
            ind = Individual(init_pheno=False, max_tiles=1, **kwargs)
            t = ind.tiles[0]
            t[1].i_set_rot(0)
            t[1].i_set_pos(t[0].pos[0],
                           t[0].pos[1] + t[1].mag_w / 2 + t[1].mag_h / 2 + t[1].padding + min_dist[1])
        elif name == "kagome":
            if min_dist is None:
                min_dist = (10, 10)
            # sorry
            ind = Individual(init_pheno=False, max_tiles=1, **kwargs)
            t = ind.tiles[0]
            t[1].i_set_rot(np.deg2rad(240))
            t[0].i_set_rot(0)
            hh, hw = (t[1].mag_h + t[1].padding) / 2, (t[1].mag_w + t[1].padding) / 2
            t[1].i_set_pos(t[0].pos[0] + hw + hw * np.sin(np.deg2rad(30)) + hh * np.sin(np.deg2rad(60)) + min_dist[0],
                           t[0].pos[1] + hh + hw * np.cos(np.deg2rad(30)) - hh * np.cos(np.deg2rad(60)) + min_dist[1])
            t.i_rotate(np.deg2rad(90), t[0].pos)
        else:
            raise Exception(f"name '{name}' not recognised")
        ind.pheno = ind.geno2pheno(ind.pheno_size)
        return ind

    @staticmethod
    def tessellate(magnets, shape=(5, 1), padding=0, centre=True, return_labels=False):
        magnets = [mag.copy() for mag in magnets]
        polygons = MultiPolygon([mag.get_polygon() for mag in magnets])
        minx, miny, maxx, maxy = polygons.bounds
        cell_size = np.array([maxx - minx, maxy - miny]) + padding
        first_row = [mag.copy() for mag in magnets]
        for col in range(1, shape[0]):
            new_col = [mag.copy() for mag in magnets]
            for mag in new_col:
                mag.i_translate(col * cell_size[0], 0)
            first_row.extend(new_col)
        result = []
        for row in range(1, shape[1]):
            new_row = [mag.copy() for mag in first_row]
            for mag in new_row:
                mag.i_translate(0, row * cell_size[1])
            result.extend(new_row)
        result.extend(first_row)
        if centre:
            centre_magnets(result)

        if return_labels:
            labels = [(x, y) for x in range(len(magnets)) for y in range(np.prod(shape))]
            return result, labels

        return result

    def plot(self, facecolor=None, edgecolor=None):
        for mag in self.pheno:
            patch = mag.as_patch()
            if facecolor:
                patch.set_facecolor(facecolor)
            if edgecolor:
                patch.set_edgecolor(edgecolor)
            plt.gca().add_patch(patch)
        plt.gca().set_aspect(1)
        plt.autoscale()

    @staticmethod
    def dataset_to_individuals(dataset, mag_w=220, mag_h=80, created=None, padding=20, num_of_magnets=None, **kwargs):
        individuals = []
        for row in dataset:
            indv = Individual(init_pheno=False, **kwargs, id=row.index["indv_id"].values[0])
            geom = read_table(row.tablefile("geom"))
            coords = geom.iloc[slice(0, num_of_magnets)][["posx", "posy"]].values
            angles = geom.iloc[slice(0, num_of_magnets)]["angle"].values
            indv.pheno = [Magnet(0, pos, angle, mag_w, mag_h, created, padding) for pos, angle in zip(coords, angles)]
            print(len(indv.pheno))
            individuals.append(indv)

        return individuals

    @staticmethod
    def pheno_from_string(s):
        array = np.array
        magnets = eval(s)
        return [Magnet(**mag) for mag in magnets]

    @staticmethod
    def set_id_start(start):
        Individual._id_counter = count(start)

    @staticmethod
    def get_default_shared_params(outdir="", gen=None, select_param=None):
        default_params = {
            "model": "CustomSpinIce",
            "encoder": "AngleSine",
            "H": 0.01,
            "phi": 90,
            "radians": True,
            "alpha": 37839,
            "sw_b": 0.4,
            "sw_c": 1,
            "sw_beta": 3,
            "sw_gamma": 3,
            "spp": 100,
            "hc": 0.03,
            "periods": 10,
            "neighbor_distance": 1000,
        }
        if select_param is not None:
            return default_params[select_param]
        if gen is not None:
            outdir = os.path.join(outdir, f"gen{gen}")
        default_params["basepath"] = outdir

        return default_params

    @staticmethod
    def get_default_run_params(pop, sweep_list=None, *, condition=None):
        if condition == "fixed_size":
            def condition(ind):
                return len(ind.coords) >= ind.pheno_size

        return Individual.super().get_default_run_params(pop, sweep_list, condition=condition)


class Tile(Sequence):
    def __init__(
        self, *, tile_size=600, mag_w=220, mag_h=80, max_symbol=1, magnets=None
    ):
        """
        make new random tile from scratch
        """
        # if magnets provided just use those; else randomly init tile
        self.locked = False
        if magnets is not None and len(magnets) > 1:
            assert type(magnets) == list
            self.magnets = magnets
        else:
            # always a magnet at the origin (centre)
            self.magnets = [
                Magnet(
                    np.random.randint(0, max_symbol, 2),
                    np.array((tile_size, tile_size)) / 2,
                    np.pi / 2,
                    mag_w,
                    mag_h,
                    0,
                )
            ]
            num_mags = 2  # dangerous to do more
            for _ in range(num_mags - 1):
                # try to place magnet randomly (throw error after many attempts fail)
                # currently only midpoint must be in bounds!
                max_attempts = 50
                for attempts in range(max_attempts + 1):
                    # make random magnet in tile
                    new_mag = Magnet(
                        np.random.randint(0, max_symbol, 2),
                        np.array(np.random.uniform(high=tile_size, size=2)),
                        np.random.uniform(low=0, high=2 * np.pi),
                        mag_w,
                        mag_h,
                        0,
                    )

                    # keep if no overlaps else try again
                    if not new_mag.is_intersecting(self.magnets):
                        self.append(new_mag)
                        break
                assert attempts < max_attempts
        self.locked = True

    def __repr__(self):
        return repr(self.magnets)

    def __len__(self):
        return len(self.magnets)

    def append(self, item):
        assert not self.locked
        self.magnets.append(item)

    def remove(self, item):
        assert not self.locked
        self.magnets.remove(item)

    def __getitem__(self, sliced):
        return self.magnets[sliced]

    def __setitem__(self, key, value):
        assert not self.locked
        self.magnets[key] = value

    def __iter__(self):
        return iter(self.magnets)

    def apply_to_mag(self, mag, current_iter=None):
        """
        creates new magnets by applying tile to a magnet, checks symbols match before applying.
        does not check for overlaps!
        """
        new_mags = []
        origin = (0, 0)

        origin_index = 0  # only use origin magnet
        i = 0
        for angle_offset in [0, np.pi]:  # do twice for the 180 degree offset
            magnet_symbol_index = 0 if mag.angle < np.pi else -1

            # equivalent to 0 if (mag.angle + angle_offset)% (2Pi) < np.pi else -1
            tile_symbol_index = (
                magnet_symbol_index if angle_offset else (-1 * magnet_symbol_index - 1)
            )
            if (
                self[origin_index].symbol[tile_symbol_index]
                != mag.symbol[magnet_symbol_index]
            ):
                continue

            # copy tile to use as the new magnets to add
            new_tile = self.copy(current_iter)
            angle_diff = mag.angle - new_tile[origin_index].angle + angle_offset
            # rotate all magnets in tile (we don't care about the displacement so we can use origin=(0,0))
            new_tile.i_rotate(angle_diff, origin)

            pos_diff = mag.pos - new_tile[origin_index].pos
            # translate all magnets and add to new_mags (except tile[tileMagIndex])
            for i in [j for j in range(len(new_tile)) if j != origin_index]:
                new_tile[i].i_translate(*pos_diff)
                new_mags.append(new_tile[i])
                new_tile[i].locked = True

        return new_mags

    def copy(self, created=None):
        return Tile(magnets=[mag.copy(created=created) for mag in self])

    def i_rotate(self, angle, origin):
        for magnet in self:
            magnet.i_rotate(angle, origin)

    def plot(self, **kwargs):
        Individual.print_mags(self, **kwargs)


class Magnet:
    def __init__(self, symbol, pos, angle, mag_w, mag_h, created=None, padding=20):
        self.symbol = symbol
        self.pos = pos
        self.angle = np.mod(angle, 2 * np.pi)
        self.mag_w = mag_w
        self.mag_h = mag_h
        self.created = created
        # padding 20nm results in min possible distance between 2 magnets as 20nm (pads each side by 20/2)
        self.padding = padding
        self.locked = False

        self._as_polygon = None
        self._bound = None

    def recalculate_polygon_and_bound(self):
        self.init_polygon()

    def clear_cached_polygon(self):
        self._as_polygon = None
        self._bound = None

    def get_bound(self):
        if self._bound is None:
            self.recalculate_polygon_and_bound()
        return self._bound

    def get_polygon(self):
        if self._as_polygon is None:
            self.recalculate_polygon_and_bound()
        return self._as_polygon

    def __eq__(self, other):
        if type(other) != Magnet:
            return False

        return (
            self.symbol == other.symbol
            and (self.pos == other.pos).all()
            and self.angle == other.angle
            and self.mag_w == other.mag_w
            and self.mag_h == other.mag_h
            and self.created == other.created
        )

    def __ne__(self, other):
        return not self == other

    IGNORED_VARS = ["as_polygon", "locked", "bound", "_as_polygon", "_bound"]

    def __repr__(self):
        return repr(
            {
                k: v
                for (k, v) in vars(self).items()
                if k not in Magnet.IGNORED_VARS
            }
        )

    def init_polygon(self):
        """return polygon repr of magnet and polygon repr of the padding boundary"""
        half_diagonal = (0.5 * self.mag_w, 0.5 * self.mag_h)
        min_x, min_y = self.pos - half_diagonal
        max_x, max_y = self.pos + half_diagonal
        rect = box(min_x, min_y, max_x, max_y)

        half_pad = self.padding / 2
        bounds = box(
            min_x - half_pad, min_y - half_pad, max_x + half_pad, max_y + half_pad
        )
        my_angle = self.angle
        self._as_polygon = rotate(rect, my_angle, "centroid", use_radians=True)
        self._bound = rotate(bounds, my_angle, "centroid", use_radians=True)

    def is_intersecting(self, others):
        # others may be a magnet or list of magnets
        if type(others) is not list:
            others = [others]
        prepped_poly = prep(self.get_bound())

        for o in others:
            if prepped_poly.intersects(o.get_bound()):
                return True
        return False

    @staticmethod
    def any_intersecting(magnets):
        for i in range(len(magnets)):
            if magnets[i].is_intersecting(magnets[i + 1: len(magnets)]):
                return True
        return False

    def as_patch(self):
        patch = patches.Polygon(np.array(self.get_polygon().exterior.coords))
        patch.iterCreated = self.created
        return patch

    def i_rotate(self, angle, origin):
        # inplace, rotate anticlockwise
        assert not self.locked

        poly = self.get_polygon()
        bound = self.get_bound()
        self.angle = (self.angle + angle) % (2 * np.pi)
        self._as_polygon = rotate(poly, angle, origin, use_radians=True)
        self._bound = rotate(bound, angle, origin, use_radians=True)
        self.pos = np.array(self._as_polygon.centroid.coords).reshape(2)

    def i_translate(self, x, y):
        assert not self.locked
        # inplace
        poly = self.get_polygon()
        bound = self.get_bound()
        self.pos += np.array((x, y))
        self._as_polygon = translate(poly, x, y)
        self._bound = translate(bound, x, y)

    def i_set_pos(self, x, y):
        self.i_translate(x - self.pos[0], y - self.pos[1])

    def i_set_rot(self, angle):
        self.i_rotate(angle - self.angle, "centroid")

    @staticmethod
    def rot(v, angle):
        v = deepcopy(v)
        v[0], v[1] = (
            v[0] * np.cos(angle) - v[1] * np.sin(angle),
            v[0] * np.sin(angle) + v[1] * np.cos(angle),
        )
        return v

    def copy(self, *, created=None):
        v = {k: v for (k, v) in vars(self).items() if k not in Magnet.IGNORED_VARS}
        if created is not None:
            v.update({"created": created})
        v["pos"] = v["pos"].copy()  # derefernce
        v["angle"] = v["angle"].copy()
        v["symbol"] = v["symbol"].copy()
        return Magnet(**v)


def centre_magnets(magnets, centre_point=(0, 0)):
    """modifies inplace, but also returns magnets"""
    centres = np.array(list(map(lambda mag: mag.pos, magnets)))
    max_x, max_y = centres.max(axis=0)
    min_x, min_y = centres.min(axis=0)
    x_shift = centre_point[0] - (max_x + min_x) * 0.5
    y_shift = centre_point[1] - (max_y + min_y) * 0.5
    shift = np.array((x_shift, y_shift))
    for mag in magnets:
        mag.locked = False
        mag.i_translate(x_shift, y_shift)
        mag.locked = True
        # mag.pos += shift

    return magnets


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
    import argparse
    from flatspin.cmdline import StoreKeyValue, eval_params
    from base_individual import make_parser

    parser = make_parser()
    args = parser.parse_args()

    evolved_params = eval_params(args.evolved_param)
    if args.evo_rotate:
        evolved_params["initial_rotation"] = [0, 2 * np.pi]

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
