# vim: tw=120
import numpy as np
import pickle as pkl
import warnings

from flatspin import plotting
from shapely.geometry import box, MultiPolygon
from shapely.affinity import rotate, translate
from shapely.prepared import prep
from itertools import count
from copy import deepcopy, copy
from collections import Sequence, OrderedDict
from time import sleep
import evo_alg as ea
from PIL import Image

from flatspin.data import Dataset, read_table, load_output, is_archive_format
from flatspin.grid import Grid
from flatspin.utils import get_default_params, import_class
from flatspin.runner import run, run_dist, run_local
from flatspin.sweep import sweep
import os
import pandas as pd
import shlex
import sys

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, writers
import logging


class Individual:
    _id_counter = count(0)
    _evolved_params = {}

    @staticmethod
    def set_evolved_params(evolved_params):
        Individual._evolved_params = evolved_params

    def __init__(self, *, max_tiles=1, tile_size=600, mag_w=220, mag_h=80, max_symbol=1, pheno_size=40,
                 pheno_bounds=None, age=0, id=None, gen=0, fitness=None, fitness_components=None, fitness_info=None,
                 tiles=None, init_pheno=True, evolved_params_values=None, fixed_geom=False, **kwargs):

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
        self.evolved_params_values = (
            evolved_params_values if evolved_params_values else {}
        )
        if any((ep not in Individual._evolved_params for ep in self.evolved_params_values)):
            warnings.warn(
                "Unexpected evolved parameter passed to Individual constructor, this will not be mutated correctly!"
            )
        for param in Individual._evolved_params:
            if self.evolved_params_values.get(param) is None:
                self.evolved_params_values[param] = np.random.uniform(
                    Individual._evolved_params[param]["low"],
                    Individual._evolved_params[param]["high"],
                    Individual._evolved_params[param].get("shape"),
                )
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
        ignore_attributes = ("pheno",)
        return repr({k: v for (k, v) in vars(self).items() if k not in ignore_attributes})

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
    def from_string(s, **overides):
        array = np.array
        inf = np.inf
        params = eval(s)
        params.update(overides)

        if not params.get("fixed_geom", False):
            # Instanciate Magnets from result of repr
            params["tiles"] = [Tile(magnets=[Magnet(**mag) for mag in tile]) for tile in params["tiles"]]
        return Individual(**params)

    def geno2pheno(self, geom_size=40, animate=False, no_change_terminator=1):
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
            self.anime = Individual.frames2animation(frames)
            # plt.show()
            # self.anime.save("anime.mp4")

        return (centre_magnets(frozen + frontier)[:geom_size]if np.isfinite(geom_size) else
                centre_magnets(frozen + frontier))

    @staticmethod
    def frames2animation(frames, interval=400, title=False, ax_color="k", color_unchanged=False, figsize=(8, 6), axis_off=False):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        bounds = np.vstack(
            list(map(lambda m: m.xy, [mag for frame in frames for mag in frame]))
        )
        xlim = bounds[:, 0].min(), bounds[:, 0].max()
        ylim = bounds[:, 1].min(), bounds[:, 1].max()
        if title:
            title = list(title)

        def step(i, maxi, xlim, ylim, title):
            ax.cla()
            for poly in frames[i]:
                ax.add_patch(poly)
            if title:
                ax.set_title(title[i] if len(title) > i else title[-1])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if axis_off:
                plt.axis('off')
                ax.add_artist(ax.patch)
                ax.patch.set_zorder(-1)
                fig.subplots_adjust(left=None, bottom=None, right=None, wspace=None, hspace=None)


            fig.canvas.draw()

        if not color_unchanged:
            colours = ea.rainbow_colours(len(frames))
            for frame in frames:
                for poly in frame:
                    poly.set_color(colours[poly.iterCreated % len(colours)])
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

    @staticmethod
    def gauss_mutate(x, std, low=None, high=None):
        x = np.random.normal(x, std)
        if low is not None or high is not None:
            x = np.clip(x, low, high)
        return x

    def mutate(self, strength=1):
        """mutate an Individual to produce children, return any children  as a list"""
        clone = self.copy(refresh=False)
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
    def crossover_evo_params(parents):
        """return new dict of evo params from randomly choosing between params of each parent"""
        evo_params = deepcopy(parents[0].evolved_params_values)
        for param, rnd in zip(evo_params, np.random.random(len(evo_params))):
            if rnd > 0.5:
                evo_params[param] = deepcopy(parents[1].evolved_params_values[param])
        return evo_params

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
    def mutate_evo_param(clone, strength):
        param_name = np.random.choice(list(clone.evolved_params_values))
        mut_param_info = Individual._evolved_params[param_name]

        new_val = Individual.gauss_mutate(
            clone.evolved_params_values[param_name],
            strength * (mut_param_info["high"] - mut_param_info["low"]) / 200,
        )

        res_info = f"{param_name} changed {clone.evolved_params_values[param_name]} -> {new_val}"

        if new_val == clone.evolved_params_values[param_name]:
            # mutation failed, terminate clone!
            clone = None
        else:
            clone.evolved_params_values[param_name] = new_val

        return res_info

    @staticmethod
    def known_spinice(name, min_dist=1e-6, **kwargs):
        min_dist = (min_dist,) * 2 if np.isscalar(min_dist) else min_dist
        if name == "square":
            ind = Individual(init_pheno=False, max_tiles=1, **kwargs)
            t = ind.tiles[0]
            t[1].i_set_rot(0)
            t[1].i_set_pos(t[0].pos[0] + t[1].mag_w / 2 + t[1].mag_h / 2 + t[1].padding + min_dist[0],
                           t[0].pos[1] + t[1].mag_w / 2 + t[1].mag_h / 2 + t[1].padding + min_dist[1])
        elif name == "ising":
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
            ind = Individual(init_pheno=False, max_tiles=1, **kwargs)
            t = ind.tiles[0]
            t[1].i_set_rot(0)
            t[1].i_set_pos(t[0].pos[0],
                           t[0].pos[1] + t[1].mag_w / 2 + t[1].mag_h / 2 + t[1].padding + min_dist[1])
        elif name == "kagome":
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
    def tessellate(magnets, shape=(5, 1), padding=0, centre=True):
        magnets = [mag.copy() for mag in magnets]
        polygons = MultiPolygon([mag.as_polygon for mag in magnets])
        minx, miny, maxx, maxy = polygons.bounds
        cell_size = np.array([maxx - minx, maxy - miny])
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
        self.as_polygon, self.bound = self.init_polygon()
        self.locked = False

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

    def __repr__(self):
        return repr(
            {
                k: v
                for (k, v) in vars(self).items()
                if k not in ["as_polygon", "locked", "bound"]
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
        return rotate(rect, self.angle, use_radians=True), rotate(
            bounds, self.angle, use_radians=True
        )

    def is_intersecting(self, others):
        # others may be a magnet or list of magnets
        if type(others) is not list:
            others = [others]
        prepped_poly = prep(self.bound)
        
        for o in others:
            if prepped_poly.intersects(o.bound):
                return True
        return False

    @staticmethod
    def any_intersecting(magnets):
        for i in range(len(magnets)):
            if magnets[i].is_intersecting(magnets[i + 1: len(magnets)]):
                return True
        return False

    def as_patch(self):
        patch = patches.Polygon(np.array(self.as_polygon.exterior.coords))
        patch.iterCreated = self.created
        return patch

    def i_rotate(self, angle, origin):
        # inplace, rotate anticlockwise
        assert not self.locked
        self.angle = (self.angle + angle) % (2 * np.pi)
        self.as_polygon = rotate(self.as_polygon, angle, origin, use_radians=True)
        self.bound = rotate(self.bound, angle, origin, use_radians=True)
        self.pos = np.array(self.as_polygon.centroid.coords).reshape(2)

    def i_translate(self, x, y):
        assert not self.locked
        # inplace
        self.pos += np.array((x, y))
        self.as_polygon = translate(self.as_polygon, x, y)
        self.bound = translate(self.bound, x, y)

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
        ignore = ["as_polygon", "locked", "bound"]
        v = {k: v for (k, v) in vars(self).items() if k not in ignore}
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
        mag.pos += shift
        # need to remake the polys with new pos
        mag.as_polygon, mag.bound = mag.init_polygon()
    return magnets


# ===================  FITNESS EVAL ========================
def evaluate_outer(outer_pop, basepath, *, max_age=0, acc=np.sum, **kwargs):
    """uses given accumulator func to reduce the fitness components to one value"""
    for i in outer_pop:
        i.fitness = acc(i.fitness_components)
    return outer_pop


def evaluate_outer_find_all(outer_pop, basepath, *, max_value=19, min_value=1, **kwargs):
    novelty_file = os.path.join(basepath, "novelty.pkl")
    if not os.path.exists(novelty_file):
        found = [-1] * (1 + max_value - min_value)
        found_id = [-1] * (1 + max_value - min_value)
    else:
        with open(novelty_file, "rb") as f:
            found, found_id = pkl.load(f)

    for i in outer_pop:
        fit = np.sum(i.fitness_components)
        if not np.isfinite(fit) or fit > max_value or fit < min_value:
            i.fitness = np.nan
            continue

        fit -= min_value

        dist = dist2missing(fit, found)
        if not np.isfinite(dist):
            # all found
            i.fitness = -1
            continue
        if dist == 0:
            i.fitness = 0
            # zero out nearby
            zero_upto_missing(found, fit)
            found_id[fit] = i.id

            continue

        if dist - int(dist) == 0:
            dist = int(dist)
            found[fit] = dist
        i.fitness = dist
    with open(novelty_file, "wb") as f:
        pkl.dump((found, found_id), f)
    print(found)
    print(found_id)
    return outer_pop


def dist2missing(x, found, missing=-1):
    """given index x, find smallest distance to a missing value in found"""
    if found[x] == missing:
        return 0
    if found[x] != 0:
        return found[x]
    left_dist = np.inf
    count = 0
    for j in range(x, -1, -1):
        if found[j] == missing:
            left_dist = count
            break
        elif found[j] != 0:
            left_dist = found[j] + count
            break
        count += 1
    right_dist = np.inf
    count = 0
    for j in range(x, len(found)):
        if found[j] == missing:
            right_dist = count
            break
        elif found[j] != 0:
            right_dist = found[j] + count
            break
        count += 1
    dist = np.min((left_dist, right_dist))
    return dist


def zero_upto_missing(found, x, missing=-1):
    """zero out values to left and right of x upto a missing value, missing values are negative"""
    found[x] = 0
    for i in range(x, -1, -1):
        if found[i] == missing:
            break
        found[i] = 0
    for i in range(x, len(found)):
        if found[i] == missing:
            break
        found[i] = 0


def scale_to_unit(x, upper, lower):
    return (x - lower) / (upper - lower)


def get_default_shared_params(outdir="", gen=None):
    default_params = {
        "run": "local",
        "model": "CustomSpinIce",
        "encoder": "AngleSine",
        "H": 0.01,
        "phi": 90,
        "radians": True,
        "alpha": 30272,
        "sw_b": 0.4,
        "sw_c": 1,
        "sw_beta": 3,
        "sw_gamma": 3,
        "spp": 100,
        "hc": 0.03,
        "periods": 10,
        "neighbor_distance": 1000,
    }
    if gen is not None:
        outdir = os.path.join(outdir, f"gen{gen}")
    default_params["basepath"] = outdir

    return default_params


def get_default_run_params(pop, sweep_list, *, condition=None):
    sweep_list = sweep_list or [[0, 0, {}]]

    if not condition:

        def condition(x):
            return True

    elif condition == "fixed_size":

        def condition(ind):
            return len(ind.pheno) >= ind.pheno_size

    id2indv = {
        individual.id: individual for individual in [p for p in pop if condition(p)]
    }

    run_params = []
    for id, indv in id2indv.items():
        for i, j, rp in sweep_list:
            if indv.fixed_geom:
                run_params.append(dict(rp, indv_id=id, sub_run_name=f"_{i}_{j}"))
            else:
                coords = (
                    np.array([mag.pos for mag in indv.pheno])
                    if not indv.fixed_geom
                    else [0]
                )
                angles = (
                    np.array([mag.angle for mag in indv.pheno])
                    if not indv.fixed_geom
                    else [0]
                )
                run_params.append(
                    dict(
                        rp,
                        indv_id=id,
                        magnet_coords=coords,
                        magnet_angles=angles,
                        sub_run_name=f"_{i}_{j}",
                    )
                )

    return run_params


def flatspin_eval(fit_func, pop, gen, outdir, *, run_params=None, shared_params=None, do_not_override_default=False,
                  sweep_params=None, condition=None, group_by=None, max_jobs=1000,
                  repeat=1, repeat_spec=None, preprocessing=None, **flatspin_kwargs):
    """
    fit_func is a function that takes a dataset and produces an iterable (or single value) of fitness components.
    if an Individual already has fitness components the value(s) will be appended
    (allows for multiple datasets per Individual)
    """
    if len(pop) < 1:
        return pop
    sweep_list = (
        list(sweep(sweep_params, repeat, repeat_spec, params=flatspin_kwargs))
        if sweep_params
        else []
    )
    default_shared = get_default_shared_params(outdir, gen)
    if shared_params is None:
        shared_params = default_shared
    elif not do_not_override_default:
        default_shared.update(shared_params)
        shared_params = default_shared

    shared_params.update(flatspin_kwargs)

    if not condition:

        def condition(x):
            return True

    elif condition == "fixed_size":

        def condition(ind):
            return len(ind.pheno) >= ind.pheno_size

    if run_params is None:
        run_params = get_default_run_params(pop, sweep_list, condition=condition)

    if preprocessing:
        run_params = preprocessing(run_params)

    if len(run_params) > 0:
        id2indv = {individual.id: individual for individual in pop}
        evolved_params = [
            id2indv[rp["indv_id"]].evolved_params_values for rp in run_params
        ]
        evo_run(run_params, shared_params, gen, evolved_params, max_jobs=max_jobs, wait=group_by)
        dataset = Dataset.read(shared_params["basepath"])
        queue = dataset
        if group_by:
            _, queue = zip(*dataset.groupby(group_by))
        queue = list(queue)
        while len(queue) > 0:
            ds = queue.pop(0)
            with np.errstate():
                indv_id = ds.index["indv_id"].values[0]
                try:
                    # calculate fitness of a dataset
                    fit_components = fit_func(ds)
                    try:
                        fit_components = list(fit_components)
                    except (TypeError):
                        fit_components = [fit_components]

                    # assign the fitness of the correct individual

                    indv = id2indv[indv_id]
                    if indv.fitness_components is not None:
                        indv.fitness_components += fit_components
                    else:
                        indv.fitness_components = fit_components
                except Exception as e:  # not done saving file
                    if shared_params["run"] != "dist" or group_by:
                        raise e
                    if type(e) != FileNotFoundError:
                        print(type(e), e)
                    queue.append(ds)  # queue.append((indv_id, ds))
                    sleep(2)

    for indv in [i for i in pop if not condition(i)]:
        indv.fitness_components = [np.nan]
    return pop


def evo_run(runs_params, shared_params, gen, evolved_params=None, wait=False, max_jobs=1000):
    """modified from run_sweep.py main()"""
    if not evolved_params:
        evolved_params = []
    model_name = shared_params.pop("model", "CustomSpinIce")
    model_class = import_class(model_name, "flatspin.model")
    encoder_name = shared_params.get("encoder", "Sine")
    encoder_class = (import_class(encoder_name, "flatspin.encoder") if type(encoder_name) is str else encoder_name)

    data_format = shared_params.get("format", "npz")

    params = get_default_params(run)
    params["encoder"] = f"{encoder_class.__module__}.{encoder_class.__name__}"
    params.update(get_default_params(model_class))
    params.update(get_default_params(encoder_class))
    params.update(shared_params)

    info = {
        "model": f"{model_class.__module__}.{model_class.__name__}",
        "model_name": model_name,
        "data_format": data_format,
        "command": " ".join(map(shlex.quote, sys.argv)),
    }

    ext = data_format if is_archive_format(data_format) else "out"

    outdir_tpl = "gen{:d}indv{:d}"

    basepath = params["basepath"]
    if os.path.exists(basepath):
        # Refuse to overwrite an existing dataset
        raise FileExistsError(basepath)
    os.makedirs(basepath)

    index = []
    filenames = []
    # Generate queue
    for i, run_params in enumerate(runs_params):
        newparams = copy(params)
        newparams.update(run_params)
        if evolved_params:
            # get any flatspin params in evolved_params and update run param with them
            run_params.update(
                {k: v for k, v in evolved_params[i].items() if k in newparams}
            )
        sub_run_name = newparams.get("sub_run_name", "x")
        outdir = outdir_tpl.format(gen, newparams["indv_id"]) + f"{sub_run_name}.{ext}"
        filenames.append(outdir)
        row = OrderedDict(run_params)
        row.update({"outdir": outdir})
        index.append(row)

    # Save dataset
    index = pd.DataFrame(index)
    dataset = Dataset(index, params, info, basepath)
    dataset.save()

    # Run!
    # print("Starting sweep with {} runs".format(len(dataset)))
    rs = np.random.get_state()
    run_type = shared_params.get("run", "local")
    if run_type == "local":
        run_local(dataset, False)

    elif run_type == "dist":
        run_dist(dataset, wait=wait, max_jobs=max_jobs)

    np.random.set_state(rs)
    return


def flips_fitness(pop, gen, outdir, num_angles=1, other_sizes_fractions=None, sweep_params=None, repeat=1,
                  repeat_spec=None, **flatspin_kwargs):
    shared_params = get_default_shared_params(outdir, gen)
    shared_params.update(flatspin_kwargs)
    if not other_sizes_fractions:
        other_sizes_fractions = []
    if num_angles > 1:
        shared_params["input"] = [0, 1] * (shared_params["periods"] // 2)
    sweep_list = (list(sweep(sweep_params, repeat, repeat_spec, params=flatspin_kwargs)) if sweep_params else [])
    run_params = get_default_run_params(pop, sweep_list, condition=flatspin_kwargs.get("condition"))
    frac_run_params = []
    if len(run_params) > 0:
        for rp in run_params:
            rp.setdefault("sub_run_name", "")

            for frac in other_sizes_fractions:
                angles_frac = rp["magnet_angles"][
                    : int(np.ceil(len(rp["magnet_angles"]) * frac))
                ]
                coords_frac = rp["magnet_coords"][
                    : int(np.ceil(len(rp["magnet_coords"]) * frac))
                ]
                frp = {
                    "indv_id": rp["indv_id"],
                    "magnet_coords": coords_frac,
                    "magnet_angles": angles_frac,
                    "sub_run_name": rp["sub_run_name"] + f"_frac{frac}",
                }
                frac_run_params.append(dict(rp, **frp))
            rp["sub_run_name"] += f"_frac{1}"

    def fit_func(ds):
        # fitness is number of steps, but ignores steps from first fifth of the run
        steps = read_table(ds.tablefile("steps"))
        fitn = (
            steps.iloc[-1]["steps"]
            - steps.iloc[(shared_params["spp"] * shared_params["periods"]) // 5][
                "steps"
            ]
        )
        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, shared_params=shared_params, run_params=run_params + frac_run_params,
                        repeat=repeat, repeat_spec=repeat_spec, **flatspin_kwargs)
    return pop


def target_state_num_fitness(pop, gen, outdir, target, state_step=None, **flatspin_kwargs):
    def fit_func(ds):
        nonlocal state_step
        if state_step is None:
            state_step = ds.params["spp"]
        spin = read_table(ds.tablefile("spin"))
        fitn = abs(len(np.unique(spin.iloc[::state_step, 1:], axis=0)) - target)
        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, **flatspin_kwargs)
    return pop


def majority_fitness(pop, gen, outdir, sweep_params, test_at=None, match=True, **flatspin_kwargs):
    if not test_at:
        test_at = [0.2, 0.4, 0.6, 0.8]

    if "test_perc" in sweep_params:
        warnings.warn("majority fitness function overwriting value of 'test_perc'")

    if "random_prob" in sweep_params:
        warnings.warn("majority fitness function overwriting value of 'random_prob'")
    sweep_params = dict(sweep_params, test_perc=str(test_at), random_prob="[test_perc]")

    def preprocessing(run_params):
        """mod angles, enforce odd number of spins"""
        for run in run_params:
            run["magnet_angles"] %= np.pi
            if len(run["magnet_angles"]) % 2 == 0:
                run["magnet_angles"] = run["magnet_angles"][:-1]
                run["magnet_coords"] = run["magnet_coords"][:-1]
            run["random_seed"] = np.random.randint(999999)
        return run_params

    def fit_func(ds):
        spin = read_table(ds.tablefile("spin"))
        majority_symbol = spin.iloc[0].mode()[0]
        if match:
            fitn = np.sum(spin.iloc[-1] == majority_symbol)
        else:
            fitn = np.sum(spin.iloc[-1] != majority_symbol)
        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, preprocessing=preprocessing, init="random", sweep_params=sweep_params,
                        **flatspin_kwargs)
    return pop


def image_match_fitness(pop, gen, outdir, image_file_loc, num_blocks=33, threshold=True, **flatspin_kwargs):
    img = np.asarray(Image.open(image_file_loc))
    l = []
    step = len(img) / num_blocks
    for y in range(num_blocks):
        row = []
        for x in range(num_blocks):
            a = img[
                int(x * step): int((x + 1) * step), int(y * step): int((y + 1) * step)
            ]
            row.append(np.mean(a))
        l.append(row)

    target = np.array(l)
    target = np.flipud(target).flatten()
    if threshold:
        target = (target > (255 / 2)) * 255

    def fit_func(ds):
        UV = load_output(ds, "mag", t=-1, grid_size=(num_blocks,) * 2, flatten=False)
        U = UV[..., 0]  # x components
        V = UV[..., 1]  # y components
        angle = plotting.vector_colors(U, V)
        colour = np.cos(angle).flatten()
        magn = np.linalg.norm(UV, axis=-1).flatten()

        # scale colour by magnitude between -1 and 1
        colour = colour * magn / np.max(magn)
        # scale colour from 0 to 255
        colour = (colour + 1) * (255 / 2)

        fitn = np.sum(np.abs(colour - target))
        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, **flatspin_kwargs)
    return pop


def mean_abs_diff_error(y_true, y_pred):
    # print(f"y_true: {y_true}")
    # print(f"y_pred: {y_pred}")
    np.abs(y_true - y_pred)
    return np.abs(y_true - y_pred).mean()


def xor_fitness(pop, gen, outdir, quantity="spin", grid_size=None, crop_width=None, win_shape=None, win_step=None, cv_folds=10,
                alpha=1, sweep_params=None, encoder="Constant", angle0=-45, angle1=45, H0=0, H=1000, input=1000, spp=1, **kwargs):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import make_scorer, accuracy_score

    sweep_params = sweep_params if sweep_params else {}

    def reduce_angle(a):
        return (a + 180) % 360 - 180

    def diff_bisector(a0, a1, bool0=0, bool1=0):
        a0 += 180 * bool0
        a1 += 180 * bool1
        return a0 + reduce_angle(a1 - a0) / 2

    # calculate simulated angles from the logical axes
    logic_values = ("00", "01", "10", "11")
    # default_angles = [diff_bisector(angle0, angle1, b0, b1) for b0 in (0, 1) for b1 in (0, 1)]
    sweep_params["logical_val"] = str(logic_values)

    id2indv = {individual.id: individual for individual in pop}

    def preprocessing(run_params):
        """calculate phi value from logical value"""
        for run_param in run_params:
            ind = id2indv[run_param["indv_id"]]
            a0 = (
                ind.evolved_params_values["angle0"]
                if "angle0" in ind.evolved_params_values
                else angle0
            )
            a1 = (
                ind.evolved_params_values["angle1"]
                if "angle1" in ind.evolved_params_values
                else angle1
            )
            b0, b1 = [b == "1" for b in run_param["logical_val"]]
            run_param["phi"] = diff_bisector(a0, a1, b0, b1)
        return run_params

    if np.isscalar(input):
        if "input" in sweep_params:
            print("Overwriting 'input' in xor_fitness()!!")
        input = [1] + [0] * (input - 1)

    def fit_func(dataset):
        scores = []

        X = []  # reservoir outputs
        y = []  # targets
        for ds in dataset:
            logic_val = ds.index["logical_val"].values[0]
            target = logic_val in ["01", "10"]  # calculate xor
            y.append(target)
            x = read_table(ds.tablefile("spin")).iloc[-1].values[1:]
            X.append(x)
        X = np.array(X)
        y = np.array(y)
        # print(f"X: {X}")
        # print(X.shape)
        # print(f"y: {y}")
        # print(y.shape)
        readout = Ridge(alpha=alpha)
        # readout.fit(X, y)

        cv = KFold(n_splits=cv_folds, shuffle=False)
        cv_scores = cross_val_score(readout, X, y, cv=cv,
                                    scoring=make_scorer(mean_abs_diff_error, greater_is_better=False), n_jobs=1)
        # score is -error (max better)
        scores.append(cv_scores)
        fitness_components = np.mean(scores, axis=-1)

        return fitness_components

    pop = flatspin_eval(fit_func, pop, gen, outdir, encoder=encoder, sweep_params=sweep_params, H=H, H0=H0, input=input,
                        spp=spp, preprocessing=preprocessing, **kwargs)
    return pop


def mem_capacity_fitness(pop, gen, outdir, n_delays=10, **kwargs):
    from mem_capacity import do_mem_capacity

    def fit_func(ds):
        delays = np.arange(0, n_delays + 1)
        spp = int(ds.params["spp"])
        t = slice(spp - 1, None, spp)
        scores = do_mem_capacity(ds, delays, t=t)
        fitness_components = scores.mean(axis=-1)
        print("MC", np.sum(fitness_components), len(ds))
        return fitness_components

    pop = flatspin_eval(fit_func, pop, gen, outdir, **kwargs)

    return pop


def correlation_fitness(pop, gen, outdir, target, **kwargs):
    from runAnalysis import fitnessFunction

    def fit_func(x):
        return abs(fitnessFunction(x) - target)

    pop = flatspin_eval(fit_func, pop, gen, outdir, **kwargs)

    return pop


def parity_fitness(pop, gen, outdir, n_delays=10, n_bits=3, **kwargs):
    from parity import do_parity

    def fit_func(ds):
        delays = np.arange(0, n_delays)
        spp = int(ds.params["spp"])
        t = slice(spp - 1, None, spp)
        scores = do_parity(ds, delays, n_bits, t=t)
        fitness_components = scores.mean(axis=-1)
        print(f"PARITY{n_bits}", np.sum(fitness_components))
        return fitness_components

    pop = flatspin_eval(fit_func, pop, gen, outdir, **kwargs)

    return pop


def state_num_fitness(pop, gen, outdir, state_step=None, **flatspin_kwargs):
    def fit_func(ds):
        nonlocal state_step
        if state_step is None:
            state_step = ds.params["spp"]
        spin = read_table(ds.tablefile("spin"))
        fitn = len(np.unique(spin.iloc[::state_step, 1:], axis=0))
        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, **flatspin_kwargs)
    return pop


def state_num_fitness2(pop, gen, outdir, t=-1, bit_len=3, sweep_params=None, group_by=None, **flatspin_kwargs):
    input = str([list(f"{i:b}".zfill(bit_len)) for i in range(2**bit_len)])

    if not sweep_params:
        sweep_params = {}
    if "init" in sweep_params or "input" in sweep_params:
        warnings.warn("Overiding input in fitness function")
    sweep_params = dict(sweep_params, input=input)

    if not group_by:
        group_by = []
    if "indv_id" not in group_by:
        group_by.append("indv_id")

    def fit_func(datasets):
        states = None
        for i, ds in enumerate(datasets):
            spin = read_table(ds.tablefile("spin"))
            if states is None:
                states = np.zeros((len(datasets), *spin.iloc[t].shape))
            states[i] = spin.iloc[t]
        fitn = len(np.unique(states, axis=0))
        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, sweep_params=sweep_params, group_by=group_by, **flatspin_kwargs)
    return pop


def pheno_size_fitness(pop, gen, outdir, **flatspin_kwargs):
    id2indv = {individual.id: individual for individual in pop}
    shared_params = {"spp": 1, "periods": 1, "H": 0, "neighbor_distance": 1}

    def fit_func(ds):
        return len(id2indv[ds.index["indv_id"].values[0]].pheno)

    pop = flatspin_eval(fit_func, pop, gen, outdir, condition=lambda x: True,
                        shared_params=shared_params, **flatspin_kwargs)
    return pop


def ca_rule_fitness(pop, gen, outdir, target, group_by=None, sweep_params=None, img_basepath="", compare="direct",
                    **flatspin_kwargs):
    from analyze_sweep import find_rule

    # \from ca_encoder import CARotateEncoder
    default_shared_params = {
        "run": "local",
        "encoder": "ca_encoder.CARotateEncoder",
        "spp": 10,
        "periods": 1,
        "timesteps": 10,
        "basepath": os.path.join(outdir, f"gen{gen}"),
    }
    input = "[[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]]"
    init = str(
        [
            os.path.join(img_basepath, "init_half_0.png"),
            os.path.join(img_basepath, "init_half_1.png"),
        ]
    )
    if not sweep_params:
        sweep_params = {}
    if "init" in sweep_params or "input" in sweep_params:
        warnings.warn("Overiding 'init' and input in fitness function")
    sweep_params = dict(sweep_params, input=input, init=init)
    if not group_by:
        group_by = []
    if "indv_id" not in group_by:
        group_by.append("indv_id")
    if "random_seed" in sweep_params and "random_seed" not in group_by:
        group_by.append("random_seed")

    if compare == "langton":
        langtons_table = {
            x: "{0:08b}".format(x).count("1") / 8 for x in range(0, 256)
        }  # lambda[rule]
    elif compare == "equiv":
        from ca_rule_tools import eq_rules

        equiv_rules = list(filter(lambda x: target in x, eq_rules))[0]
    id2indv = {individual.id: individual for individual in pop}

    def fit_func(ds):
        """takes a group of ds of same indv_id and seed (one full run of all ca inputs on a system)"""
        rule = find_rule((None, ds))[1]
        if compare == "langton":
            fitn = abs(langtons_table[rule] - langtons_table[target])
        elif compare == "equiv":
            fitn = int(rule in equiv_rules)
        else:  # direct compare
            fitn = int(rule == target)
        id = ds.index["indv_id"].values[0]
        indv = id2indv[id]
        indv.fitness_info = [] if indv.fitness_info is None else indv.fitness_info
        indv.fitness_info.append(f"rule {rule}")

        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, group_by=group_by, sweep_params=sweep_params, shared_params=default_shared_params,
                        do_not_override_default=True, **flatspin_kwargs)
    return pop


def stray_field_ca_fitness(pop, gen, outdir, sweep_params, grid_size=(5, 1), target="ClassIII,ClassIV", angle0=0,
                           angle1=np.pi, padding=80, group_by=None, **flatspin_kwargs):
    from ca_rule_tools import full_classes
    """target can be rule number: 93
    a equivelence class : 'eq93'
    or 1 or more classes: 'ClassIV' / 'ClassIII,ClassIV'
    """

    if "init_input" in sweep_params:
        warnings.warn("majority fitness function overwriting value of 'init_input'")

    if type(grid_size) in (int, float):
        grid_size = (grid_size, grid_size)
    sweep_params = dict(
        sweep_params, init_input=str(list(range(2**(grid_size[0] * grid_size[1]))))
    )

    if not group_by:
        group_by = []
    if "indv_id" not in group_by:
        group_by.append("indv_id")

    # parse target
    if type(target) is not str or target.isnumeric():
        target = (int(target),)
    elif "eq" in target:
        target = (int(target[2:]),)
    elif "Class" in target:
        target = frozenset((rule for class_num in target for rule in full_classes[class_num[len("Class"):]]))

    id2indv = {individual.id: individual for individual in pop}

    def preprocessing(run_params):
        """do tessalting"""
        for run in run_params:
            indv = id2indv[run["indv_id"]]
            tess = Individual.tessellate(indv.pheno, grid_size, padding=padding)
            run["magnet_angles"] = np.array([mag.angle for mag in tess])
            run["magnet_coords"] = np.array([mag.pos for mag in tess])

        return run_params

    def fit_func(ds):  # need to group to check rule, or do something clever in outer eval
        rule_table = OrderedDict()
        for run in ds:
            mag = load_output(run, "mag", t=-1, grid_size=grid_size, flatten=False)
            state = mag[:, 1] > 0  # check y comp as we move join and stimulate in x
            init_input = int(run.index['init_input'])
            binput = bin(init_input)[2:]
            rule_table[binput] = state

        rules = check_ca_rules(rule_table)
        # find class
        # calc hamming dist i.e sum(xor())
    pop = flatspin_eval(fit_func, pop, gen, outdir, preprocessing=preprocessing,
                        sweep_params=sweep_params, **flatspin_kwargs)
    return pop


def check_ca_rules(rule_table):
    num_cells = len(rule_table.keys()[0])
    rules = []
    for i in range(1, num_cells-1):
        inputs = [k[i-1:i+1] for k in rule_table]
        outputs = [rule_table[k][i] for k in rule_table]
        rules.append(check_ca_rule(inputs, outputs))
    return rules


def check_ca_rule(inputs, outputs):
    [y for _, y in sorted(zip(map(int, inputs), outputs), reverse=1)]
    sorted_outs = [outputs[i] for i in sorted(range(len(outputs)), key=lambda x: int(inputs[x]), reverse=1)]
    rule_num = int("".join(sorted_outs), 2)
    return rule_num


def std_grid_field_fitness(pop, gen, outdir, angles=np.linspace(0, 2 * np.pi, 8), grid_size=4, **flatspin_kwargs):
    shared_params = {}
    shared_params["phi"] = 360
    shared_params["input"] = (angles % (2 * np.pi)) / (2 * np.pi)

    if np.isscalar(grid_size):
        grid_size = (grid_size, grid_size)

        def fit_func(ds):
            mag = load_output(
                ds, "mag", t=ds.params["spp"], grid_size=grid_size, flatten=False
            )
            magnitude = np.linalg.norm(mag, axis=3)
            summ = np.sum(magnitude, axis=0)
            fitn = np.std(summ) * np.mean(summ)
            return fitn

    pop = flatspin_eval(
        fit_func, pop, gen, outdir, shared_params=shared_params, **flatspin_kwargs
    )
    return pop


def get_range(a):
    mn, mx = minmax(a)
    return mx - mn


def minmax(a):
    a = np.array(a) if type(a) != np.ndarray else a
    if a.ndim > 1:
        a = a.reshape(-1, a.shape[-1])

    return a.min(axis=0), a.max(axis=0)


def target_order_percent_fitness(pop, gen, outdir, grid_size=4, threshold=0.5, condition=None, **flatspin_kwargs):
    # shared_params = {}
    # shared_params["encoder"] = "Rotate"
    # shared_params["input"] = np.linspace(1, 0, shared_params["periods"])
    if np.isscalar(grid_size):
        grid_size = (grid_size, grid_size)

    for i in pop:
        i.poss = [mag.pos for mag in i.pheno]
        i.grid = Grid.fixed_grid(np.array(i.poss), grid_size)

    # check there are magnets in at least half of grid
    # condition = lambda i: (len(np.unique(i.grid._grid_index, axis=0)) >= 0.5 * grid_size[0] * grid_size[1]) and \
    #                       len(i.pheno) >= i.pheno_size
    fixed_size = condition == "fixed_size"

    def condition(indv):
        x, y = get_range(indv.poss)

        return (0.5 * y <= x <= 2 * y) and not (
            fixed_size and len(indv.pheno) < indv.pheno_size
        )

    id2indv = {individual.id: individual for individual in pop}

    def fit_func(ds):
        mag = load_output(ds, "mag", t=-1, grid_size=grid_size, flatten=False)
        magnitude = np.linalg.norm(mag, axis=3)[0]
        indv = id2indv[ds.index["indv_id"].values[0]]
        cells_with_mags = [(x, y) for x, y in np.unique(indv.grid._grid_index, axis=0)]
        # old
        """
        # fitness is std of the magnitudes of the cells minus std of the number of magnets in each cell
        fitn = np.std([magnitude[x][y] for x, y in cells_with_mags]) - \
               np.std([len(indv.grid.point_index([x, y]))
                      for x, y in cells_with_mags])
        """
        fitn = abs(
            (
                (np.array([magnitude[x][y] for x, y in cells_with_mags]) < threshold)
                * 2
                - 1
            ).sum()
        )

        return fitn

    pop = flatspin_eval(
        fit_func, pop, gen, outdir, condition=condition, **flatspin_kwargs
    )
    return pop


def main(outdir=r"results\tileTest", inner="flips", outer="default", minimize_fitness=True, calculate_fit_only=False, **kwargs):
    known_fits = {
        "target_state_num": target_state_num_fitness,
        "state_num": state_num_fitness,
        "flips": flips_fitness,
        "std_grid_field": std_grid_field_fitness,
        "target_order_percent": target_order_percent_fitness,
        "default": evaluate_outer,
        "find_all": evaluate_outer_find_all,
        "pheno_size": pheno_size_fitness,
        "image": image_match_fitness,
        "mem_capacity": mem_capacity_fitness,
        "parity": parity_fitness,
        "majority": majority_fitness,
        "correlation": correlation_fitness,
        "xor": xor_fitness,
        "ca_rule": ca_rule_fitness,
        "state_num2": state_num_fitness2,
    }
    inner = known_fits.get(inner, inner)
    outer = known_fits.get(outer, outer)

    if calculate_fit_only:
        return ea.only_run_fitness_func(outdir, Individual, inner, outer, minimize_fitness=minimize_fitness, **kwargs)
    else:
        return ea.main(outdir, Individual, inner, outer, minimize_fitness=minimize_fitness, **kwargs)


# m = main(outdir=r"results\flatspinTile26",inner=flipsMaxFitness, popSize=3, generationNum=10)
if __name__ == "__main__":
    import argparse
    from flatspin.cmdline import StoreKeyValue, eval_params

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
        "--group-by", nargs="*", help="group by parameter(s) for fitness evaluation"
    )
    parser.add_argument(
        "--calculate-fit-only",
        action="store_true",
        help="use if you only want to run a fitness func once on some individuals (don't run EA)",
    )
    args = parser.parse_args()

    evolved_params = eval_params(args.evolved_param)
    if args.evo_rotate:
        evolved_params["initial_rotation"] = [0, 2 * np.pi]

    outpath = os.path.join(os.path.curdir, args.output)
    logpath = os.path.join(outpath, args.log)
    os.makedirs(outpath)
    logging.basicConfig(filename=logpath, level=logging.INFO)
    main(
        outdir=args.output,
        **eval_params(args.parameter),
        evolved_params=evolved_params,
        individual_params=eval_params(args.individual_param),
        outer_eval_params=eval_params(args.outer_eval_param),
        sweep_params=args.sweep_param,
        repeat=args.repeat,
        repeat_spec=args.repeat_spec,
        group_by=args.group_by,
        calculate_fit_only=args.calculate_fit_only,
    )
