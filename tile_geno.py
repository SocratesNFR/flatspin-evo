import numpy as np
import pickle as pkl

from flatspin import plotting
from shapely.geometry import box
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

import os
import pandas as pd
import shlex
import sys

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, writers


class Individual:
    _id_counter = count(0)
    _evolved_params = {}

    @staticmethod
    def set_evolved_params(evolved_params):
        Individual._evolved_params = evolved_params

    def __init__(self, *, max_tiles=1, tile_size=600, mag_w=220, mag_h=80, max_symbol=1,
                 pheno_size=40, pheno_bounds=None, age=0, id=None, gen=0, fitness=None, fitness_components=None,
                 tiles=None, init_pheno=True, evolved_params_values={}, **kwargs):

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
        if self.pheno_bounds and np.isscalar(pheno_bounds):
            self.pheno_bounds = [pheno_bounds] * 2
        self.fitness = fitness
        self.fitness_components = fitness_components

        self.evolved_params_values = evolved_params_values
        for param in Individual._evolved_params:
            if self.evolved_params_values.get(param) is None:
                self.evolved_params_values[param] = np.random.uniform(Individual._evolved_params[param]["low"],
                                                                      Individual._evolved_params[param]["high"],
                                                                      Individual._evolved_params[param].get("shape"))

        if mag_h > mag_w:
            raise Warning("conversion to flatspin assumes magnet height < magnet width!")
        if tiles is not None and 1 <= len(tiles):
            if len(tiles) > self.max_tiles:
                raise ValueError("Individual has more tiles than the value of 'max_tiles'")
            self.tiles = tiles
        else:
            self.tiles = [Tile(mag_w=mag_w, mag_h=mag_h, tile_size=tile_size, max_symbol=max_symbol) for _ in
                          range(np.random.randint(1, max_tiles + 1))]
        if init_pheno:
            self.pheno = self.geno2pheno(geom_size=self.pheno_size)

    def refresh(self):
        self.pheno = self.geno2pheno(geom_size=self.pheno_size)
        self.fitness = None
        self.fitness_components = None

    def __repr__(self):
        # defines which attributes are ignored by repr
        ignore_attributes = ("pheno")
        return repr({k: v for (k, v) in vars(self).items() if k not in ignore_attributes})

    def copy(self, **overide_kwargs):
        # defines which attributes are used when copying
        ignore_attributes = ("gen", "evolved_params_values", "pheno", "id", "init_pheno")
        params = {k: v for (k, v) in vars(self).items() if k not in ignore_attributes}
        params.update(overide_kwargs)
        new_indv = Individual(**params)
        new_indv.evolved_params_values = deepcopy(new_indv.evolved_params_values)
        # copy attributes that are referenced to unlink
        new_indv.tiles = [Tile(magnets=[mag.copy() for mag in tile]) for tile in new_indv.tiles]

        return new_indv

    @staticmethod
    def from_string(s):
        array = np.array
        inf = np.inf
        params = eval(s)
        # Instanciate Magnets from result of repr
        params["tiles"] = [Tile(magnets=[Magnet(**mag) for mag in tile]) for tile in params["tiles"]]
        return Individual(**params)

    def geno2pheno(self, geom_size=40, animate=False, no_change_terminator=10):
        frontier, frozen = [], []
        iter_count = 0
        frames = []
        since_change = 0
        max_len = 0
        # add origin magnet in the first tile to the frontier
        frontier.append(self.tiles[0][0].copy(created=iter_count))
        frontier[0].symbol.fill(0)
        if "initial_rotation" in self.evolved_params_values:
            frontier[0].i_rotate(self.evolved_params_values["initial_rotation"], 'centroid')
        frames.append(list(map(lambda m: m.as_patch(), frontier + frozen)))

        while len(frontier) + len(frozen) < geom_size and len(frontier) > 0 and since_change < no_change_terminator:
            if animate:
                frames.append(list(map(lambda m: m.as_patch(), frontier + frozen)))
            iter_count += 1
            new_front = []
            for magnet in frontier:
                for tile in self.tiles:
                    new_mags = tile.apply_to_mag(magnet, iter_count)

                    for new_mag in new_mags:
                        # check doesnt intersect any magnets and if there are bounds is within them
                        if not new_mag.is_intersecting(frontier + frozen + new_front) and not (
                                self.pheno_bounds
                                and not (-self.pheno_bounds[0] < new_mag.pos[0] < self.pheno_bounds[0]
                                         and -self.pheno_bounds[1] < new_mag.pos[1] < self.pheno_bounds[1])
                        ):
                            new_front.append(new_mag)

            frozen.extend(frontier)
            frontier = new_front
            if max_len < len(frontier) + len(frozen):
                since_change = 0
                max_len = len(frontier) + len(frozen)
            else:
                since_change += 1
            frames.append(list(map(lambda m: m.as_patch(), frontier + frozen)))

        if animate:
            self.anime = Individual.frames2animation(frames)
            # plt.show()
            # self.anime.save("anime.mp4")

        return centre_magnets(frozen + frontier)[:geom_size] if np.isfinite(geom_size) else centre_magnets(
            frozen + frontier)

    @staticmethod
    def frames2animation(frames, interval=400, title=False, ax_color="k", color_unchanged=False, figsize=(8, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        bounds = np.vstack(list(map(lambda m: m.xy, [mag for frame in frames for mag in frame])))
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

            fig.canvas.draw()

        if not color_unchanged:
            colours = ea.rainbow_colours(len(frames))
            for frame in frames:
                for poly in frame:
                    poly.set_color(colours[poly.iterCreated % len(colours)])
        ax.set_facecolor(ax_color)
        return FuncAnimation(fig, step, frames=len(frames),
                             fargs=(len(frames) - 1, xlim, ylim, title),
                             blit=False, interval=interval)

    @staticmethod
    def print_mags(mags, facecolor=None, edgecolor=None):
        plt.figure()
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
        clone = self.copy(init_pheno=False)
        mut_types = ["magPos", "magAngle", "symbol", "tile"]
        if len(self.evolved_params_values) > 0:
            mut_types.append("param")
        if self.max_tiles == 1:
            mut_types.remove("tile")  # cannot add/remove tiles when max tile is 1
        mut_type = np.random.choice(mut_types)

        if mut_type == "magPos":
            # pick a tile at random, pick a magnet excluding the first magnet
            # and move it (only position of magnet center is confined by tile_size)
            tile = clone.tiles[np.random.randint(0, len(clone.tiles))]
            x = 1 + np.random.randint(len(tile[1:]))
            copy_mag = tile[x].copy()
            distance = Individual.gauss_mutate(copy_mag.pos, strength * self.tile_size / 200, 0,
                                               clone.tile_size) - copy_mag.pos
            copy_mag.i_translate(*distance)

            if not copy_mag.is_intersecting(tile[:x] + tile[x + 1:]):
                # only mutate if does not cause overlap
                tile.locked = False
                tile[x] = copy_mag
                tile.locked = True
            else:
                # return nothing, mutation failed
                return []

        elif mut_type == "magAngle":
            # pick a tile at random, pick a magnet excluding the first magnet and rotate it (about centroid)
            tile = clone.tiles[np.random.randint(0, len(clone.tiles))]
            x = 1 + np.random.randint(len(tile[1:]))
            copy_mag = tile[x].copy()
            rotation = np.random.normal(0, strength * (2 * np.pi) / 200)
            copy_mag.i_rotate(rotation, "centroid")

            if not copy_mag.is_intersecting(tile[:x] + tile[x + 1:]):
                # only mutate if does not cause overlap
                tile.locked = False
                tile[x] = copy_mag
                tile.locked = True
            else:
                # return nothing, mutation failed
                return []

        elif mut_type == "symbol":
            tile = clone.random_tiles()[0]
            magnet = np.random.choice(tile)
            magnet.symbol = np.random.randint(clone.max_symbol, size=2)

        elif mut_type == "tile":
            if len(clone.tiles) == 1:  # if just one don't allow deletion of tile
                mut_types = ["clone tile", "add random tile"]
            # if at max tiles only allow deletion
            elif clone.max_tiles == len(clone.tiles):
                mut_types = ["delete tile"]

            else:
                mut_types = ["clone tile", "delete tile", "add random tile"]

            mut_type = np.random.choice(mut_types)
            if mut_type == "delete tile":
                clone.tiles.remove(clone.random_tiles()[0])

            elif mut_type == "clone tile":
                clone.tiles.append(clone.random_tiles()[0].copy())

            elif mut_type == "add random tile":
                clone.tiles.append(
                    Tile(mag_w=self.mag_w, mag_h=self.mag_h, tile_size=self.tile_size, max_symbol=self.max_symbol))
            else:
                raise (Exception("unhandled mutation type"))

        elif mut_type == "param":
            param_name = np.random.choice(list(self.evolved_params_values))
            mut_param_info = Individual._evolved_params[param_name]
            new_val = Individual.gauss_mutate(self.evolved_params_values[param_name],
                                              strength * (mut_param_info["high"] - mut_param_info["low"]) / 200)
            clone.evolved_params_values[param_name] = new_val
        else:
            raise (Exception("unhandled mutation type"))
        clone.age = 0
        clone.refresh()
        return [clone]

    def random_tiles(self, num=1, replace=False):
        return [self.tiles[i] for i in list(np.random.choice(range(len(self.tiles)), size=num, replace=replace))]

    def crossover(self, other):
        """crossover 2 individuls, return any new children as a list """
        if len(self.tiles) == 1 and len(other.tiles) == 1:
            # if both parents have 1 tile, cross over the angle and positions
            # of the magnets
            parents = [self, other]
            np.random.shuffle(parents)

            child = parents[0].copy(init_pheno=False)
            for i in range(1, len(child.tiles[0])):  # probably just [1]
                angles = (parents[0].tiles[0][i].angle, parents[1].tiles[0][i].angle)
                poss = (parents[0].tiles[0][i].pos, parents[1].tiles[0][i].pos)

                new_angle = np.random.rand() * np.abs(angles[0] - angles[1]) + min(angles)
                new_pos = np.random.rand() * np.abs(poss[0] - poss[1]) + np.min(poss, axis=0)

                rot = new_angle - child.tiles[0][i].angle
                child.tiles[0][i].i_rotate(rot, "centroid")

                translate = new_pos - child.tiles[0][i].pos
                child.tiles[0][i].i_translate(*translate)

            if Magnet.any_intersecting(child.tiles[0]):  # crossover failed
                return []
        else:
            num_tiles = (len(self.tiles), len(other.tiles))
            num_tiles = min(self.max_tiles, np.random.randint(min(num_tiles), max(num_tiles) + 1))

            from_first_parent = np.random.randint(0, num_tiles + 1)
            tiles = self.random_tiles(num=min(from_first_parent, len(self.tiles)))
            tiles += other.random_tiles(num=min(num_tiles - from_first_parent, len(other.tiles)))

            tiles = [tile.copy() for tile in tiles]
            child = self.copy()
            child.tiles = tiles

        child.age = 0
        child.refresh()
        return [child]

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
    def __init__(self, *, mag_w=20, mag_h=50, tile_size=150, max_symbol=1, magnets=None):
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
            self.magnets = [Magnet(np.random.randint(0, max_symbol, 2), np.array((tile_size, tile_size)) / 2, np.pi / 2,
                                   mag_w, mag_h, 0)]
            num_mags = 2  # dangerous to do more
            for _ in range(num_mags - 1):
                # try to place magnet randomly (throw error after many attempts fail)
                # currently only midpoint must be in bounds!
                max_attempts = 50
                for attempts in range(max_attempts + 1):
                    # make random magnet in tile
                    new_mag = Magnet(np.random.randint(0, max_symbol, 2),
                                     np.array(np.random.uniform(high=tile_size, size=2)),
                                     np.random.uniform(low=0, high=2 * np.pi), mag_w, mag_h, 0
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
            tile_symbol_index = magnet_symbol_index if angle_offset else (-1 * magnet_symbol_index - 1)
            if self[origin_index].symbol[tile_symbol_index] != mag.symbol[magnet_symbol_index]:
                continue

            new_tile = self.copy(current_iter)  # copy tile to use as the new magnets to add
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


class Magnet:
    def __init__(self, symbol, pos, angle, mag_w, mag_h, created=None, padding=20):
        self.symbol = symbol
        self.pos = pos
        self.angle = np.mod(angle, 2 * np.pi)
        self.mag_w = mag_w
        self.mag_h = mag_h
        self.created = created
        self.padding = padding

        self.as_polygon, self.bound = self.init_polygon()
        self.locked = False

    def __eq__(self, other):
        if type(other) != Magnet:
            return False

        return (self.symbol == other.symbol and (self.pos == other.pos).all() and
                self.angle == other.angle and self.mag_w == other.mag_w and
                self.mag_h == other.mag_h and self.created == other.created)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return repr({k: v for (k, v) in vars(self).items() if k not in ["as_polygon", "locked", "bound"]})

    def init_polygon(self):
        """return polygon repr of magnet and polygon repr of the padding boundary"""
        half_diagonal = (0.5 * self.mag_w, 0.5 * self.mag_h)
        min_x, min_y = self.pos - half_diagonal
        max_x, max_y = self.pos + half_diagonal
        rect = box(min_x, min_y, max_x, max_y)

        half_pad = self.padding / 2
        bounds = box(min_x - half_pad, min_y - half_pad, max_x + half_pad, max_y + half_pad)
        return rotate(rect, self.angle, use_radians=True), rotate(bounds, self.angle, use_radians=True)

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
            if magnets[i].is_intersecting(magnets[i + 1:len(magnets)]):
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

    @staticmethod
    def rot(v, angle):
        v = deepcopy(v)
        v[0], v[1] = (v[0] * np.cos(angle) - v[1] * np.sin(angle),
                      v[0] * np.sin(angle) + v[1] * np.cos(angle))
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
        mag.as_polygon, mag.bound = mag.init_polygon()  # need to remake the polys with new pos
    return magnets


# ===================  FITNESS EVAL ========================
def evaluate_outer(outer_pop, *, max_age=0, **kwargs):
    for i in outer_pop:
        i.fitness = np.sum(i.fitness_components)
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
        if not np.isfinite(fit):
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
    default_params = {"run": "local", "model": "CustomSpinIce", "encoder": "AngleSine", "H": 0.01, "phi": 90,
                      "radians": True, "alpha": 30272, "sw_b": 0.4, "sw_c": 1, "sw_beta": 3, "sw_gamma": 3, "spp": 100,
                      "hc": 0.03, "periods": 10, "neighbor_distance": 1000}
    if gen is not None:
        outdir = os.path.join(outdir, f"gen{gen}")
    default_params["basepath"] = outdir

    return default_params


def get_default_run_params(pop, condition=lambda i: len(i.pheno) >= i.pheno_size):
    run_params = []
    for indv in [i for i in pop if condition(i)]:
        run_params.append({"indv_id": indv.id,
                           "magnet_coords": [mag.pos for mag in indv.pheno],
                           "magnet_angles": [mag.angle for mag in indv.pheno]})
    return run_params


def flatspin_eval(fit_func, pop, gen, outdir, *, run_params=None, shared_params=None,
                  condition=lambda i: len(i.pheno) >= i.pheno_size, group_by_indv=False, **flatspin_kwargs):
    """
    fit_func is a function that takes a dataset and produces an iterable (or single value) of fitness components.
    if an Individual already has fitness components the lists will be summed element wise
    (allows for multiple datasets per Individual)
    """
    if len(pop) < 1:
        return pop
    if run_params is None:
        run_params = get_default_run_params(pop, condition)
    default_shared = get_default_shared_params(outdir, gen)
    if shared_params is not None:
        default_shared.update(shared_params)
    shared_params = default_shared
    shared_params.update(flatspin_kwargs)

    if len(run_params) > 0:
        id2indv = {individual.id: individual for individual in pop}
        evolved_params = [id2indv[rp["indv_id"]].evolved_params_values for rp in run_params]
        evo_run(run_params, shared_params, gen, evolved_params, wait=group_by_indv)

        datasets = Dataset.read(shared_params["basepath"])
        if not group_by_indv:
            queue = list(datasets)
            while len(queue) > 0:
                ds = queue.pop(0)
                if not os.path.exists(os.path.join(shared_params["basepath"], ds.index["outdir"].iloc[0])):
                    queue.append(ds)  # if file not exist yet add it to the end and check next
                else:
                    with np.errstate(all='ignore'):
                        try:
                            # calculate fitness of a dataset
                            fit_components = fit_func(ds)
                            try:
                                fit_components = list(fit_components)
                            except(TypeError):
                                fit_components = [fit_components]

                            # assign the fitness of the correct individual
                            indv = id2indv[ds.index["indv_id"].values[0]]
                            if indv.fitness_components is not None:
                                indv.fitness_components += fit_components
                            else:
                                indv.fitness_components = fit_components
                        except:  # not done saving file
                            queue.append(ds)
                            sleep(2)
        else:
            group = datasets.groupby("indv_id")
            for id, ds in group:
                try:
                    fit_components = list(fit_components)
                except(TypeError):
                    fit_components = [fit_components]
                indv = id2indv[id]
                if indv.fitness_components is not None:
                    indv.fitness_components += fit_components
                else:
                    indv.fitness_components = fit_components

    for indv in [i for i in pop if not condition(i)]:
        indv.fitness_components = [np.nan]
    return pop


def evo_run(runs_params, shared_params, gen, evolved_params=[], wait=False):
    """ modified from run_sweep.py main()"""
    model_name = shared_params.pop("model", "generated")
    model_class = import_class(model_name, 'flatspin.model')
    encoder_name = shared_params.get("encoder", "Sine")
    encoder_class = import_class(encoder_name, 'flatspin.encoder')

    data_format = shared_params.get("format", "npz")

    params = get_default_params(run)
    params['encoder'] = f'{encoder_class.__module__}.{encoder_class.__name__}'
    params.update(get_default_params(model_class))
    params.update(get_default_params(encoder_class))
    params.update(shared_params)

    info = {
        'model': f'{model_class.__module__}.{model_class.__name__}',
        'model_name': model_name,
        'data_format': data_format,
        'command': ' '.join(map(shlex.quote, sys.argv)),
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
            run_params.update({k: v for k, v in evolved_params[i].items() if k in newparams})
        sub_run_name = newparams["sub_run_name"] if "sub_run_name" in newparams else "x"
        outdir = outdir_tpl.format(gen, newparams["indv_id"]) + f"{sub_run_name}.{ext}"
        filenames.append(outdir)
        row = OrderedDict(run_params)
        row.update({'outdir': outdir})
        index.append(row)

    # Save dataset
    index = pd.DataFrame(index)
    dataset = Dataset(index, params, info, basepath)
    dataset.save()

    # Run!
    # print("Starting sweep with {} runs".format(len(dataset)))
    rs = np.random.get_state()
    run_type = shared_params.get("run", "local")
    if run_type == 'local':
        run_local(dataset, False)

    elif run_type == 'dist':
        run_dist(dataset, wait=wait)

    np.random.set_state(rs)
    return


def flips_fitness(pop, gen, outdir, num_angles=1, other_sizes_fractions=[], **flatspin_kwargs):
    shared_params = get_default_shared_params(outdir, gen)
    shared_params.update(flatspin_kwargs)
    if num_angles > 1:
        shared_params["input"] = [0, 1] * (shared_params["periods"] // 2)

    run_params = get_default_run_params(pop)
    frac_run_params = []
    if len(run_params) > 0:
        for rp in run_params:
            rp["sub_run_name"] = f"frac{1}"
            for frac in other_sizes_fractions:
                angles_frac = rp["magnet_angles"][:int(np.ceil(len(rp["magnet_angles"]) * frac))]
                coords_frac = rp["magnet_coords"][:int(np.ceil(len(rp["magnet_coords"]) * frac))]
                frac_run_params.append({"indv_id": rp["indv_id"],
                                        "magnet_coords": coords_frac,
                                        "magnet_angles": angles_frac,
                                        "sub_run_name": f"frac{frac}"})

    def fit_func(ds):
        # fitness is number of steps, but ignores steps from first fifth of the run
        steps = read_table(ds.tablefile("steps"))
        fitn = steps.iloc[-1]["steps"] - steps.iloc[(shared_params["spp"] * shared_params["periods"]) // 5]["steps"]
        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, shared_params=shared_params,
                        run_params=run_params + frac_run_params, **flatspin_kwargs)
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


def image_match_fitness(pop, gen, outdir, image_file_loc, num_blocks=33, threshold=True, min_mags=50, **flatspin_kwargs):
    img = np.asarray(Image.open(image_file_loc))
    l = []
    step = len(img) / num_blocks
    for y in range(num_blocks):
        row = []
        for x in range(num_blocks):
            a = img[int(x * step):int((x + 1) * step), int(y * step):int((y + 1) * step)]
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

    pop = flatspin_eval(fit_func, pop, gen, outdir, condition=lambda x: len(x.pheno)>=min_mags,**flatspin_kwargs)
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


def pheno_size_fitness(pop, gen, outdir, **flatspin_kwargs):
    id2indv = {individual.id: individual for individual in pop}
    shared_params = {"spp": 1, "periods": 1, "H": 0, "neighbor_distance": 1}

    def fit_func(ds):
        return len(id2indv[ds.index["indv_id"].values[0]].pheno)

    pop = flatspin_eval(fit_func, pop, gen, outdir, condition=lambda x: True, shared_params=shared_params,
                        **flatspin_kwargs)
    return pop


def std_grid_field_fitness(pop, gen, outdir, angles=np.linspace(0, 2 * np.pi, 8), grid_size=4, **flatspin_kwargs):
    shared_params = {}
    shared_params["phi"] = 360
    shared_params["input"] = (angles % (2 * np.pi)) / (2 * np.pi)

    if np.isscalar(grid_size):
        grid_size = (grid_size, grid_size)

        def fit_func(ds):
            mag = load_output(ds, "mag", t=ds.params["spp"], grid_size=grid_size, flatten=False)
            magnitude = np.linalg.norm(mag, axis=3)
            summ = np.sum(magnitude, axis=0)
            fitn = np.std(summ) * np.mean(summ)
            return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, shared_params=shared_params, **flatspin_kwargs)
    return pop


def target_order_percent_fitness(pop, gen, outdir, grid_size=4, **flatspin_kwargs):
    shared_params = {}
    shared_params["encoder"] = "Rotate"
    shared_params["input"] = np.linspace(1, 0, shared_params["periods"])
    if np.isscalar(grid_size):
        grid_size = (grid_size, grid_size)

    for i in pop:
        i.grid = Grid.fixed_grid(np.array([mag.pos for mag in i.pheno]), grid_size)

    # check there are magnets in at least half of grid
    condition = lambda i: (len(np.unique(i.grid._grid_index, axis=0)) >= 0.5 * grid_size[0] * grid_size[1]) and \
                          len(i.pheno) >= i.pheno_size
    id2indv = {individual.id: individual for individual in pop}

    def fit_func(ds):
        mag = load_output(ds, "mag", t=-1, grid_size=grid_size, flatten=False)
        magnitude = np.linalg.norm(mag, axis=3)[0]
        indv = id2indv[ds.index["indv_id"].values[0]]
        cells_with_mags = [(x, y) for x, y in np.unique(indv.grid._grid_index, axis=0)]
        # fitness is std of the magnitudes of the cells minus std of the number of magnets in each cell
        fitn = np.std([magnitude[x][y] for x, y in cells_with_mags]) - \
               np.std([len(indv.grid.point_index([x, y])) for x, y in cells_with_mags])
        return fitn

    pop = flatspin_eval(fit_func, pop, gen, outdir, condition=condition, shared_params=shared_params, **flatspin_kwargs)
    return pop


def main(outdir=r"results\tileTest", inner="flips", outer="default", individual_params={},
         outer_eval_params={}, evolved_params={}, minimize_fitness=True, **kwargs):
    known_fits = {"target_state_num": target_state_num_fitness,
                  "state_num": state_num_fitness,
                  "flips": flips_fitness,
                  "std_grid_field": std_grid_field_fitness,
                  "target_order_percent": target_order_percent_fitness,
                  "default": evaluate_outer,
                  "find_all": evaluate_outer_find_all,
                  "pheno_size": pheno_size_fitness,
                  "image": image_match_fitness}
    inner = known_fits.get(inner, inner)
    outer = known_fits.get(outer, outer)

    return ea.main(outdir, Individual, inner, outer, minimize_fitness, individual_params=individual_params,
                   outer_eval_params=outer_eval_params, evolved_params=evolved_params, **kwargs)


# m = main(outdir=r"results\flatspinTile26",inner=flipsMaxFitness, popSize=3, generationNum=10)
if __name__ == '__main__':
    import argparse
    from flatspin.cmdline import StoreKeyValue, eval_params

    parser = argparse.ArgumentParser(description=__doc__)

    # common
    parser.add_argument('-o', '--output', metavar='FILE',
                        help=r'¯\_(ツ)_/¯')
    parser.add_argument('-p', '--parameter', action=StoreKeyValue, default={},
                        help="param passed to flatspin and inner evaluate fitness function")
    parser.add_argument('-e', '--evolved_params', action=StoreKeyValue, default={},
                        help="param passed to flatspin and inner evaluate that is under evolutionary control, format: [param_name, low, high] or [param_name, low, high, shape*]")
    parser.add_argument('-i', '--individual_param', action=StoreKeyValue, default={},
                        help="param passed to Individual constructor")
    parser.add_argument('-f', '--outer_eval_param', action=StoreKeyValue, default={},
                        help="param past to outer evlauate fitness function")

    args = parser.parse_args()

    main(outdir=args.output, **eval_params(args.parameter), evolved_params=eval_params(args.evolved_params),
         individual_params=eval_params(args.individual_param),
         outer_eval_params=eval_params(args.outer_eval_param))
