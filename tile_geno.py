import numpy as np
import pickle as pkl
from shapely.geometry import box
from shapely.affinity import rotate, translate
from shapely.prepared import prep
from itertools import count
from copy import deepcopy
from collections import Sequence
from time import sleep
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import evo_alg as ea

from flatspin.data import Dataset, read_table, load_output
from flatspin.grid import Grid
from flatspin.cmdline import parse_time
import os
import pandas as pd
# to make animation work in pycharm
import matplotlib
# matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation, writers


class Individual:
    _id_counter = count(0)

    def __init__(self, *, max_tiles=1, tile_size=600, mag_w=220, mag_h=80, initial_rotation=None, max_symbol=1,
                 pheno_size=40, age=0, id=None, fitness=None, fitness_components=None, tiles=None, **kwargs):

        self.id = id if id is not None else next(Individual._id_counter)
        self.max_tiles = max_tiles
        self.tile_size = tile_size
        self.mag_w = mag_w
        self.mag_h = mag_h
        self.age = age
        self.max_symbol = max_symbol
        self.pheno_size = pheno_size

        self.fitness = fitness
        self.fitness_components = fitness_components
        if initial_rotation is not None:
            self.initial_rotation = initial_rotation
        else:
            self.initial_rotation = np.random.uniform(0, 2 * np.pi)

        if mag_h > mag_w:
            raise Warning("conversion to flatspin assumes magnet height < magnet width!")
        if tiles is not None and 1 <= len(tiles):
            if len(tiles) > self.max_tiles:
                raise ValueError("Individual has more tiles than the value of 'max_tiles'")
            self.tiles = tiles
        else:
            self.tiles = [Tile(mag_w=mag_w, mag_h=mag_h, tile_size=tile_size, max_symbol=max_symbol) for _ in
                          range(np.random.randint(1, max_tiles + 1))]
        self.pheno = self.geno2pheno(geom_size=self.pheno_size)

    def refresh(self):
        self.pheno = self.geno2pheno(geom_size=self.pheno_size)
        self.fitness = None
        self.fitness_components = None

    def __repr__(self):
        # defines which attributes can be stored and displayed with repr
        repr_attributes = ("max_tiles", "tile_size", "mag_w", "mag_h", "age", "tiles", "pheno_size",
                           "initial_rotation", "id", "fitness", "fitness_components")
        return repr({k: v for (k, v) in vars(self).items() if k in repr_attributes})

    def copy(self):
        # defines which attributes are used when copying
        copy_attributes = ("max_tiles", "tile_size", "mag_w", "mag_h", "age", "tiles", "initial_rotation",
                           "fitness", "fitness_components", "pheno_size")
        new_indv = Individual(**{k: v for (k, v) in vars(self).items() if k in copy_attributes})
        # copy attributes that are referenced to unlink
        new_indv.tiles = [Tile(magnets=[mag.copy() for mag in tile]) for tile in new_indv.tiles]

        return new_indv

    @staticmethod
    def from_string(s):
        array = np.array
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
        frontier[0].i_rotate(self.initial_rotation, 'centroid')
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
                        if not new_mag.is_intersecting(frontier + frozen + new_front):
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

        return centre_magnets(frozen + frontier)[:geom_size]

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

    def mutate(self, strength=5):
        clone = self.copy()
        mut_types = ["magPos", "magAngle", "symbol", "tile"]
        if self.initial_rotation is not None:
            mut_types.append("initRot")
        if self.max_tiles == 1:
            mut_types.remove("tile")  # cannot add/remove tiles when max tile is 1
        mut_type = np.random.choice(mut_types)

        if mut_type == "magPos":
            # pick a tile at random, pick a magnet excluding the first magnet
            # and move it (only position of magnet center is confined by tile_size)
            tile = clone.tiles[np.random.randint(0, len(clone.tiles))]
            x = 1 + np.random.randint(len(tile[1:]))
            copy_mag = tile[x].copy()
            distance = Individual.gauss_mutate(copy_mag.pos, 5 * strength, 0, clone.tile_size) - copy_mag.pos
            copy_mag.i_translate(*distance)

            if not copy_mag.is_intersecting(tile[:x] + tile[x + 1:]):
                # only mutate if does not cause overlap
                tile.locked = False
                tile[x] = copy_mag
                tile.locked = True

        elif mut_type == "magAngle":
            # pick a tile at random, pick a magnet excluding the first magnet and rotate it (about centroid)
            tile = clone.tiles[np.random.randint(0, len(clone.tiles))]
            x = 1 + np.random.randint(len(tile[1:]))
            copy_mag = tile[x].copy()
            rotation = np.random.normal(0, 5 * strength)
            copy_mag.i_rotate(rotation, "centroid")

            if not copy_mag.is_intersecting(tile[:x] + tile[x + 1:]):
                # only mutate if does not cause overlap
                tile.locked = False
                tile[x] = copy_mag
                tile.locked = True

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
        elif mut_type == "initRot":
            clone.initial_rotation = Individual.gauss_mutate(self.initial_rotation, 10) % (2 * np.pi)
        else:
            raise (Exception("unhandled mutation type"))

        clone.age = 0
        clone.refresh()
        return clone

    def random_tiles(self, num=1, replace=False):
        return [self.tiles[i] for i in list(np.random.choice(range(len(self.tiles)), size=num, replace=replace))]

    def crossover(self, other):
        if len(self.tiles) == 1 and len(other.tiles) == 1:
            # if both parents have 1 tile, cross over the angle and positions
            # of the magnets
            parents = [self, other]
            np.random.shuffle(parents)

            child = parents[0].copy()
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
                return None
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
        return child

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
        for counter, angleOffset in zip([0, -1], [0, np.pi]):  # do twice for the 180 degree offset

            # check symbol of origin magnet matches symbol of mag (in current rotation)
            if self[origin_index].angle > np.pi != mag.angle > np.pi:
                # if both have angle within the same half disc (0<np.pi or np.pi<2*np.pi)  use corresponding symbols
                if self[origin_index].symbol[counter] != mag.symbol[counter]:
                    continue
            else:
                # else use opposite symbols i.e 0 and -1 or -1 and 0
                if self[origin_index].symbol[counter] != mag.symbol[-counter - 1]:
                    continue

            new_tile = self.copy(current_iter)  # copy tile to use as the new magnets to add
            angle_diff = mag.angle - new_tile[origin_index].angle + angleOffset
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


def flips_fitness(pop, gen, outdir, num_angles=1, other_sizes_fractions=[], **kwargs):
    if len(pop) < 1:
        return pop
    shared_params = get_default_shared_params(outdir, gen)
    shared_params.update(kwargs)

    if num_angles > 1:
        shared_params["input"] = [0, 1] * (shared_params["periods"] // 2)

    run_params = get_default_run_params(pop)
    frac_run_params = []
    if len(run_params) > 0:
        for rp in run_params:

            for frac in other_sizes_fractions:
                angles_frac = rp["magnet_angles"][:int(np.ceil(len(rp["magnet_angles"]) * frac))]
                coords_frac = rp["magnet_coords"][:int(np.ceil(len(rp["magnet_coords"]) * frac))]
                frac_run_params.append({{"indv_id": rp["indv_id"],
                                         "magnet_coords": coords_frac,
                                         "magnet_angles": angles_frac}})
        ea.evo_run(run_params + frac_run_params, shared_params, gen)  # run full

        id2indv = {individual.id: individual for individual in pop}
        for indv in [i for i in pop if len(i.pheno) >= i.pheno_size]:
            indv.fitness_components = [0]
        queue = list(Dataset.read(shared_params["basepath"]))
        while len(queue) > 0:
            ds = queue.pop(0)
            if not os.path.exists(os.path.join(shared_params["basepath"], ds.index["outdir"].iloc[0])):
                queue.append(ds)  # if file not exist yet add it to the end and check next
            else:
                try:
                    steps = read_table(ds.tablefile("steps"))
                    # fitness is number of steps, but ignores steps from first fifth of the run
                    fitn = steps.iloc[-1]["steps"] - steps.iloc[(shared_params["spp"] * shared_params["periods"]) // 5][
                        "steps"]
                    id2indv[ds.index["indv_id"].values[0]].fitness_components[0] += fitn
                except:  # not done saving file
                    queue.append(ds)
    for indv in [i for i in pop if len(i.pheno) < i.pheno_size]:
        indv.fitness_components = [np.nan]
    # for i in pop:
    #   print("fit comp :" + str(i.fitness_components))
    return pop


def target_state_num_fitness(pop, gen, outdir, target, state_step=None, **kwargs):
    if len(pop) < 1:
        return pop
    shared_params = get_default_shared_params(outdir, gen)
    shared_params.update(kwargs)
    run_params = get_default_run_params(pop)
    if state_step is None:
        state_step = shared_params["spp"]
    if len(run_params) > 0:
        ea.evo_run(run_params, shared_params, gen)
        id2indv = {individual.id: individual for individual in pop}

        queue = list(Dataset.read(shared_params["basepath"]))
        while len(queue) > 0:
            ds = queue.pop(0)
            if not os.path.exists(os.path.join(shared_params["basepath"], ds.index["outdir"].iloc[0])):
                queue.append(ds)  # if file not exist yet add it to the end and check next
            else:
                try:
                    spin = read_table(ds.tablefile("spin"))
                    fitn = abs(len(np.unique(spin.iloc[::state_step, 1:], axis=0)) - target)
                    id2indv[ds.index["indv_id"].values[0]].fitness_components = [fitn, ]
                except:  # not done saving file
                    queue.append(ds)

    for indv in [i for i in pop if len(i.pheno) < i.pheno_size]:
        indv.fitness_components = [np.nan]
    return pop


def state_num_fitness(pop, gen, outdir, state_step=None, **kwargs):
    if len(pop) < 1:
        return pop
    shared_params = get_default_shared_params(outdir, gen)
    shared_params.update(kwargs)
    run_params = get_default_run_params(pop)
    if state_step is None:
        state_step = shared_params["spp"]
    if len(run_params) > 0:
        ea.evo_run(run_params, shared_params, gen)
        id2indv = {individual.id: individual for individual in pop}

        queue = list(Dataset.read(shared_params["basepath"]))
        while len(queue) > 0:
            ds = queue.pop(0)
            if not os.path.exists(os.path.join(shared_params["basepath"], ds.index["outdir"].iloc[0])):
                queue.append(ds)  # if file not exist yet add it to the end and check next
            else:
                try:
                    spin = read_table(ds.tablefile("spin"))
                    fitn = len(np.unique(spin.iloc[::state_step, 1:], axis=0))
                    id2indv[ds.index["indv_id"].values[0]].fitness_components = [fitn, ]
                except:  # not done saving file
                    queue.append(ds)

    for indv in [i for i in pop if len(i.pheno) < i.pheno_size]:
        indv.fitness_components = [np.nan]
    return pop


def std_grid_field_fitness(pop, gen, outdir, angles=np.linspace(0, 2 * np.pi, 8), grid_size=4, **kwargs):
    if len(pop) < 1:
        return pop
    shared_params = get_default_shared_params(outdir, gen)
    shared_params.update(kwargs)

    shared_params["phi"] = 360
    shared_params["input"] = (angles % (2 * np.pi)) / (2 * np.pi)
    t = parse_time(f"::{shared_params['spp']}")
    if np.isscalar(grid_size):
        grid_size = (grid_size, grid_size)

    run_params = get_default_run_params(pop)
    if len(run_params) > 0:
        ea.evo_run(run_params, shared_params, gen)
        id2indv = {individual.id: individual for individual in pop}

        queue = list(Dataset.read(shared_params["basepath"]))
        while len(queue) > 0:
            ds = queue.pop(0)
            if not os.path.exists(os.path.join(shared_params["basepath"], ds.index["outdir"].iloc[0])):
                queue.append(ds)  # if file not exist yet add it to the end and check next
            else:
                try:
                    mag = load_output(ds, "mag", t=t, grid_size=grid_size, flatten=False)
                    magnitude = np.linalg.norm(mag, axis=3)
                    summ = np.sum(magnitude, axis=0)
                    fitn = np.std(summ) * np.mean(summ)
                    id2indv[ds.index["indv_id"].values[0]].fitness_components = [fitn, ]
                except:  # not done saving file
                    queue.append(ds)

    for indv in [i for i in pop if len(i.pheno) < i.pheno_size]:
        indv.fitness_components = [np.nan]
    return pop


def target_order_percent_fitness(pop, gen, outdir, grid_size=4, **kwargs):
    if len(pop) < 1:
        return pop
    shared_params = get_default_shared_params(outdir, gen)
    shared_params.update(kwargs)
    shared_params["encoder"] = "Rotate"
    shared_params["input"] = np.linspace(1, 0, shared_params["periods"])
    if np.isscalar(grid_size):
        grid_size = (grid_size, grid_size)

    for i in pop:
        i.grid = Grid.fixed_grid(np.array([mag.pos for mag in i.pheno]), grid_size)

    # check there are magnets in at least half of grid
    condition = lambda i: (len(np.unique(i.grid._grid_index, axis=0)) >= 0.5 * grid_size[0] * grid_size[1]) and len(
        i.pheno) >= i.pheno_size
    run_params = get_default_run_params(pop, condition)
    if len(run_params) > 0:
        ea.evo_run(run_params, shared_params, gen)
        id2indv = {individual.id: individual for individual in pop}

        queue = list(Dataset.read(shared_params["basepath"]))
        while len(queue) > 0:
            ds = queue.pop(0)
            if not os.path.exists(os.path.join(shared_params["basepath"], ds.index["outdir"].iloc[0])):
                queue.append(ds)  # if file not exist yet add it to the end and check next
                sleep(1)
            else:
                try:
                    mag = load_output(ds, "mag", t=-1, grid_size=grid_size, flatten=False)
                except:  # not done saving file
                    queue.append(ds)
                    sleep(1)
                    continue

                magnitude = np.linalg.norm(mag, axis=3)[0]
                indv = id2indv[ds.index["indv_id"].values[0]]
                cells_with_mags = [(x, y) for x, y in np.unique(indv.grid._grid_index, axis=0)]
                # fitness is std of the magnitudes of the cells minus std of the number of magnets in each cell
                fitn = np.std([magnitude[x][y] for x, y in cells_with_mags]) - \
                       np.std([len(indv.grid.point_index([x, y])) for x, y in cells_with_mags])
                indv.fitness_components = [fitn, ]

    for indv in [i for i in pop if not condition(i)]:
        indv.fitness_components = [np.nan]
    return pop


def main(outdir=r"results\tileTest", inner="flips", outer="default", individual_params={},
         outer_eval_params={},
         minimize_fitness=True,
         **kwargs):
    known_fits = {"target_state_num": target_state_num_fitness,
                  "state_num": state_num_fitness,
                  "flips": flips_fitness,
                  "std_grid_field": std_grid_field_fitness,
                  "target_order_percent": target_order_percent_fitness}
    known_outer = {"default": evaluate_outer,
                   "find_all": evaluate_outer_find_all}
    if inner in known_fits:
        inner = known_fits[inner]

    if outer in known_outer:
        outer = known_outer[outer]

    return ea.main(outdir, Individual, inner, outer, minimize_fitness, individual_params=individual_params,
                   outer_eval_params=outer_eval_params, **kwargs)


# m = main(outdir=r"results\flatspinTile26",inner=flipsMaxFitness, popSize=3, generationNum=10)
if __name__ == '__main__':
    import argparse
    from flatspin.cmdline import StoreKeyValue, eval_params

    parser = argparse.ArgumentParser(description=__doc__)

    # common
    parser.add_argument('-o', '--output', metavar='FILE',
                        help=r'¯\_(ツ)_/¯')
    parser.add_argument('-p', '--parameter', action=StoreKeyValue, default={})
    parser.add_argument('-i', '--individual_param', action=StoreKeyValue, default={})
    parser.add_argument('-e', '--outer_eval_param', action=StoreKeyValue, default={})
    args = parser.parse_args()
    main(outdir=args.output, **eval_params(args.parameter), individual_params=eval_params(args.individual_param),
         outer_eval_params=eval_params(args.outer_eval_param))
