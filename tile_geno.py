import numpy as np

from shapely.geometry import box
from shapely.affinity import rotate, translate
from shapely.prepared import prep
from itertools import count
from copy import deepcopy
from collections import Sequence

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import evo_alg as ea

from flatspin.data import Dataset, read_table

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

    def __init__(self, *, max_tiles=1, tile_size=440, mag_w=220, mag_h=80, initial_rotation=None, max_symbol=1,
                 pheno_size=40, age=0,
                 tiles=None, **kwargs):

        self.id = next(Individual._id_counter)
        self.max_tiles = max_tiles
        self.tile_size = tile_size
        self.mag_w = mag_w
        self.mag_h = mag_h
        self.age = age
        self.max_symbol = max_symbol
        self.pheno_size = pheno_size
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

        self.fitness = None
        self.fitness_components = None

    def refresh(self):
        self.pheno = self.geno2pheno(geom_size=self.pheno_size)
        self.fitness = None
        self.fitness_components = None

    def __repr__(self):
        # defines which attributes can be stored and displayed with repr
        repr_attributes = ("max_tiles", "tile_size", "mag_w", "mag_h", "age", "tiles")
        return repr({k: v for (k, v) in vars(self).items() if k in repr_attributes})

    def copy(self):
        # defines which attributes are used when copying
        copy_attributes = ("max_tiles", "tile_size", "mag_w", "mag_h", "age", "tiles")
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
    def frames2animation(frames, interval=400, title=False):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        bounds = np.vstack(list(map(lambda m: m.xy, [mag for frame in frames for mag in frame])))
        xlim = bounds[:, 0].min(), bounds[:, 0].max()
        ylim = bounds[:, 1].min(), bounds[:, 1].max()

        def step(i, maxi, xlim, ylim, title):
            ax.cla()
            for poly in frames[i]:
                ax.add_patch(poly)
            if title:
                ax.set_title(f"i = {i}")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            fig.canvas.draw()

        colours = ea.rainbow_colours(len(frames))
        for frame in frames:
            for poly in frame:
                poly.set_color(colours[poly.iterCreated % len(colours)])
        ax.set_facecolor('k')
        return FuncAnimation(fig, step, frames=len(frames),
                             fargs=(len(frames) - 1, xlim, ylim, title),
                             blit=False, interval=interval)

    @staticmethod
    def print_mags(mags):
        plt.figure()
        for mag in mags:
            plt.gca().add_patch(mag.as_patch())
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
        elif mut_type=="initRot":
            self.initial_rotation = Individual.gauss_mutate(self.initial_rotation, 10) % (2*np.pi)
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

    def plot(self):
        for mag in self.pheno:
            plt.gca().add_patch(mag.as_patch())
        plt.gca().set_aspect(1)
        plt.autoscale()


class Tile(Sequence):
    def __init__(self, *, mag_w=20, mag_h=50, tile_size=100, max_symbol=1, magnets=None):
        """
        make new random tile from scratch
        """
        # if magnets provided just use those; else randomly init tile
        self.locked = False
        if magnets is not None and len(magnets) > 1:
            assert type(magnets) == list
            self.magnets = magnets
        else:
            # always a magnet at the origin
            self.magnets = [Magnet(np.random.randint(0, max_symbol, 2), np.array((mag_w, mag_h)) / 2, np.pi / 2,
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
        return Magnet(**v)


# ===================  FITNESS EVAL ========================
# def evaluateInner(inner_pop):
#    target = makeSquareAsi(220,80,5,4,4,5,370)
#    for indv in inner_pop:
#        indv.fitness_components = similarityMeasure(indv, target,aggregate=False)
#    
#    return inner_pop
def evaluate_inner(inner_pop):
    for indv in inner_pop:
        indv.fitness_components = [-regularity_measure(indv)]  # / len(indv.pheno)]

    return inner_pop


def evaluate_outer(outer_pop, global_best=None, global_worst=None, max_age=0):
    for i in outer_pop:
        i.fitness = np.sum(i.fitness_components)
    return outer_pop


def scale_to_unit(x, upper, lower):
    return (x - lower) / (upper - lower)


def regularity_measure(indv, target_distance="even", target_angle=np.pi / 2, num_neighbors=4, remove_edge_effect=0.5):
    """
    target distance: either 'even' a float/int or a len 2 list/tuple
    target angle: should be modulo np.pi
    """
    if len(indv.pheno) < num_neighbors or len(indv.pheno) < indv.pheno_size:
        return 99999999999999
    angle_badness = []
    pos_badness = []
    mag_poss = [mag.pos for mag in indv.pheno]
    mag_angles = [mag.angle for mag in indv.pheno]

    tree = KDTree(mag_poss)

    if type(target_distance) not in [list, tuple, str]:
        target_distance = (target_distance,) * 2

    for i in range(len(indv.pheno)):
        _, neighbors = tree.query(mag_poss[i], num_neighbors)
        angle_badness.append(sum([abs(abs(mag_angles[i] % np.pi - mag_angles[n] % np.pi) % np.pi - target_angle)
                                  for n in neighbors]))
        if type(target_distance) is str and target_distance == 'even':
            target_distance = np.mean([np.abs(np.subtract(mag_poss[i], mag_poss[n]))
                                       for n in neighbors], 0)

        pos_badness.append(np.sum([np.abs(np.subtract(np.abs(np.subtract(mag_poss[i], mag_poss[n])), target_distance))
                                   for n in neighbors]))
    angle_badness = np.sort(angle_badness)[:int(len(angle_badness) * remove_edge_effect)].sum()
    pos_badness = np.sort(pos_badness)[:int(len(pos_badness) * remove_edge_effect)].sum()
    return angle_badness  # + posBadness


def make_square_asi(mag_h, mag_w, n_h_rows, n_h_cols, n_v_rows, n_v_cols,
                    lattice_space, centre=(0, 0)):
    # returns [horiz, vert] = [[[x1,y1,w1,h1],[x2,y2,w2,h2],...],[[x1,y1,w1,h1],[x2,y2,w2,h2],...]]

    # horiz mask
    h_mask = []
    for row in range(n_h_rows):
        for col in range(n_h_cols):
            x = (col + 0.5) * lattice_space
            y = (row + 0.5) * lattice_space

            h_mask.append(Magnet(None, np.array([x, y], dtype=np.float64), angle=np.pi / 2, mag_h=mag_h, mag_w=mag_w))

    # vert mask
    v_mask = []
    for row in range(n_v_rows):
        for col in range(n_v_cols):
            x = col * lattice_space
            y = (row + 1) * lattice_space

            v_mask.append(Magnet(None, np.array([x, y], dtype=np.float64), angle=0, mag_h=mag_h, mag_w=mag_w))
    mask = h_mask + v_mask
    if centre is not None:  # translate array to be centred on centre
        mask = centre_magnets(mask, centre)
    return mask


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

    # =============================================================================


def flips_max_fitness(pop, gen, outdir, run="local", num_angles=1, H=0.1, use_cuda=False, **kwargs):
    if len(pop) < 1:
        return pop
    shared_params = {"run": run, "model": "CustomSpinIce", "encoder": "angle-sin", "H": H, "phi": 90, "radians": True,
                     "periods": 10, "use_cuda": use_cuda, "basepath": os.path.join(outdir, f"gen{gen}")}
    if num_angles > 1:
        shared_params["input"] = [0, 1] * 5
    run_params = []
    for indv in pop:
        run_params.append({"indv_id": indv.id,
                           "magnet_coords": [mag.pos for mag in indv.pheno],
                           "magnet_angles": [mag.angle for mag in indv.pheno]})
    ea.evo_run(run_params, shared_params, gen)
    id2indv = {individual.id: individual for individual in pop}

    queue = list(Dataset.read(shared_params["basepath"]))
    while len(queue) > 0:
        ds = queue.pop(0)
        try:  # try to read file, if not there yet add to end of queue
            steps = read_table(ds.tablefile("steps"))
        except FileNotFoundError:
            queue.append(ds)
            continue
        id2indv[ds.index["indv_id"].values[0]].fitness_components = [steps["steps"].iloc[-1], ]
    for i in pop:
        print("fit comp :")
        print(i.fitness_components)
    return pop


def min_flips_fitness(pop, gen, outdir, run="local", **kwargs):
    pop = flips_max_fitness(pop, gen, outdir, run, **kwargs)
    for i in pop:
        if len(i.pheno) < i.pheno_size:
            i.fitness_components = [-666 for x in i.fitness_components]
        else:
            i.fitness_components = [-x for x in i.fitness_components]
    return pop


def main(outdir=r"results\tileTest", inner=min_flips_fitness, **kwargs):
    return ea.main(outdir, Individual, inner, evaluate_outer, **kwargs)


# m = main(outdir=r"results\flatspinTile26",inner=flipsMaxFitness, popSize=3, generationNum=10)
if __name__ == '__main__':
    import argparse
    from flatspin.cmdline import StoreKeyValue, eval_params

    parser = argparse.ArgumentParser(description=__doc__)

    # common
    parser.add_argument('-o', '--output', metavar='FILE',
                        help=r'¯\_(ツ)_/¯')
    parser.add_argument('-p', '--parameter', action=StoreKeyValue, default={})
    args = parser.parse_args()
    main(outdir=args.output, **eval_params(args.parameter))
