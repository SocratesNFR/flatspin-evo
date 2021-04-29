import networkx as nx
from flatspin.data import read_csv
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import grandalf as grand
from grandalf.layouts import SugiyamaLayout
from functools import lru_cache
import numpy as np


def family_tree(ind_id, logfile, max_depth=float("inf")):
    with open(logfile, "r") as f:
        log = f.read()
    g = build_family_net(ind_id, log, max_depth=max_depth, g=None)

    # pos = graphviz_layout(g, prog='dot')
    # nx.draw(g, pos, with_labels=False, arrows=False)
    plot_tree(g)


def build_family_net(ind_id, log, max_depth=float("inf"), g=None):
    g = g if g else nx.DiGraph()
    if max_depth == 0: return g
    g.add_node(ind_id)
    parents = get_parents(ind_id, log)  # TODO: speed up only search for parents above the current in the log
    for parent in parents:
        if parent not in g:
            build_family_net(parent, log, max_depth - 1, g=g)
        g.add_edge(parent, ind_id)

    return g


def get_parents(ind_id, log):
    start = log.rfind(f"ind {ind_id} created")
    if start < 0:
        return []
    end = log.find("\n", start)
    line = log[start:end]
    if "mutate" in line:
        start_descrim = "on ind"
        start = line.find(start_descrim) + len(start_descrim)
        end = line.find("info", start)
        parents = [int(line[start:end])]
    else:
        start_descrim = "with parents ["
        start = line.find(start_descrim) + len(start_descrim)
        end = line.find("]", start)
        parents = [int(p) for p in line[start:end].split(",")]
    return parents


def plot_tree(g):
    grn = grand.utils.convert_nextworkx_graph_to_grandalf(g)  # undocumented function

    class defaultview(object):
        w, h = 10, 10

    for v in grn.C[0].sV: v.view = defaultview()

    sug = SugiyamaLayout(grn.C[0])
    sug.init_all()  # roots=[V[0]])
    sug.draw()  # This is a bit of a misnomer, as grandalf doesn't actually come with any visualization methods. This method instead calculates positions

    poses = {v.data: (v.view.xy[0], v.view.xy[1]) for v in grn.C[0].sV}  # Extracts the positions
    nx.draw(g, pos=poses, with_labels=True)
    plt.show()


def mutation_stats(logfile, indexfile):
    with open(logfile, "r") as f:
        log = f.read()
    index = read_csv(indexfile)
    beneficial = {}
    benign = {}
    malignant = {}
    fail = {}
    prefix = "INFO"

    _get_fit = lru_cache()(lambda id: get_fitness(id, index))

    for line in log.splitlines():
        if not line.startswith(prefix): continue
        failed, ind, action, parents, info = parse_log_line(line)
        if failed:
            if action in fail:
                fail[action] += 1
            else:
                fail[action] = 1
        else:
            child_fit = _get_fit(ind)
            if child_fit == "not found":
                continue
            # print(failed, ind, action, parents, info)
            parent_fit = np.mean([_get_fit(p) for p in parents])

            category = beneficial if child_fit > parent_fit else (benign if child_fit == parent_fit else malignant)
            if action in category:
                category[action] += 1
            else:
                category[action] = 1
    return beneficial, benign, malignant, fail


def mutation_pie(logfile, indexfile):
    beneficial, benign, malignant, fail = mutation_stats(logfile, indexfile)
    i = 1
    plt.figure()
    categories = [beneficial, benign, malignant, fail]
    cat_names = ["beneficial", "benign", "malignant", "fail"]
    for data, title in zip(categories, cat_names):
        plt.subplot(4, 1, i)
        keys, vals = zip(*data.items())
        plt.pie(vals, labels=keys)
        plt.title(title)
        i += 1
    plt.figure()
    cat_map = dict(zip(cat_names, categories))
    mut_names = set([k for cat in categories for k in cat])
    cols = 3
    rows = len(mut_names) // cols + (len(mut_names) % cols)
    i = 1

    mut_data = [{cat_name: cat_map[cat_name][mut_name] for cat_name in cat_map if mut_name in cat_map[cat_name]} for
                mut_name in mut_names]
    for data, title in zip(mut_data, mut_names):
        plt.subplot(rows, cols, i)
        keys, vals = zip(*data.items())

        keys_with_total = [key + f"{data[key]}" for key in keys]
        plt.pie(vals, labels=keys_with_total,)
        plt.title(title, wrap=True, fontsize=8)
        i += 1

    plt.show()


def absolute_value(val):
    a = np.round(val / 100. * sizes.sum(), 0)
    return a


def parse_log_line(line):
    log_prefix = "INFO:root:"
    line = line[len(log_prefix):]
    failed = line.startswith("Failed")
    ind = info = None

    try:
        if "mutate" in line:
            if failed:
                pref = "Failed mutation: "
                action, cdr = line[len(pref):].split(" on ind ")
                parents, info = cdr.split(" info: ")
            else:
                pref = "ind"
                ind, cdr = line[len(pref):].split(" created from ")
                action, cdr = cdr.split(" on ind ")
                parents, info = cdr.split(" info: ")

            parents = [int(parents)]
        else:
            if failed:
                action, parents = line[line.find(":") + 1:line.find("]")].strip().split(" with parents [")
            else:
                pref = "ind"
                ind, cdr = line[len(pref):].split(" created from ")
                action, parents = cdr[:-1].split(" with parents [")

            parents = [int(p) for p in parents.split(",")]

        ind = int(ind) if ind else None
    except Exception as e:
        print(f"failed on line:\n{line}")
        raise e

    return failed, ind, action, parents, info


def fitness_diversity(indexfile, show_max=False):
    index = read_csv(indexfile)
    fig, ax = plt.subplots(3 if show_max else 2, 1)
    index.groupby("gen").agg({"fitness": "nunique"}).plot(ax=ax[0], title="unique")
    index.groupby("gen").agg({"fitness": "std"}).plot(ax=ax[1], title="std")
    if show_max:
        index.groupby("gen").agg({"fitness": "max"}).plot(ax=ax[2], title="max")
    plt.show()


def get_fitness(indv_id, index):
    """gets fitness value for *first appearance* of indv_id"""
    indv = index[index["indv_id"] == indv_id]
    return indv.iloc[0]["fitness"] if len(indv) > 0 else "not found"
