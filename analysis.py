import networkx as nx
from flatspin.data import read_csv
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import grandalf as grand
from grandalf.layouts import SugiyamaLayout
from functools import lru_cache
import numpy as np


def family_tree(ind_id, logfile, max_depth=float("inf"), **kwargs):
    with open(logfile, "r") as f:
        log = f.read()
    g = build_family_net(ind_id, log, max_depth=max_depth, g=None)

    # pos = graphviz_layout(g, prog='dot')
    # nx.draw(g, pos, with_labels=False, arrows=False)
    plot_tree(g, **kwargs)


def build_family_net(ind_id, log, max_depth=float("inf"), g=None):
    g = g if g else nx.DiGraph()
    if max_depth == 0:
        return g
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


def plot_tree(g, width=10, height=10, with_labels=True, **kwargs):
    grn = grand.utils.convert_nextworkx_graph_to_grandalf(g)  # undocumented function

    class defaultview(object):
        w, h = width, height

    for v in grn.C[0].sV:
        v.view = defaultview()

    sug = SugiyamaLayout(grn.C[0])
    sug.init_all()  # roots=[V[0]])
    sug.draw()  # This is a bit of a misnomer, as grandalf doesn't actually come with any visualization methods. This method instead calculates positions

    poses = {v.data: (v.view.xy[0], v.view.xy[1]) for v in grn.C[0].sV}  # Extracts the positions
    nx.draw(g, pos=poses, with_labels=with_labels, **kwargs)
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
        if not line.startswith(prefix):
            continue
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


def mutation_pie(logfile, indexfile, **kwargs):
    beneficial, benign, malignant, fail = mutation_stats(logfile, indexfile)
    i = 1
    plt.figure(**kwargs)
    categories = [beneficial, benign, malignant, fail]
    cat_names = ["beneficial", "benign", "malignant", "fail"]
    mut_names = set([k for cat in categories for k in cat])
    cat_map = dict(zip(cat_names, categories))
    mut_totals = {mut_name: sum([cat_map[cat][mut_name] for cat in cat_names if mut_name in cat_map[cat]]) for mut_name
                  in mut_names}
    for data, title in zip(categories, cat_names):
        plt.subplot(4, 2, i)
        keys, vals = zip(*data.items())
        plt.pie(vals, labels=keys)
        plt.title(title)
        i += 1
        # normalised
        plt.subplot(4, 2, i)
        vals = [round(vals[i] / mut_totals[keys[i]], 3) for i in range(len(vals))]
        plt.pie(vals, labels=[f"{keys[i]}:{vals[i]}" for i in range(len(vals))])
        plt.title(f"norm {title}")
        i += 1
    plt.figure(**kwargs)

    cols = 3
    rows = len(mut_names) // cols + (len(mut_names) % cols)
    i = 1

    mut_data = [{cat_name: cat_map[cat_name][mut_name] for cat_name in cat_map if mut_name in cat_map[cat_name]} for
                mut_name in mut_names]
    for data, title in zip(mut_data, mut_names):
        plt.subplot(rows, cols, i)
        keys, vals = zip(*data.items())

        keys_with_total = [key + f":{data[key]}" for key in keys]
        plt.pie(vals, labels=keys_with_total, )
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


def fitness_diversity(indexfile, agg=None, **kwargs):
    index = read_csv(indexfile)
    fig, ax = plt.subplots(3 if agg else 2, 1)
    index.groupby("gen").agg({"fitness": "nunique"}).plot(ax=ax[0], title="#unique", **kwargs)
    index.groupby("gen").agg({"fitness": "std"}).plot(ax=ax[1], title="std", **kwargs)
    if agg:
        index.groupby("gen").agg({"fitness": agg}).plot(ax=ax[2], title=agg, **kwargs)
    plt.show()


def get_fitness(indv_id, index):
    """gets fitness value for *first appearance* of indv_id"""
    indv = index[index["indv_id"] == indv_id]
    return indv.iloc[0]["fitness"] if len(indv) > 0 else "not found"


def plot_scatter(indexfile, **kwargs):
    plt.style.use("dark_background")
    index = read_csv(indexfile)
    best = index[index["best"] == 1]
    not_best = index[index["best"] == 0]
    v_min, v_max = index["fitness"].min(), index["fitness"].max()
    plt.scatter(y=not_best["indv_id"], x=not_best["gen"], c=not_best["fitness"], cmap="rainbow", alpha=0.5, vmin=v_min,
                vmax=v_max, **kwargs)
    plt.scatter(y=best["indv_id"], x=best["gen"], c=best["fitness"], marker="*", cmap="rainbow", alpha=0.5, vmin=v_min,
                vmax=v_max, **kwargs)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    import argparse
    import os
    from flatspin.cmdline import StoreKeyValue, eval_params

    parser = argparse.ArgumentParser(description=__doc__)

    # common
    parser.add_argument('action', metavar="action", choices=["mut-pie", "family-tree", "diversity", "scatter"],
                        help="[mut-pie/family-tree/diversity]")
    parser.add_argument('-l', '--log', metavar='FILE', default="evo.log",
                        help=r'name of log')
    parser.add_argument('-b', '--basepath', metavar='FILE', default="",
                        help=r'location of log and index')

    parser.add_argument('-p', '--param', action=StoreKeyValue, default={},
                        help="keyword params")

    args = parser.parse_args()

    log = os.path.join(args.basepath, args.log)
    index = os.path.join(args.basepath, "index.csv")

    if args.action == "mut-pie":
        mutation_pie(log, index, **args.param)
    elif args.action == "diversity":
        fitness_diversity(index, **args.param)
    elif args.action == "family-tree":
        if "ind_id" not in args.param:
            raise Exception("parameter ind_id must be supplied for action 'family-tree'")
        ind_id = args.param.pop("ind_id")
        family_tree(ind_id, log, **args.param)
    elif args.action == "scatter":
        plot_scatter(index, **args.param)
    else:
        raise Exception(f"Unknown action: '{args.action}'")
