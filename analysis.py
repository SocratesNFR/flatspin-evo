import networkx as nx
from flatspin.data import read_csv
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import grandalf as grand
from grandalf.layouts import SugiyamaLayout
from functools import lru_cache

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
        log = f.readlines()
    index = read_csv(indexfile)

    for line in log:

def parse_log_line(log):
    status = "failed" if log.startswith("INFO:root:Failed") else "succeeded"

    return status, action, parnets, info

def fitness_diversity(indexfile, show_max=False):
     index = read_csv(indexfile)
     fig, ax = plt.subplots(3 if show_max else 2,1)
     index.groupby("gen").agg({"fitness": "nunique"}).plot(ax=ax[0], title="unique")
     index.groupby("gen").agg({"fitness": "std"}).plot(ax=ax[1], title="std")
     if show_max:
         index.groupby("gen").agg({"fitness": "max"}).plot(ax=ax[2], title="max")
     plt.show()

@lru_cache
def get_fitness(indv_id, index):
    """gets fitness value for *first appearance* of indv_id"""
    return index[index["indv_id"]==indv_id].iloc[0]["fitness"]
