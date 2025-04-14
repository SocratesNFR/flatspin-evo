# %%
import numpy as np
import re
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import pandas as pd
import os

def parse_slice(s):
    """Parse a slice string like ':', '5', '3:', '-5:-1' into a slice object."""
    if s == ":":
        return slice(None, None)
    if ":" not in s:
        return slice(int(s), int(s) + 1)
    parts = s.split(":")
    parts = [int(p) if p else None for p in parts]
    return slice(*parts)

def resolve_gen_range(df, gen_slice):
    gen_min = df["gen"].min()
    gen_max = df["gen"].max() + 1  # inclusive range

    def resolve(val):
        if val is None:
            return None
        return gen_max + val if val < 0 else val

    start = resolve(gen_slice.start)
    stop = resolve(gen_slice.stop)
    step = gen_slice.step if gen_slice.step is not None else 1

    if start is None:
        start = gen_min
    if stop is None:
        stop = gen_max

    return list(range(start, stop, step))

def main(log_file, basepath, gens, fps, size):

    with open(os.path.join(basepath, log_file), "r") as f:
        log = f.read()


    log_data = []
    pattern = r"coord:\((.*?)\), indv_id:(\d+), fit:(\S+)"

    for line in log.split("=== gen: ")[1:]:
        i = line.find("bounds")
        gen = int(line[:i])

        bounds, pop = line[i:].split(" ===\n")
        bounds = eval(bounds.replace("bounds: ",""))  # Keeping eval() as per your request

        matches = [re.match(pattern, s).groups() for s in pop.split("\n")[:-1]]
        pop_data = [
            {
                "gen": gen,  # Add generation column
                "coord": tuple(map(int, coord.split(", "))),
                "indv_id": int(indv_id),
                "fit": float(fit) if fit != "nan" else np.nan
            }
            for coord, indv_id, fit in matches
        ]

        log_data.extend(pop_data)  # Collect all data

    # Convert to a single DataFrame
    df = pd.DataFrame(log_data)

    # Display the first few rows
    print(df)

    def make_circles(group):
        GAP_RATIO = 0.2  # Relative gap size

        n = len(group)
        n_root = int(np.ceil(np.sqrt(n)))  # Determine grid size

        # Compute radius and gaps
        available_space = 1  # Full grid cell size
        step_size = available_space / (n_root + (n_root + 1) * GAP_RATIO)  
        gap = step_size * GAP_RATIO  # Set edge gap equal to computed gap
        radius = step_size / 2  

        # Compute positions with equal edge and internal gaps
        positions = np.linspace(gap + radius, 1 - gap - radius, n_root)
        x, y = np.meshgrid(positions, positions)
        x, y = x.flatten()[:n], y.flatten()[:n]  # Limit to `n` circles

        # Create array with [x, y, radius, fitness]
        circles = np.column_stack((x, y, np.full(n, radius), group["fit"].values))

        return circles



    figsize = np.array(size)
    figsize *= 12/figsize.max()
    # Create the figure and axis
    fig, (ax, cax) = plt.subplots(figsize=figsize, ncols=2, width_ratios=[1,0.05])


    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.set_xticks(np.arange(0, size[0] + 1))
    ax.set_yticks(np.arange(0, size[1] + 1))
    ax.grid()

    # Normalize and create the colormap
    cmap = cm.get_cmap("rainbow").copy()
    cmap.set_bad("black")

    # Store the colorbar outside of the update_plot function
    cbar = None

    # Define the function to update the plot for each frame (generation)
    def update_plot(gen):

        ax.clear()  # Clear previous circles
        ax.set_xlim(0, size[0])
        ax.set_ylim(0, size[1])
        ax.set_xticks(np.arange(0, size[0] + 1))
        ax.set_yticks(np.arange(0, size[1] + 1))
        ax.grid()
        ax.set_aspect('equal', 'box')
        plt.suptitle(f"gen {gen}")

        norm = mcolors.Normalize(vmin=np.nanmin(df[df["gen"] == gen].fit.values), vmax=np.nanmax(df[df["gen"] == gen].fit.values))
        # Loop through all the coordinates and plot the circles
        for coord, group in df[df["gen"] == gen].groupby("coord"):
            circles = make_circles(group)
            for x, y, r, fit in circles:
                color = cmap(norm(fit)) if np.isfinite(fit) else "black"
                circle = plt.Circle((x + coord[0], y + coord[1]), r, facecolor=color, edgecolor="black")
                ax.add_patch(circle)

        # Update the colorbar for the current generation
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Empty array so it doesn't override the colors
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Fitness Value")
        fig.subplots_adjust()


    frames = resolve_gen_range(df, gens)

    # Create the animation (here we loop over a range of generations, assuming gen values are integers)
    anim = FuncAnimation(fig, update_plot, frames=frames, interval=1000/fps)

    plt.show()



if __name__ == "__main__":
    import argparse
    from flatspin.cmdline import StoreKeyValue, eval_params

    parser = argparse.ArgumentParser(description="visualise the elite map")

    # common
    parser.add_argument('-l', '--log', metavar='FILE', default="elite_map.log",
                        help=r'name of log')
    parser.add_argument('-b', '--basepath', metavar='FILE', default="",
                        help=r'location of log and index')
    parser.add_argument("-g", "--gen", type=str, default=slice(None, None),
                    help="Slice like for generations")

    parser.add_argument('-f', '--fps', default=2, type=float)
    parser.add_argument('-s', '--size', default=(20,20), type=int, nargs=2)


    args = parser.parse_args()
    main(args.log, args.basepath, parse_slice(args.gen), args.fps, args.size)
