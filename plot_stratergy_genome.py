import numpy as np
import pandas as pd
from flatspin import data

import matplotlib.pyplot as plt

def main(basepath):
    ds = data.Dataset.read(basepath)

    ds.index['stratergy_genome'] = ds.index['stratergy_genome'].apply(lambda s: eval(s, {'array': np.array}))

    ind = ds.index
    genome_df = pd.DataFrame(ind['stratergy_genome'].tolist(),
                         index=ind.index,
                         columns=[f'gene_{i}' for i in range(len(ind['stratergy_genome'].iloc[0]))])

    result = pd.concat([ind[['gen']], genome_df], axis=1)


    grouped_mean = result.groupby('gen').mean()
    grouped_std = result.groupby('gen').std()

    plt.figure(figsize=(20, 15))

    for gene in grouped_mean.columns:
        mean = grouped_mean[gene]
        std = grouped_std[gene]

        plt.plot(grouped_mean.index, mean, label=gene)
        plt.fill_between(grouped_mean.index, mean - std, mean + std, alpha=0.2)

    plt.xlabel('Generation')
    plt.ylabel('Gene value')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="visualise the elite map")

    parser.add_argument('-b', '--basepath', metavar='FILE', default="",
                        help=r'location of log and index')

    args = parser.parse_args()

    main(basepath = args.basepath)