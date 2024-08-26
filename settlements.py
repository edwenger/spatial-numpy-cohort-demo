import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


def parse_settlements():

    england_wales_path = "./CvsMeasles/"

    populations = pd.read_csv(os.path.join(england_wales_path, "ewPu4464.csv"), index_col=0)
    initial_pops = populations.iloc[0]
    # print(initial_pops.head())

    locations = pd.read_csv(os.path.join(england_wales_path, "ewXYu4464.csv"), index_col=0).T
    # print(locations.head())

    births = pd.read_csv(os.path.join(england_wales_path, "ewBu4464.csv"), index_col=0)
    initial_births = births.iloc[0]
    # print(initial_births.head())

    df = locations.join(initial_pops.rename("population")).join(initial_births.rename("births")).sort_values(by="population", ascending=False)
    # print(df.head(25))

    return df


def parse_grid3_settlements(adm1_names):

    df = None
    for adm1_name in adm1_names:
        path = os.path.join("GRID3", "%s_grid3_parsed.csv" % adm1_name.lower())
        tmp_df = pd.read_csv(path, index_col=0)
        if df is not None:
            df = pd.concat([df, tmp_df])
        else:
            df = tmp_df.copy()

    df["births"] = df.under1
    df["Long"] = df.x
    df["Lat"] = df.y

    return df[df.population > 500]


def plot_settlements(df):

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    df["birth_rate"] = df.births / df.population
    df.plot(kind="scatter", x="Long", y="Lat", s=0.1*np.sqrt(df.population), alpha=0.5, c='population', norm=LogNorm(), cmap='magma', ax=axs[0], title="population")
    df.plot(kind="scatter", x="Long", y="Lat", s=0.1*np.sqrt(df.population), alpha=0.5, c='birth_rate', ax=axs[1], title="birth rate")
    fig.set_tight_layout(True)


if __name__ == '__main__':

    settlements_df = parse_settlements()
    # settlements_df = parse_grid3_settlements(["Jigawa", "Kano"])
    plot_settlements(settlements_df)
    plt.show()