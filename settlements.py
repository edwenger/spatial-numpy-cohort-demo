import os

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns


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


def parse_grid3_settlements(adm1_names, country="Nigeria"):

    if country == "Nigeria":
        lga_shape_path = os.path.join("GRID3", "GRID3_NGA_-_Operational_LGA_Boundaries", "GRID3_NGA_-_Operational_LGA_Boundaries.shp")
        lgas = gpd.read_file(lga_shape_path)
        lgas["geometry"] = lgas["geometry"].to_crs(crs="EPSG:4326")
        lgas = lgas[lgas.statename.isin(adm1_names)]
    elif country == "Niger":
        lga_shape_path = os.path.join("GRID3", "ner_adm_ignn_20230720_em_shp", "NER_admbnda_adm2_IGNN_20230720_em.shp")
        lgas = gpd.read_file(lga_shape_path)
        lgas["geometry"] = lgas["geometry"].to_crs(crs="EPSG:4326")
        lgas = lgas[lgas.ADM1_FR.isin(adm1_names)]
    else:
        raise ValueError("Country %s not supported" % country)

    df = None
    for adm1_name in adm1_names:
        path = os.path.join("GRID3", "%s_grid3_parsed.csv" % adm1_name.lower())
        tmp_df = pd.read_csv(path, index_col=0)
        tmp_gdf = gpd.GeoDataFrame(tmp_df, geometry=gpd.points_from_xy(tmp_df['x'], tmp_df['y']), crs="EPSG:4326")

        pointInPoly = gpd.sjoin(lgas, tmp_gdf, predicate='contains').set_index("index_right")
        if country == "Nigeria":
            tmp_gdf["adm1_name"] = pointInPoly.statename
            tmp_gdf["adm2_name"] = pointInPoly.lganame
        elif country == "Niger":
            tmp_gdf["adm1_name"] = pointInPoly.ADM1_FR
            tmp_gdf["adm2_name"] = pointInPoly.ADM2_FR

            tmp_gdf = tmp_gdf[tmp_gdf.adm1_name == adm1_name]

            # tmp hacks to reconcile Niger settlements file not having normalized building counts to population totals
            tmp_gdf = tmp_gdf[(tmp_gdf.probability > 0.9) & (tmp_gdf.building_count > 20)]
            tmp_gdf["population"] = tmp_gdf.building_count * 4.5
            tmp_gdf["under1"] = tmp_gdf.population * 0.05
            tmp_gdf["under5"] = tmp_gdf.population * 0.25
        else:
            raise ValueError("Country %s not supported" % country)

        if df is not None:
            df = pd.concat([df, tmp_gdf])
        else:
            df = tmp_gdf.copy()

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


def plot_birth_rate_vs_log_population(df):
    # plot birth rate (y-axis) vs log population (x-axis)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    df["birth_rate"] = df.births / df.population
    df["log10_population"] = np.log10(df.population)
    # Use seaborn to color by adm1_name (categorical)
    sns.scatterplot(
        data=df,
        x="log10_population",
        y="birth_rate",
        hue="adm1_name",
        alpha=0.1,
        ax=ax,
    )
    ax.set_title("Birth rate vs log(population) colored by adm1_name")
    fig.set_tight_layout(True)


if __name__ == '__main__':

    # settlements_df = parse_settlements()
    adm1_names = ["Jigawa", "Kano", "Katsina", "Kaduna"]
    adm1_names += ["Yobe", "Borno", "Bauchi", "Gombe"]
    # adm1_names = ["Sokoto", "Kebbi", "Zamfara"]
    # adm1_names = ["Yobe", "Borno", "Bauchi", "Gombe", "Kano", "Jigawa", "Katsina", "Sokoto", "Kebbi", "Zamfara", "Kaduna"]

    nigeria_settlements_df = parse_grid3_settlements(adm1_names, "Nigeria")

    # adm1_names = ["Zinder", "Maradi", "Diffa", "Tahoua", "Dosso"]
    adm1_names = ["Zinder", "Maradi", "Diffa"]
    niger_settlements_df = parse_grid3_settlements(adm1_names, country="Niger")

    settlements_df = pd.concat([nigeria_settlements_df, niger_settlements_df])

    settlements_df.sort_values(by="population", ascending=False, inplace=True)
    print(settlements_df[settlements_df.adm1_name == "Jigawa"].head(25))

    # Compute population-weighted centroids and total population per admin2
    def weighted_mean(group, value_col):
        return np.average(group[value_col], weights=group["population"])

    adm2_centroids = settlements_df.groupby(["adm1_name", "adm2_name"]).apply(
        lambda g: pd.Series({
            "Long": weighted_mean(g, "Long"),
            "Lat": weighted_mean(g, "Lat"),
            "population": g["population"].sum(),
            "births": g["births"].sum(),
        })
    ).reset_index()

    # plot_settlements(settlements_df)
    plot_settlements(adm2_centroids)

    # plot_birth_rate_vs_log_population(settlements_df)

    plt.show()