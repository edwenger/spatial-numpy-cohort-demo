import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


datadir = "~/GitHub/epipack-spatial-demo/data"  # TODO: change this to internal path (or include download code)


def parse(adm1_name):

    os.makedirs("GRID3", exist_ok=True)
    output_path = os.path.join("GRID3", "%s_grid3_parsed.csv" % adm1_name.lower())

    if os.path.exists(output_path):
        df = pd.read_csv(output_path, index_col=0)
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs="EPSG:4326")

    print("Parsing settlements in %s state..." % adm1_name)
    state_shapes = gpd.read_file(os.path.join(datadir, "GRID3_Nigeria_-_State_Boundaries.geojson"))
    mask = state_shapes[state_shapes.statename == adm1_name]
    gdf = gpd.read_file(os.path.join(datadir, "GRID3_Nigeria_Settlement_Extents_Version_01.02..geojson"), mask=mask)
    gdf = gdf[gdf.adm1_name == adm1_name]

    centroids = gdf.to_crs('+proj=cea').centroid.to_crs(gdf.crs)
    gdf["x"] = centroids.x
    gdf["y"] = centroids.y

    gdf = gdf[["x", "y", "population", "adm1_name", "adm2_name", "type", "under5", "under1"]]
    gdf.to_csv(output_path)

    return gdf


def parse_niger(adm1_name):
    
    os.makedirs("GRID3", exist_ok=True)
    output_path = os.path.join("GRID3", "%s_grid3_parsed.csv" % adm1_name.lower())

    if os.path.exists(output_path):
        df = pd.read_csv(output_path, index_col=0)
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs="EPSG:4326")

    print("Parsing settlements in %s state..." % adm1_name)
    state_shapes = gpd.read_file(os.path.join(datadir, "ner_adm_ignn_20230720_em_shp", "NER_admbnda_adm1_IGNN_20230720_em.shp"))
    mask = state_shapes[state_shapes.ADM1_FR == adm1_name]
    gdf = gpd.read_file(os.path.join(datadir, "GRID3_NER_Settlement_Extents_v3_0_7780863971779087557.geojson"), mask=mask)
    print(gdf.head())
    print(gdf.iloc[0])
    # gdf = gdf[gdf.adm1_name == adm1_name]

    centroids = gdf.to_crs('+proj=cea').centroid.to_crs(gdf.crs)
    gdf["x"] = centroids.x
    gdf["y"] = centroids.y

    gdf = gdf[["x", "y", "building_count", "building_area", "probability", "type"]]
    gdf.to_csv(output_path)

    return gdf

def plot_settlements(df):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    df["birth_rate"] = df.under1 / df.population
    df.plot(kind="scatter", x="x", y="y", s=0.3*np.sqrt(df.population), alpha=0.5, c='birth_rate', ax=ax)
    fig.set_tight_layout(True)


if __name__ == "__main__":
    
    adm1_names = ["Zinder", "Maradi", "Diffa", "Tahoua", "Dosso"]

    lga_shape_path = os.path.join("GRID3", "ner_adm_ignn_20230720_em_shp", "NER_admbnda_adm2_IGNN_20230720_em.shp")
    lgas = gpd.read_file(lga_shape_path)
    lgas["geometry"] = lgas["geometry"].to_crs(crs="EPSG:4326")
    lgas = lgas[lgas.ADM1_FR.isin(adm1_names)]

    gdf = None
    for adm1_name in adm1_names:

        tmp_gdf = parse_niger(adm1_name)
        pointInPoly = gpd.sjoin(lgas, tmp_gdf, predicate='contains').set_index("index_right")
        tmp_gdf["adm1_name"] = pointInPoly.ADM1_FR
        tmp_gdf["adm2_name"] = pointInPoly.ADM2_FR

        tmp_gdf = tmp_gdf[tmp_gdf.adm1_name == adm1_name]

        # tmp hacks
        tmp_gdf = tmp_gdf[(tmp_gdf.probability > 0.9) & (tmp_gdf.building_count > 20)]
        tmp_gdf["population"] = tmp_gdf.building_count * 4.5
        tmp_gdf["under1"] = tmp_gdf.population * 0.05
        tmp_gdf["under5"] = tmp_gdf.population * 0.25

        if gdf is not None:
            gdf = pd.concat([gdf, tmp_gdf])
        else:
            gdf = tmp_gdf.copy()

    # adm1_names = ["Kano", "Jigawa", "Katsina", "Kaduna"]
    # # adm1_names = ["Sokoto", "Kebbi", "Zamfara"]
    # # adm1_names = ["Yobe", "Borno", "Bauchi", "Gombe", "Kano", "Jigawa", "Katsina"]

    # lga_shape_path = os.path.join("GRID3", "GRID3_NGA_-_Operational_LGA_Boundaries", "GRID3_NGA_-_Operational_LGA_Boundaries.shp")
    # lgas = gpd.read_file(lga_shape_path)
    # lgas["geometry"] = lgas["geometry"].to_crs(crs="EPSG:4326")
    # lgas = lgas[lgas.statename.isin(adm1_names)]

    # gdf = None
    # for adm1_name in adm1_names:
    #     tmp_gdf = parse(adm1_name)

    #     pointInPoly = gpd.sjoin(lgas, tmp_gdf, predicate='contains').set_index("index_right")
    #     # print(pointInPoly[["lganame", "adm2_name"]].value_counts().head(30))
    #     # print(pointInPoly.loc[14569][["lganame", "adm2_name", "statename", "adm1_name", "population", "under1"]])
    #     # print(pointInPoly.head())
    #     tmp_gdf["adm1_name"] = pointInPoly.statename
    #     tmp_gdf["adm2_name"] = pointInPoly.lganame

    #     if gdf is not None:
    #         gdf = pd.concat([gdf, tmp_gdf])
    #     else:
    #         gdf = tmp_gdf.copy()

    gdf["birth_rate"] = gdf.under1 / gdf.population
    gdf = gdf[gdf.population > 500]

    print(gdf.shape)
    print(gdf.head())
    print(gdf.iloc[0])

    plot_settlements(gdf)
    plt.show()