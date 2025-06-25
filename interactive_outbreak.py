"""
panel serve interactive_outbreak.py --autoreload
"""

import os
from itertools import cycle

import geopandas as gpd
import numpy as np
import pandas as pd
import panel as pn
from bokeh import models, plotting
from bokeh.palettes import Reds256, Oranges256, Blues256, diverging_palette, Pastel1

from mixing import init_gravity_diffusion
from settlements import parse_grid3_settlements
from spatial_outbreak_sim import Params, init_state, step_state, simulate

PRIMARY_COLOR = "#0072B5"
SECONDARY_COLOR = "#B54300"

nigeria_admin1_names = ["Jigawa", "Kano", "Katsina", "Kaduna"]
nigeria_admin1_names += ["Bauchi", "Yobe", "Borno", "Gombe"]

niger_admin1_names = ["Zinder", "Diffa", "Maradi"]


@pn.cache
def get_data():

    nigeria_settlements_df = parse_grid3_settlements(nigeria_admin1_names, "Nigeria")

    niger_settlements_df = parse_grid3_settlements(niger_admin1_names, country="Niger")

    return pd.concat([nigeria_settlements_df, niger_settlements_df])

def reset_params():

    return Params(
         beta=8, seasonality=0.1, demog_scale=0.8, 
         mixing_scale=0.003, distance_exponent=1.5)

def reset_state():

    return init_state(settlements_df, params)

def on_beta_change(value):
    params.beta = value

def on_seasonality_change(value):
    params.seasonality = value

def on_demog_scale_change(value):
    params.demog_scale = value
    params.biweek_avg_births = params.demog_scale * params.births / 26.
    params.biweek_death_prob = params.demog_scale * params.births / params.population / 26.

def on_mixing_scale_change(value):
    params.mixing_scale = np.power(10, value)
    params.mixing = init_gravity_diffusion(settlements_df, params.mixing_scale, params.distance_exponent)

def on_distance_exponent_change(value):
    params.distance_exponent = value
    params.mixing = init_gravity_diffusion(settlements_df, params.mixing_scale, params.distance_exponent)


pn.extension(design="material", sizing_mode="stretch_width")

settlements_df = get_data().reset_index()


params = reset_params()

beta_slider = pn.widgets.FloatSlider(value=params.beta, start=0, end=50, step=1, name='beta')
bound_beta = pn.bind(on_beta_change, value=beta_slider)

seasonality_slider = pn.widgets.FloatSlider(value=params.seasonality, start=0, end=0.3, step=0.02, name='seasonality')
bound_seasonality = pn.bind(on_seasonality_change, value=seasonality_slider)

demog_scale_slider = pn.widgets.FloatSlider(value=params.demog_scale, start=0.1, end=1.5, step=0.05, name='demog_scale')
bound_demog_scale = pn.bind(on_demog_scale_change, value=demog_scale_slider)

mixing_scale_slider = pn.widgets.FloatSlider(value=np.log10(params.mixing_scale), start=-4, end=-2, name='log10(mixing_scale)')
bound_mixing_scale = pn.bind(on_mixing_scale_change, value=mixing_scale_slider)

distance_exponent_slider = pn.widgets.FloatSlider(value=params.distance_exponent, start=0.5, end=2.5, step=0.1, name='distance_exponent')
bound_distance_exponent = pn.bind(on_distance_exponent_change, value=distance_exponent_slider)


state = reset_state()

source = models.ColumnDataSource(dict(
    name=settlements_df.index,
    x=settlements_df.Long, 
    y=settlements_df.Lat,
    lganame=settlements_df.adm2_name,
    statename=settlements_df.adm1_name,
    size=0.03*np.sqrt(settlements_df.population),
    population=settlements_df.population,
    births=settlements_df.births,
    prevalence=state[:, 1] / state[:, :].sum(axis=-1),
    reff=params.beta * state[:, 0] / state[:, :].sum(axis=-1),
))


path = os.path.join("GRID3", "GRID3_NGA_-_Operational_LGA_Boundaries", "GRID3_NGA_-_Operational_LGA_Boundaries.shp")
lgas = gpd.read_file(path)
lgas["geometry"] = lgas["geometry"].to_crs(crs="EPSG:4326")
lgas = lgas[lgas.statename.isin(nigeria_admin1_names)]
# print(lgas.iloc[0])
geo_source = models.GeoJSONDataSource(
    geojson=lgas.to_json()
)


lga_names = settlements_df.adm2_name.dropna().unique()  # TODO: extend to unique (adm1, adm2)
colors = []
for _, c in zip(lga_names, cycle(Pastel1[6])):
    colors.append(c)

lga_ts_source = models.ColumnDataSource(dict(time=[]) | {k: [] for k in lga_names})
lga_focus_ts = plotting.figure(x_axis_label="Time (years)", y_axis_label="Detected AFP", width=500, height=200,
                               tools="hover", tooltips="$name: @$name")
vbars = lga_focus_ts.vbar_stack(lga_names, x='time', width=0.9/26, source=lga_ts_source, 
                                # legend_label=lga_names, 
                                color=colors, alpha=0.8)
lga_focus_ts.visible = False
# lga_focus_ts.legend.location = "top_left"
# lga_focus_ts.legend.orientation = "horizontal"
# lga_focus_ts.legend.label_text_font_size = "6pt"

def lga_selection(attr, old, new):

    global callback, vbars

    selected_lgas = lgas.iloc[new].lganame.values
    indices = settlements_df[settlements_df.adm2_name.isin(selected_lgas)].index
    source.selected.indices = indices

    for vbar in vbars: 
        lga_focus_ts.renderers.remove(vbar)
    vbars = lga_focus_ts.vbar_stack(selected_lgas, x='time', width=0.9/26, source=lga_ts_source, 
                                    # legend_label=lga_names, 
                                    color=colors[:len(selected_lgas)], alpha=0.8)

    lga_focus_ts.visible = len(new) > 0
    callback.running = len(new) == 0

    if len(new) == 1:
        lga_focus_ts.title.text = "%s, %s" % (lgas.iloc[new].lganame.values[0], lgas.iloc[new].statename.values[0])
    elif len(new) > 1:
        lga_focus_ts.title.text = "Multiple selected LGAs (hover for details)"

geo_source.selected.on_change('indices', lga_selection)

prev_cmap = models.LogColorMapper(palette=Reds256[::-1], low=1e-4, high=0.01)
reff_cmap = models.LogColorMapper(palette=diverging_palette(Blues256, Oranges256, n=256), low=0.25, high=4.0)

prev_scatter = plotting.figure(
    x_axis_label="Longitude", y_axis_label="Latitude",
    title="Prevalence", width=500, height=500,
)
prev_scatter.add_tools("tap", "box_select", "lasso_select")
points = prev_scatter.scatter(x="x", y="y", size="size", color={"field": "prevalence", "transform": prev_cmap}, source=source, alpha=0.5)
hover = models.HoverTool(
    renderers=[points],
    tooltips=[
    ("name", "@name"),
    ("population", "@population{0.0 a}"),
    # ("births", "@births"),
    ("prevalence", "@prevalence{%0.2f}"),
    ("reff", "@reff"),
    ("lga", "@lganame"),
    ("state", "@statename"),
])
# prev_scatter.add_tools(hover)

shapes = prev_scatter.patches('xs', 'ys', source=geo_source, fill_alpha=0.1, fill_color="lightgray", line_color="lightgray", line_width=0.5)
hover2 = models.HoverTool(
    renderers=[shapes], 
    tooltips=[
    ("LGA", "@lganame"),
    ("State", "@statename"),
])
prev_scatter.add_tools(hover2)

reff_scatter = plotting.figure(
    x_axis_label="Longitude", y_axis_label="Latitude",
    title="Effective reproductive number", width=500, height=500,
)
reff_scatter.add_tools(hover)
reff_scatter.patches('xs', 'ys', source=geo_source, fill_alpha=0.1, fill_color="lightgray", line_color="lightgray", line_width=0.5)
reff_scatter.scatter(x="x", y="y", size="size", color={"field": "reff", "transform": reff_cmap}, source=source, alpha=0.5)

ts_source = models.ColumnDataSource(dict(
    time=[],
    infected_fraction=[],
))

prev_ts = plotting.figure(x_axis_label="Time (years)", y_axis_label="Prevalence (%)", width=500, height=200)
prev_ts.line(x="time", y="infected_fraction", source=ts_source, color="red")
prev_ts.title.text = "Regional total infections"


def stream():
    step_state(state, params)

    new_data = dict(
        time=[state.t/26.],
        infected_fraction=[float(100 * state[:, 1].sum() / state[:, :].sum())])

    ts_source.stream(new_data, rollover=260)

    settlements_df["AFP"] = np.random.poisson(lam=state[:, 1]/2000.)
    afp_by_lga = settlements_df.groupby("adm2_name").AFP.sum()

    # print(afp_by_lga.to_dict())

    lga_ts_source.stream(dict(time=[state.t/26.]) | {k: [afp_by_lga.loc[k]] for k in lga_names}, rollover=26)

    prev_scatter.title.text = "Prevalence (year = {:.2f})".format(state.t/26.)
    source.data["prevalence"] = state[:, 1] / state[:, :].sum(axis=-1)
    source.data["reff"] = params.beta * state[:, 0] / state[:, :].sum(axis=-1)


callback_period = 100
callback = pn.state.add_periodic_callback(stream, callback_period)

speed_slider = pn.widgets.FloatSlider(value=callback_period, start=10, end=200, step=10, name='refresh rate (ms)')
def on_speed_change(value):
    callback.period = value
bound_speed = pn.bind(on_speed_change, value=speed_slider)

reset_button = pn.widgets.Button(name='Reset', button_type='primary')
def reset(event):
    global params, state, effective_campaign_coverage
    params = reset_params()
    state = reset_state()
    beta_slider.value = params.beta
    seasonality_slider.value = params.seasonality
    demog_scale_slider.value = params.demog_scale
    mixing_scale_slider.value = np.log10(params.mixing_scale)
    distance_exponent_slider.value = params.distance_exponent
    speed_slider.value = callback_period
    if not callback.running:
        callback.start()
    ts_source.data = {k: [] for k in ts_source.data}
    lga_ts_source.data = {k: [] for k in lga_ts_source.data}
reset_button.on_click(reset)

pause_button = pn.widgets.Toggle(name='Pause/Resume', value=True)
pause_button.link(callback, bidirectional=True, value='running')

import_infections_button = pn.widgets.Button(name='Import infections', button_type='primary')
def import_infections(event):
    global state
    # print(source.selected.indices)
    # print(state.shape)
    # print(source.data["name"][source.selected.indices])
    n_susceptible = state[source.selected.indices, 0]
    # print(n_susceptible)
    dI = np.minimum(n_susceptible, 10)
    # print(dI)
    state[source.selected.indices, 1] += dI  # new infections
    state[source.selected.indices, 0] -= dI  # no longer susceptible
import_infections_button.on_click(import_infections)


sliders = pn.Column(
    "### Simulation parameters",
    pn.Row(beta_slider, bound_beta),
    pn.Row(seasonality_slider, bound_seasonality),
    pn.Row(demog_scale_slider, bound_demog_scale),
    pn.layout.Divider(),
    "### Mixing parameters",
    pn.Row(mixing_scale_slider, bound_mixing_scale),
    pn.Row(distance_exponent_slider, bound_distance_exponent),
    pn.layout.Divider(),
    "### Playback controls",
    pn.Row(speed_slider, bound_speed),
    pn.Row(reset_button, pause_button),
    pn.Row(import_infections_button),
)

pn.template.MaterialTemplate(
    site="OPV Demo",
    title="Interactive Spatial Simulation",
    sidebar=[sliders],
    main=[pn.Row(prev_scatter, reff_scatter), pn.Row(prev_ts, lga_focus_ts)],
).servable()