"""
panel serve interactive_opv.py --autoreload
"""
import os
from itertools import cycle

import geopandas as gpd
import numpy as np
import panel as pn
from bokeh import models, plotting
from bokeh.palettes import Reds256, Oranges256, Blues256, diverging_palette, Pastel1

from mixing import init_gravity_diffusion
from settlements import parse_settlements, parse_grid3_settlements
from spatial_opv_sim import Params, init_state, step_state

PRIMARY_COLOR = "#0072B5"
SECONDARY_COLOR = "#B54300"

# adm1_names = ["Jigawa"]
# adm1_names = ["Jigawa", "Kano", "Katsina"]
adm1_names = ["Sokoto", "Kebbi"]


@pn.cache
def get_data():
    # return parse_settlements()
    return parse_grid3_settlements(adm1_names)

def reset_params():

    return Params(
        beta=8, seasonality=0.1, demog_scale=0.8, 
        mixing_scale=0.003, distance_exponent=1.5,
        opv_reversion_fraction=0.05, opv_relative_beta=0.3)

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

def on_opv_reversion_change(value):
    params.opv_reversion_fraction = value

def on_opv_beta_change(value):
    params.opv_relative_beta = value

###

pn.extension(design="material", sizing_mode="stretch_width")

###

settlements_df = get_data().reset_index()
# settlements_df = settlements_df.iloc[slice(0, 20), :]  # include only largest few settlements for testing

###

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

opv_reversion_slider = pn.widgets.FloatSlider(value=params.opv_reversion_fraction, start=0, end=1, step=0.05, name='opv_reversion_rate')
bound_opv_reversion = pn.bind(on_opv_reversion_change, value=opv_reversion_slider)

opv_beta_slider = pn.widgets.FloatSlider(value=params.opv_relative_beta, start=0, end=1, step=0.05, name='opv_rel_beta')
bound_opv_beta = pn.bind(on_opv_beta_change, value=opv_beta_slider)

###

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
lgas = lgas[lgas.statename.isin(adm1_names)]
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
    prev_WPV=[],
    prev_OPV=[],
))

prev_ts = plotting.figure(x_axis_label="Time (years)", y_axis_label="Prevalence (%)", width=500, height=200)
prev_ts.line(x="time", y="prev_WPV", source=ts_source, color="red")
prev_ts.line(x="time", y="prev_OPV", source=ts_source, color="blue")
prev_ts.title.text = "Regional total infections"

def stream():
    step_state(state, params)

    new_data = dict(
        time=[state.t/26.],
        prev_WPV=[float(100 * state[:, 1].sum() / state[:, :].sum())],  # (prev in %)
        prev_OPV=[float(100 * state[:, 2].sum() / state[:, :].sum())])

    ts_source.stream(new_data, rollover=260)

    settlements_df["AFP"] = np.random.poisson(lam=state[:, 1]/2000.)
    afp_by_lga = settlements_df.groupby("adm2_name").AFP.sum()

    # print(afp_by_lga.to_dict())

    lga_ts_source.stream(dict(time=[state.t/26.]) | {k: [afp_by_lga.loc[k]] for k in lga_names}, rollover=26)

    prev_scatter.title.text = "Prevalence (year = {:.2f})".format(state.t/26.)
    source.data["prevalence"] = state[:, 1] / state[:, :].sum(axis=-1)
    source.data["reff"] = params.beta * state[:, 0] / state[:, :].sum(axis=-1)
    
    if campaigns_per_year > 0 and state.t % (26./campaigns_per_year) < 1:
        
        missed = np.random.random(len(params.population)) < (np.log10(params.population) / missed_campaign_log10_pop_threshold)

        dI = np.random.binomial(n=state[:, 0], p=effective_campaign_coverage * missed)
        state[:, 2] += dI
        state[:, 0] -= dI


effective_campaign_coverage = 0.2
coverage_slider = pn.widgets.FloatSlider(value=effective_campaign_coverage, start=0, end=1, step=0.05, name='effective campaign coverage')
def on_coverage_change(value):
    global effective_campaign_coverage
    effective_campaign_coverage = value
bound_coverage = pn.bind(on_coverage_change, value=coverage_slider)

campaigns_per_year = 2
campaign_frequency_slider = pn.widgets.FloatSlider(value=campaigns_per_year, start=0, end=13, step=1, name='campaigns per year')
def on_campaign_frequency_change(value):
    global campaigns_per_year
    campaigns_per_year = value
bound_campaign_frequency = pn.bind(on_campaign_frequency_change, value=campaign_frequency_slider)

missed_campaign_log10_pop_threshold = 8
missed_campaign_pop_threshold_slider = pn.widgets.FloatSlider(value=missed_campaign_log10_pop_threshold, start=1, end=10, step=0.5, name='missed campaign log10(pop) thresh')
def on_missed_campaign_pop_threshold_change(value):
    global missed_campaign_log10_pop_threshold
    missed_campaign_log10_pop_threshold = value
bound_missed_campaign_pop_threshold = pn.bind(on_missed_campaign_pop_threshold_change, value=missed_campaign_pop_threshold_slider)

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
    opv_reversion_slider.value = params.opv_reversion_fraction
    opv_beta_slider.value = params.opv_relative_beta
    speed_slider.value = callback_period
    if not callback.running:
        callback.start()
    ts_source.data = {k: [] for k in ts_source.data}
    lga_ts_source.data = {k: [] for k in lga_ts_source.data}
reset_button.on_click(reset)

pause_button = pn.widgets.Toggle(name='Pause/Resume', value=True)
pause_button.link(callback, bidirectional=True, value='running')

OPV_campaign_button = pn.widgets.Button(name='OPV Campaign', button_type='primary')
def opv_campaign(event):
    global state
    # print(source.selected.indices)
    # print(source.data["name"][source.selected.indices])
    dI = np.random.binomial(n=state[source.selected.indices, 0], p=effective_campaign_coverage)
    state[source.selected.indices, 2] += dI
    state[source.selected.indices, 0] -= dI
OPV_campaign_button.on_click(opv_campaign)

###

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
    "### OPV parameters",
    pn.Row(opv_reversion_slider, bound_opv_reversion),
    pn.Row(opv_beta_slider, bound_opv_beta),
    pn.Row(coverage_slider, bound_coverage),
    pn.Row(campaign_frequency_slider, bound_campaign_frequency),
    pn.Row(missed_campaign_pop_threshold_slider, bound_missed_campaign_pop_threshold),
    "### Playback controls",
    pn.Row(speed_slider, bound_speed),
    pn.Row(reset_button, pause_button),
    pn.Row(OPV_campaign_button),
)

pn.template.MaterialTemplate(
    site="OPV Demo",
    title="Interactive Spatial Simulation",
    sidebar=[sliders],
    main=[pn.Row(prev_scatter, reff_scatter), pn.Row(prev_ts, lga_focus_ts)],
).servable()