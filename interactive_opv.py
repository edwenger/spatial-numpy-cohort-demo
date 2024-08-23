from collections import deque

import numpy as np
import panel as pn
from bokeh import models, plotting
from bokeh.palettes import Reds256, Oranges256, Blues256, diverging_palette

from mixing import init_gravity_diffusion
from settlements import parse_settlements
from spatial_opv_sim import Params, init_state, step_state

PRIMARY_COLOR = "#0072B5"
SECONDARY_COLOR = "#B54300"

@pn.cache
def get_data():
    return parse_settlements()

def reset_params():

    return Params(
        beta=8, seasonality=0.1, demog_scale=0.65, 
        mixing_scale=0.001, distance_exponent=1.5,
        opv_reversion_fraction=0.2, opv_relative_beta=0.2)

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

settlements_df = get_data()
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
    size=0.03*np.sqrt(settlements_df.population),
    population=settlements_df.population,
    births=settlements_df.births,
    prevalence=state[:, 1] / state[:, :].sum(axis=-1),
    reff=params.beta * state[:, 0] / state[:, :].sum(axis=-1),
))

hover = models.HoverTool(tooltips=[
    ("name", "@name"),
    ("population", "@population{0.0 a}"),
    # ("births", "@births"),
    ("prevalence", "@prevalence{%0.2f}"),
    ("reff", "@reff"),
])

prev_cmap = models.LogColorMapper(palette=Reds256[::-1], low=1e-4, high=0.01)
reff_cmap = models.LogColorMapper(palette=diverging_palette(Blues256, Oranges256, n=256), low=0.25, high=4.0)

prev_scatter = plotting.figure(
    x_axis_label="Longitude", y_axis_label="Latitude",
    title="Prevalence", width=500, height=500,
)
prev_scatter.add_tools(hover)
prev_scatter.add_tools("tap", "box_select", "lasso_select")
prev_scatter.scatter(x="x", y="y", size="size", color={"field": "prevalence", "transform": prev_cmap}, source=source, alpha=0.5)

reff_scatter = plotting.figure(
    x_axis_label="Longitude", y_axis_label="Latitude",
    title="Effective reproductive number", width=500, height=500,
)
reff_scatter.add_tools(hover)
reff_scatter.scatter(x="x", y="y", size="size", color={"field": "reff", "transform": reff_cmap}, source=source, alpha=0.5)


ts_source = models.ColumnDataSource(dict(
    time=np.arange(0, 10*26),
    prev_WPV=np.zeros(10*26),
    prev_OPV=np.zeros(10*26),
))

prev_ts = plotting.figure(x_axis_label="Time (years)", y_axis_label="Prevalence (%)", width=500, height=200)
prev_ts.line(x="time", y="prev_WPV", source=ts_source, color="red")
prev_ts.line(x="time", y="prev_OPV", source=ts_source, color="blue")

time_ts_list = deque()
prev_WPV_ts_list = deque()
prev_OPV_ts_list = deque()

def stream():
    step_state(state, params)

    prev_scatter.title.text = "Prevalence (year = {:.2f})".format(state.t/26.)
    source.data["prevalence"] = state[:, 1] / state[:, :].sum(axis=-1)
    source.data["reff"] = params.beta * state[:, 0] / state[:, :].sum(axis=-1)

    time_ts_list.append(state.t/26.)
    prev_WPV_ts_list.append(100 * state[:, 1].sum() / state[:, :].sum())  # (prev in %)
    prev_OPV_ts_list.append(100 * state[:, 2].sum() / state[:, :].sum())  # (prev in %)
    # prev_ts_list.append(np.random.poisson(lam=state[:, 1].sum()/2000.))  # (downsampled case counts)
    # prev_ts_list.append(100 * (state[:, 1] > 0).sum() / len(state[:, 1]))  # (non-zero prevalence %)
    if len(time_ts_list) > 10*26:
        time_ts_list.popleft()
        prev_WPV_ts_list.popleft()
        prev_OPV_ts_list.popleft()
    ts_source.data = dict(time=list(time_ts_list),
                          prev_WPV=list(prev_WPV_ts_list),
                          prev_OPV=list(prev_OPV_ts_list))

effective_campaign_coverage = 0.3
coverage_slider = pn.widgets.FloatSlider(value=effective_campaign_coverage, start=0, end=1, step=0.05, name='effective campaign coverage')
def on_coverage_change(value):
    global effective_campaign_coverage
    effective_campaign_coverage = value
bound_coverage = pn.bind(on_coverage_change, value=coverage_slider)

callback_period = 100
callback = pn.state.add_periodic_callback(stream, callback_period)

speed_slider = pn.widgets.FloatSlider(value=callback_period, start=10, end=200, step=10, name='refresh rate (ms)')
def on_speed_change(value):
    callback.period = value
bound_speed = pn.bind(on_speed_change, value=speed_slider)

reset_button = pn.widgets.Button(name='Reset', button_type='primary')
def reset(event):
    global params, state
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
    time_ts_list.clear()
    prev_WPV_ts_list.clear()
    prev_OPV_ts_list.clear()
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
    "### Playback controls",
    pn.Row(speed_slider, bound_speed),
    pn.Row(reset_button, pause_button),
    pn.Row(OPV_campaign_button),
)

pn.template.MaterialTemplate(
    site="OPV Demo",
    title="Interactive Spatial Simulation",
    sidebar=[sliders],
    main=[pn.Row(prev_scatter, reff_scatter), prev_ts],
).servable()