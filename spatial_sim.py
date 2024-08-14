from dataclasses import dataclass

import numpy as np
from timer import timer

from mixing import init_gravity_diffusion


@dataclass
class Params:
    beta: float
    seasonality: float
    demog_scale: float
    mixing_scale: float
    distance_exponent: float


class ModelState(np.ndarray):

    def __new__(cls, input_array, t=0):        
        obj = np.asarray(input_array).view(cls)
        obj.t = t
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.t = getattr(obj, 't', None)


def init_state(settlements_df, params):
    
    population_s = settlements_df.population.astype(int)
    births_s = settlements_df.births.astype(int)

    N = population_s
    S = births_s * 2
    I = (S / 26. / 2.).astype(int)

    state = ModelState([S, I, N-S-I]).T

    params.population = population_s

    params.births = births_s
    params.biweek_avg_births = params.demog_scale * params.births / 26.
    params.biweek_death_prob = params.demog_scale * params.births / N / 26.

    params.mixing = init_gravity_diffusion(settlements_df, params.mixing_scale, params.distance_exponent)

    return state


def step_state(state, params):
    
        t = state.t

        expected = params.beta * (1 + params.seasonality * np.cos(2*np.pi*t/26.)) * np.matmul(params.mixing, state[:, 1])
        prob = 1 - np.exp(-expected/state.sum(axis=1))
        dI = np.random.binomial(n=state[:, 0], p=prob)

        state[:, 2] += state[:, 1]
        state[:, 1] = 0

        births = np.random.poisson(lam=params.biweek_avg_births)
        deaths = np.random.binomial(n=state, p=np.tile(params.biweek_death_prob, (3, 1)).T)

        state[:, 0] += births
        state -= deaths

        state[:, 1] += dI
        state[:, 0] -= dI

        state.t += 1

        assert np.all(state >= 0)  # TODO: verify ordering of updates (recover, infect, birth, death)


@timer("simulate", unit="ms")
def simulate(init_state, params, n_steps):
    
    state_timeseries = np.zeros((n_steps, *init_state.shape), dtype=int)

    state = init_state

    for i in range(n_steps):
        state_timeseries[i, :, :] = state
        step_state(state, params)
    
    return state_timeseries


if __name__ == "__main__":
    
    import logging
    logging.basicConfig()
    logging.getLogger('timer').setLevel(logging.DEBUG)

    import matplotlib.pyplot as plt 

    from settlements import parse_settlements
    settlements_df = parse_settlements()

    settlements_slice = slice(None, None)  # all settlements
    # settlements_slice = slice(None, 20)  # only biggest N
    # settlements_slice = slice(400, None)  # exclude biggest N

    biweek_steps = 26 * 20
    params = Params(
         beta=32, seasonality=0.16, demog_scale=1.0, 
         mixing_scale=0.002, distance_exponent=1.5)
    print(params)

    init_state = init_state(settlements_df.iloc[settlements_slice, :], params)
    states = simulate(init_state, params, n_steps=biweek_steps)

    # ========

    # Characterize fraction of time with infections present as function of population size:
    # presence_tsteps = (states[:, :, 1] > 0).sum(axis=0)
    # plt.scatter(settlements_df.population[settlements_slice], presence_tsteps / 26.)
    # ax = plt.gca()
    # ax.set(xscale='log', xlabel='population', ylabel='years with infections present')

    # --------

    # Visualize characteristic time series of single locations:
    # test_ix = 5  # a large city to see some endemic dynamics
    # # test_ix = 200  # a medium city to see some extinction dynamics + frequent reintroductions
    # # test_ix = 800  # a smaller village to see some extinction + random reintroduction dynamics
    # print(settlements_df.iloc[test_ix])
    # test_states = states[:, test_ix, :]  # (time, location, SIR)

    # from plotting import plot_timeseries
    # plot_timeseries(test_states)
    # from wavelet import plot_wavelet_spectrum
    # plot_wavelet_spectrum(test_states[:, 1])  # infecteds

    # --------

    # Spatial correlations in phase difference:
    # Emulating https://www.nature.com/articles/414716a#Sec4 and/or Xia et al. (2004)
    # and https://github.com/krosenfeld-IDM/sandbox-botorch/blob/main/laser/london/analyze.py#L68

    from wavelet import get_phase_diffs
    from mixing import pairwise_haversine
    ref_name = "London"
    sdf = settlements_df.iloc[settlements_slice, :]
    ref_ix = sdf.index.get_loc(ref_name)
    phase_diffs = get_phase_diffs(
        states[:, :, 1], ref_ix,
        period_range=(1.5, 2.5), timestep_range=slice(0*26, 2*26))
    distances_km = pairwise_haversine(sdf)[ref_ix]

    fig, ax = plt.subplots()
    ax.scatter(distances_km, phase_diffs, s=0.1*np.sqrt(sdf.population), alpha=0.4, c='gray')
    ax.set(xlabel="distance from %s (km)" % ref_name, ylabel="phase difference (radians)")

    # Linear fit to phase-difference vs. distance
    dist_threshold = 60
    ind = np.where(np.logical_and(distances_km < dist_threshold, distances_km > 0))
    polyfit1 = np.polyfit(distances_km[ind], phase_diffs[ind], 1)
    print("slope = %0.4g" % polyfit1[0])
    ax.plot(distances_km[ind], np.polyval(polyfit1, distances_km[ind]), c='black')

    # Spot-check time series in small region to understand bulk phase-difference features
    fig, ax = plt.subplots()
    ind = np.where(distances_km < dist_threshold)
    import pandas as pd
    pd.DataFrame(states[:, ind, 1].squeeze()).plot(ax=ax, legend=False, color='gray', alpha=0.1)
    ax.set(title="within %dkm of London" % dist_threshold, xlabel="time (biweeks)", ylabel="infecteds")


    # --------

    # # Animate spatial dynamics on map:
    # from plotting import plot_animation
    # ani = plot_animation(
    #     states, 
    #     settlements_df.iloc[settlements_slice, :], 
    #     # 'figures/ew_spatial_animation.gif'
    # )

    plt.show()