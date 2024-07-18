from dataclasses import dataclass

import numpy as np
from timer import timer


@dataclass
class Params:
    beta: float
    seasonality: float
    demog_scale: float


def init_state(settlement_s, params):
    
    population = settlement_s.population.astype(int)
    births = settlement_s.births.astype(int)

    N = population
    S = births * 2
    I = int(S / 26. / 2.)

    state = np.array([S, I, N-S-I])

    params.biweek_avg_births = params.demog_scale * births / 26.
    params.biweek_death_prob = params.demog_scale * births / N / 26.

    return state


def step_state(state, params, t, keep_alive=False):
    
        expected = params.beta * (1 + params.seasonality * np.cos(2*np.pi*t/26.)) * state[1]
        prob = 1 - np.exp(-expected/state.sum())
        dI = np.random.binomial(n=state[0], p=prob)

        state[2] += state[1]
        state[1] = 0

        births = np.random.poisson(lam=params.biweek_avg_births)
        deaths = np.random.binomial(n=state, p=params.biweek_death_prob)

        state[0] += births
        state -= deaths

        if keep_alive:
             dI += 1

        state[1] += dI
        state[0] -= dI


@timer("simulate", unit="ms")
def simulate(init_state, params, n_steps, keep_alive=False):
    
    state_timeseries = np.zeros((n_steps, 3), dtype=int)

    state = init_state

    for t in range(n_steps):
        state_timeseries[t, :] = state
        step_state(state, params, t, keep_alive)
    
    return state_timeseries


if __name__ == "__main__":
    
    import logging
    logging.basicConfig()
    logging.getLogger('timer').setLevel(logging.DEBUG)

    from settlements import parse_settlements
    settlements_df = parse_settlements()
    settlement = settlements_df.loc["London"]

    biweek_steps = 26 * 20
    params = Params(beta=32, seasonality=0.06, demog_scale=1.5)
    print(params)

    init_state = init_state(settlement, params)
    states = simulate(init_state, params, n_steps=biweek_steps)

    from plotting import plot_timeseries, plot_wavelet_spectrum
    plot_timeseries(states)
    plot_wavelet_spectrum(states[:, 1])

    import matplotlib.pyplot as plt 
    plt.show()