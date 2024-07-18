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


def init_state(settlements_df, params):
    
    population_s = settlements_df.population.astype(int)
    births_s = settlements_df.births.astype(int)

    N = population_s
    S = births_s * 2
    I = (S / 26. / 2.).astype(int)

    state = np.array([S, I, N-S-I]).T

    params.biweek_avg_births = params.demog_scale * births_s / 26.
    params.biweek_death_prob = params.demog_scale * births_s / N / 26.

    params.mixing = init_gravity_diffusion(settlements_df, params.mixing_scale, params.distance_exponent)

    return state


def step_state(state, params, t):
    
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


@timer("simulate", unit="ms")
def simulate(init_state, params, n_steps):
    
    state_timeseries = np.zeros((n_steps, *init_state.shape), dtype=int)

    state = init_state

    for t in range(n_steps):
        state_timeseries[t, :, :] = state
        step_state(state, params, t)
    
    return state_timeseries


if __name__ == "__main__":
    
    import logging
    logging.basicConfig()
    logging.getLogger('timer').setLevel(logging.DEBUG)

    import matplotlib.pyplot as plt 

    from settlements import parse_settlements
    settlements_df = parse_settlements()

    biweek_steps = 26 * 20
    params = Params(beta=32, seasonality=0.15, demog_scale=1.0,
                    mixing_scale=0.001, distance_exponent=1.5)
    print(params)

    n_settlements = None
    init_state = init_state(settlements_df.iloc[:n_settlements, :], params)
    states = simulate(init_state, params, n_steps=biweek_steps)

    # --------

    presence_tsteps = (states[:, :, 1] > 0).sum(axis=0)
    plt.scatter(settlements_df.population[:n_settlements], presence_tsteps / 26.)
    ax = plt.gca()
    ax.set(xscale='log', xlabel='population', ylabel='years with infections present')

    # --------

    # test_ix = 800  # a smaller village to see some extinction + reintroduction dynamics
    # print(settlements_df.iloc[test_ix])
    # test_states = states[:, test_ix, :]  # (time, location, SIR)

    # from plotting import plot_timeseries, plot_wavelet_spectrum
    # plot_timeseries(test_states)
    # plot_wavelet_spectrum(test_states[:, 1])  # infecteds

    # --------

    # TODO: Something like https://www.nature.com/articles/414716a#Sec4 and/or Xia et al. (2004)

    # --------

    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots()

    scat = ax.scatter(
        settlements_df.Long, 
        settlements_df.Lat, 
        s=0.1*np.sqrt(settlements_df.population), 
        c=states[0, :, 1] / states[0, :, :].sum(axis=-1), 
        cmap="Reds", norm=LogNorm(vmin=1e-4, vmax=0.01), alpha=0.5)

    def animate(i):
         ax.set_title("{:.2f} years".format(i/26.))
         scat.set_array(states[i, :, 1] / states[i, :, :].sum(axis=-1))
         return scat,

    ani = animation.FuncAnimation(fig, animate, frames=states.shape[0]-1, interval=50, blit=False)

    # To save the animation using Pillow as a gif
    # writer = animation.PillowWriter(fps=15,
    #                                 metadata=dict(artist='Me'),
    #                                 bitrate=1800)
    # ani.save('figures/ew_spatial_animation.gif', writer=writer)

    plt.show()