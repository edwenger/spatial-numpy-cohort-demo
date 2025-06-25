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
    
    params.population = settlements_df.population.astype(int)
    params.births = settlements_df.births.astype(int)

    # GRID3 under-1 estimates result in some settlements with ~20% crude birth rates (!)
    birth_rate = settlements_df.births / settlements_df.population
    max_birth_rate = 0.05

    # birth_rate = np.clip(birth_rate, 0, max_birth_rate)  # clip to max value
    birth_rate = np.clip(birth_rate, max_birth_rate, max_birth_rate)  # assign same value everywhere

    params.births = (birth_rate * settlements_df.population).astype(int)

    params.biweek_avg_births = params.demog_scale * params.births / 26.
    params.biweek_death_prob = params.demog_scale * params.births / params.population / 26.

    params.mixing = init_gravity_diffusion(settlements_df, params.mixing_scale, params.distance_exponent)

    # initialize roughly near equilibrium
    # S = (params.population / params.beta).astype(int)
    # I = params.biweek_avg_births.astype(int)

    # initialize near cessation with variation by state, LGA, settlement-population random effects
    states = settlements_df.adm2_name.unique()
    init_susc_by_LGA = {s: np.random.uniform(0.02, 0.08) for s in states}
    init_susc = settlements_df.adm2_name.map(init_susc_by_LGA)
    init_susc[settlements_df.adm1_name == "Borno"] += 0.1
    init_susc[settlements_df.population > 1e5] = 0.02
    init_susc[settlements_df.population < 1e3] = 0.1
    init_susc[settlements_df.adm2_name == "Geidam"] = 0.08
    # init_susc += np.random.uniform(-0.03, 0.03, size=init_susc.size)
    init_susc = np.clip(init_susc, 0, 0.2)

    S = (params.population * init_susc).astype(int)  # TODO: expose as free parameters
    I = np.zeros_like(S)
    state = ModelState([S, I, params.population-S-I]).T
    
    return state


def step_state(state, params):
    
        t = state.t

        beta = params.beta * (1 + params.seasonality * np.cos(2*np.pi*t/26.))

        expected = beta * np.matmul(params.mixing, state[:, 1])
        prob = 1 - np.exp(-expected/state.sum(axis=1))
        dI = np.random.binomial(n=state[:, 0], p=prob)

        state[:, -1] += state[:, 1]  # recovered I
        state[:, 1] = 0

        state[:, 1] += dI
        state[:, 0] -= dI

        births = np.random.poisson(lam=params.biweek_avg_births)
        deaths = np.random.binomial(n=state, p=np.tile(params.biweek_death_prob, (3, 1)).T)

        state[:, 0] += births
        state -= deaths

        state.t += 1

        assert np.all(state >= 0)  # TODO: verify ordering of updates (recover, infect, birth, death)


@timer("simulate", unit="ms")
def simulate(init_state, params, n_steps, settlements_df=None):
    state_timeseries = np.zeros((n_steps, *init_state.shape), dtype=int)

    state = init_state

    # Find the index of the most populated settlement in Geidam, Yobe
    outbreak_idx = None
    if settlements_df is not None:
        mask = (settlements_df["adm2_name"] == "Geidam")
        if mask.any():
            masked_settlements = settlements_df[mask]
            outbreak_idx = masked_settlements["population"].idxmax()
            # Convert to positional index if index is not default integer
            outbreak_idx = settlements_df.index.get_loc(outbreak_idx)

    for i in range(n_steps):
        state_timeseries[i, :, :] = state

        # Introduce as post-cessation OBR
        if i == 26 * 1 and outbreak_idx is not None:
            if state[outbreak_idx, 0] >= 10:
                state[outbreak_idx, 0] -= 10
                state[outbreak_idx, 1] += 10

        step_state(state, params)
    
    return state_timeseries


def plot_admin2_prevalence(states, settlements_df, adm1_names=["Jigawa", "Yobe"]):
    """
    Plot time series of sum(I)/sum(N) for each admin-2 in the specified adm1_names.
    """

    n_axs = len(adm1_names)

    fig, axs = plt.subplots(n_axs, 1, figsize=(10, 6), sharex=True, sharey=True)
    for ax, adm1_name in zip(axs, adm1_names):

        # settlements_df must have a default integer index matching states
        mask = settlements_df["adm1_name"] == adm1_name
        admin2s = settlements_df.loc[mask, "adm2_name"].unique()

        for adm2 in admin2s:
            adm2_mask = (settlements_df["adm1_name"] == adm1_name) & (settlements_df["adm2_name"] == adm2)
            adm2_idxs = settlements_df.index[adm2_mask].to_numpy()
            # adm2_idxs are now positional indices matching states' axis 1
            I = states[:, adm2_idxs, 1].sum(axis=1)
            ax.plot(I, label=adm2)
        # plt.xlabel("Time step")
        # plt.ylabel("Infected (I)")
        ax.set_title(f"Prevalence time series for admin-2s in {adm1_name}")
        # plt.legend(title="admin-2")

    fig.set_tight_layout(True)


def plot_admin2_bubble_prevalence(states, settlements_df, adm1_names=["Jigawa", "Yobe"]):
    """
    Plot time series of sum(I)/sum(N) for each admin-2 in the specified adm1_names.
    """
    n_axs = len(adm1_names)
    fig, axs = plt.subplots(n_axs, 1, figsize=(6, 8), sharex=True)
    for ax, adm1_name in zip(axs, adm1_names):

        # settlements_df must have a default integer index matching states
        mask = settlements_df["adm1_name"] == adm1_name
        admin2s = settlements_df.loc[mask, "adm2_name"].unique()

        for i, adm2 in enumerate(admin2s):
            adm2_mask = (settlements_df["adm1_name"] == adm1_name) & (settlements_df["adm2_name"] == adm2)
            adm2_idxs = settlements_df.index[adm2_mask].to_numpy()
            # adm2_idxs are now positional indices matching states' axis 1
            I = states[:, adm2_idxs, 1].sum(axis=1)
            ax.scatter([(ts/26. + 2016) for ts in range(len(I))], [i]*len(I), s=np.sqrt(I)*10, label=adm2, alpha=0.5)
        # plt.xlabel("Time step")
        # plt.ylabel("Infected (I)")
        ax.set_yticks(range(len(admin2s)))
        ax.set_yticklabels(admin2s)
        ax.set_title(f"Prevalence time series for admin-2s in {adm1_name}")
        # plt.legend(title="admin-2")

    fig.set_tight_layout(True)


if __name__ == "__main__":
    
    import logging
    logging.basicConfig()
    logging.getLogger('timer').setLevel(logging.DEBUG)

    import pandas as pd
    import matplotlib.pyplot as plt 

    from settlements import parse_grid3_settlements

    adm1_names = ["Jigawa"]#, "Kano", "Katsina", "Kaduna"]
    adm1_names += ["Yobe", "Borno", "Bauchi", "Gombe"]
    nigeria_settlements_df = parse_grid3_settlements(adm1_names, "Nigeria")

    adm1_names = ["Zinder"]#, "Maradi", "Diffa"]
    niger_settlements_df = parse_grid3_settlements(adm1_names, country="Niger")

    settlements_df = pd.concat([nigeria_settlements_df])#, niger_settlements_df])

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

    biweek_steps = 26 * 3
    params = Params(
         beta=8, seasonality=0.1, demog_scale=0.8, 
         mixing_scale=0.003, distance_exponent=1.5)
    print(params)

    # settlements_df = adm2_centroids  # collapse onto admin-2 level spatial resolution of simulation
    settlements_df = settlements_df.reset_index(drop=True)  # ensure settlements_df has a default integer index matching states
    print(settlements_df.head())

    init_state_arr = init_state(settlements_df, params)
    states = simulate(init_state_arr, params, n_steps=biweek_steps, settlements_df=settlements_df)

    from plotting import plot_animation
    ani = plot_animation(
        states, 
        settlements_df,
        save_path='/Users/ewenger/Desktop/outbreak_spatial_animation.gif',
        params=params,
    )

    # Plot prevalence for all admin-2s in Jigawa
    # plot_admin2_prevalence(states, settlements_df, adm1_names=["Jigawa", "Yobe"])
    plot_admin2_bubble_prevalence(states, settlements_df, adm1_names=["Jigawa", "Yobe"])

    plt.show()