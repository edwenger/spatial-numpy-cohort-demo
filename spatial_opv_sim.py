from dataclasses import dataclass

import numpy as np

from mixing import init_gravity_diffusion


@dataclass
class Params:
    beta: float
    seasonality: float
    demog_scale: float
    mixing_scale: float
    distance_exponent: float
    opv_reversion_fraction: float  # fraction of I_opv that become I_wpv on transmission in biweek timestep
    opv_relative_beta: float  # governs I_opv infectiousness relative to I_wpv


class ModelState(np.ndarray):

    def __new__(cls, input_array, t=0):        
        obj = np.asarray(input_array).view(cls)
        obj.t = t
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.t = getattr(obj, 't', None)


def init_state(settlements_df, params: Params) -> ModelState:
    
    params.population = settlements_df.population.astype(int)
    params.births = settlements_df.births.astype(int)

    params.biweek_avg_births = params.demog_scale * params.births / 26.
    params.biweek_death_prob = params.demog_scale * params.births / params.population / 26.

    params.mixing = init_gravity_diffusion(settlements_df, params.mixing_scale, params.distance_exponent)

    S = (params.population / params.beta).astype(int)  # initialize roughly near equilibrium
    I_wpv = params.biweek_avg_births.astype(int)
    I_opv = np.zeros_like(I_wpv)
    state = ModelState([S, I_wpv, I_opv, params.population-S-I_wpv]).T
    
    return state


def step_state(state: ModelState, params: Params):
    
        t = state.t

        beta = params.beta * (1 + params.seasonality * np.cos(2*np.pi*t/26.))

        expected_wpv = beta * np.matmul(params.mixing, state[:, 1])
        expected_opv = params.opv_relative_beta * beta * np.matmul(params.mixing, state[:, 2])

        reverted_opv = expected_opv * params.opv_reversion_fraction
        expected_wpv += reverted_opv
        expected_opv -= reverted_opv
        expected = expected_wpv + expected_opv

        prob = 1 - np.exp(-expected/state.sum(axis=1))
        dI = np.random.binomial(n=state[:, 0], p=prob)
        p_opv = np.divide(expected_opv, expected, out=np.zeros_like(expected), where=expected!=0)
        dI_opv = np.random.binomial(n=dI, p=p_opv)
        dI_wpv = dI - dI_opv

        state[:, -1] += state[:, 1]  # recovered I_wpv
        state[:, -1] += state[:, 2]  # recovered I_opv
        state[:, 1] = 0
        state[:, 2] = 0

        state[:, 1] += dI_wpv
        state[:, 2] += dI_opv
        state[:, 0] -= dI

        births = np.random.poisson(lam=params.biweek_avg_births)
        deaths = np.random.binomial(n=state, p=np.tile(params.biweek_death_prob, (4, 1)).T)

        state[:, 0] += births
        state -= deaths

        state.t += 1

        assert np.all(state >= 0)  # TODO: verify ordering of updates (recover, infect, birth, death)
