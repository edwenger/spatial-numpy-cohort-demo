import matplotlib.pyplot as plt
import numpy as np
from timer import timer

from settlements import parse_settlements
from single_sim import Params, init_state, simulate
from wavelet import get_max_wavelet_power


@timer("sweep", unit="s")
def sweep_dynamic_periodicity(settlements_df):

    biweek_steps = 26 * 20

    seasonalities = []
    demog_scales = []
    max_power_periods = []

    for seasonality in np.arange(0.02, 0.3, 0.02):
        for demog_scale in np.arange(0.6, 3.5, 0.1):

            settlement = settlements_df.loc["London"]

            params = Params(beta=32, seasonality=seasonality, demog_scale=demog_scale)
            # print(params)

            state = init_state(settlement, params)
            states = simulate(state, params, n_steps=biweek_steps, keep_alive=True)

            period = get_max_wavelet_power(states[:, 1])
            # print(seasonality, demog_scale, period)
            seasonalities.append(seasonality)
            demog_scales.append(demog_scale)
            max_power_periods.append(period)

    plt.scatter(seasonalities, demog_scales, c=max_power_periods, cmap="viridis", vmin=0.8, vmax=3.2)
    plt.gca().set(xlabel="seasonality", ylabel="demog_scale")
    plt.colorbar()


if __name__ == '__main__':
    
    import logging
    logging.basicConfig()
    logging.getLogger('timer.sweep').setLevel(logging.DEBUG)

    settlements_df = parse_settlements()

    sweep_dynamic_periodicity(settlements_df)

    plt.show()