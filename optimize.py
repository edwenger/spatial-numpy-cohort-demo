import numpy as np

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback

from settlements import parse_settlements
from single_sim import Params, init_state, simulate
from wavelet import get_max_wavelet_power


def objective(trial, ref, settlement, biweek_steps=26*20):
    
    # TODO: Extend to include artifacts as in https://github.com/edwenger/emod-demo/blob/master/examples/optimize.py

    seasonality = trial.suggest_float("seasonality", 0.02, 0.3)
    mlflow.log_param("seasonality", seasonality)

    demog_scale = trial.suggest_float("demog_scale", 0.6, 3.5)
    mlflow.log_param("demog_scale", demog_scale)

    params = Params(beta=32, seasonality=seasonality, demog_scale=demog_scale)

    state = init_state(settlement, params)
    states = simulate(state, params, n_steps=biweek_steps, keep_alive=True)

    period = get_max_wavelet_power(states[:, 1])

    return np.abs(period - ref)


if __name__ == '__main__':
 
    """
    To run this test calibration study:
    > python optimize.py
    
    To explore optuna-tracked meta-data DB + artifacts:
    > optuna-dashboard sqlite:///optuna.db --artifact-dir optuna-artifacts

    To explore mlflow-tracked details:
    > mlflow ui --backend-store-uri sqlite:///mlruns.db
    """

    settlements_df = parse_settlements()
    settlement = settlements_df.loc["London"]

    mlflc = MLflowCallback(
        tracking_uri='sqlite:///mlruns.db',
        metric_name="my metric score")

    @mlflc.track_in_mlflow()
    def periodicity_calibration(trial):
        return objective(trial, ref=2.0, settlement=settlement, biweek_steps=26*20)

    study_name = "test-optuna"  # unique identifier
    storage_name = "sqlite:///optuna.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name,
                                pruner=optuna.pruners.MedianPruner(),
                                direction='minimize',
                                load_if_exists=True)

    study.optimize(periodicity_calibration, n_trials=200, callbacks=[mlflc])  # increase n_trials and/or re-run to append more trials