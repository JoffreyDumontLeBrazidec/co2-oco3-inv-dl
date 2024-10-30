# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import itertools
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional

import joblib
import matplotlib.pyplot as plt
import matplotlib_functions as mympf
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xarray as xr
from cmcrameri import cm
from hydra import compose, initialize
from hydra.utils import call, instantiate
from icecream import ic
from omegaconf import OmegaConf
from scipy.optimize import differential_evolution
from sklearn import preprocessing

from cfg.convert_cfg_to_yaml import save_myyaml_from_mycfg
from Data import Data_eval
from include.loss import pixel_weighted_cross_entropy
from models.preprocessing import (
    CloudsLayer,
    ConditionalNoiseLayer,
    TrainingTimeNormalization,
)


def get_cloud_layer(dir_res: str) -> tf.keras.models.Model:
    """Get adapted cloud layer for evaluation."""
    cloud_layer = tf.keras.models.load_model(
        os.path.join(dir_res, "cloud_layer.keras"),
        custom_objects={"CloudsLayer": CloudsLayer},
    )
    assert isinstance(cloud_layer, tf.keras.models.Model)
    return cloud_layer


def get_norm_layer(dir_res: str) -> tf.keras.models.Model:
    """Get adapted norm layer for evaluation."""
    norm_layer = tf.keras.models.load_model(
        os.path.join(dir_res, "norm_layer.keras"),
        custom_objects={"TrainingTimeNormalization": TrainingTimeNormalization},
    )
    assert isinstance(norm_layer, tf.keras.models.Model)
    return norm_layer


def get_inversion_model(
    dir_res: str,
    name_w: str = "w_last.h5",
    optimiser: str = "adam",
    loss=tf.keras.losses.MeanAbsoluteError(),
):
    """Get inversion neural network model."""
    from efficientnet.tfkeras import (
        EfficientNetB4,
    )  # to import "FixedDropout" and fix keras issues

    model = tf.keras.models.load_model(
        os.path.join(dir_res, name_w),
        compile=False,
    )
    assert isinstance(model, tf.keras.models.Model)
    model.compile(optimiser, loss=loss)
    return model


def get_data_for_inversion(
    dir_res: str,
    path_eval_nc: str,
    cloud_threshold: float = None,
) -> Data_eval:
    """Prepare Data_eval object with name_dataset."""

    cfg = OmegaConf.load(os.path.join(dir_res, "config.yaml"))

    data = Data_eval(path_eval_nc)

    data.prepare_input(
        cfg.data.input.chan_0,
        cfg.data.input.chan_1,
        cfg.data.input.chan_2,
        cfg.data.input.chan_3,
        cfg.data.input.chan_4,
        clouds_threshold=(
            cloud_threshold
            if cloud_threshold is not None
            else cfg.data.input.clouds_threshold
        ),
        norm_model=get_norm_layer(dir_res),
        cloud_model=get_cloud_layer(dir_res),
    )

    data.prepare_output_inversion(cfg.data.output.N_emissions)
    return data


def print_inv_metrics(metrics, message=""):
    """Print MAE and MAPE inv metrics."""
    print(message)
    print("MAE", np.mean(metrics["mae"]), np.median(metrics["mae"]))
    print("MAPE", np.mean(metrics["mape"]), np.median(metrics["mape"]))


def get_inv_metrics_pred_from_ensemble_paths(
    list_paths_model, path_ds_nc, name_w="w_best.h5", data_similar=True
):
    """Get inv metrics from ensemble of paths to models."""
    data = get_data_for_inversion(list_paths_model[0], path_ds_nc)
    pred = tf.zeros_like(data.y.eval, np.float32)
    for path_model in list_paths_model:
        if not data_similar:
            data = get_data_for_inversion(path_model, path_ds_nc)
        model_eval = InversionModelEvaluation(data=data, dir_res=path_model)
        pred += tf.convert_to_tensor(model_eval.pred, np.float32)
        print_inv_metrics(model_eval.get_metrics(), message=path_model)
    pred = pred / len(list_paths_model)
    ensemble_model_eval = InversionModelEvaluation(data=data, pred=pred)
    metrics = ensemble_model_eval.get_metrics()
    print_inv_metrics(metrics, message="\nEnsemble")
    return {"metrics": metrics, "pred": pred, "data": data}


@dataclass
class InversionModelEvaluation:
    dir_res: Optional[str] = None
    path_eval_nc: Optional[str] = None
    model: Optional[tf.keras.models.Model] = None
    data: Optional[Data_eval] = None
    pred: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    x: Optional[tf.Tensor] = field(default=None, init=False)

    def __post_init__(self):
        if self.path_eval_nc and self.dir_res:
            self.data = get_data_for_inversion(self.dir_res, self.path_eval_nc)
        if self.dir_res:
            self.model = get_inversion_model(self.dir_res)
        if self.data:
            self.y = self.data.y.eval
        if self.model and self.data:
            self.x = tf.convert_to_tensor(self.data.x.eval, np.float32)
            self.pred = self.model.predict(self.x, verbose=0)
        if self.pred is None or self.y is None:
            raise ValueError("Invalid input parameters")

    def get_metrics(self, metrics_to_return: list[str] = ["mae", "mape"]):
        """Function to return metrics."""
        results = {}

        if metrics_to_return is None or "mae" in metrics_to_return:
            f_mae = tf.keras.losses.MeanAbsoluteError(
                reduction=tf.losses.Reduction.NONE
            )
            results["mae"] = f_mae(self.y, self.pred)

        if metrics_to_return is None or "mape" in metrics_to_return:
            f_mape = tf.keras.losses.MeanAbsolutePercentageError(
                reduction=tf.losses.Reduction.NONE
            )
            results["mape"] = f_mape(self.y, self.pred)

        if metrics_to_return is None or "raw_diff" in metrics_to_return:
            y_np = np.squeeze(np.array(self.y))
            pred_np = np.squeeze(np.array(self.pred))
            results["raw_diff"] = pred_np - y_np

        if metrics_to_return is None or "raw_relative_diff" in metrics_to_return:
            y_np = np.squeeze(np.array(self.y))
            pred_np = np.squeeze(np.array(self.pred))
            results["raw_relative_diff"] = (pred_np - y_np) / y_np

        self.results = results
        return results


def draw_idx(
    all_scores: np.ndarray, ds: xr.Dataset, interval: list = [], idx: int = -1
) -> list:
    """Draw a specific field/plume/emiss index to plot given potential interval."""
    if idx > 0:
        pass
    elif interval:
        quantiles = np.quantile(all_scores, interval)
        idx = np.random.choice(
            np.argwhere(
                (quantiles[0] < all_scores) & (all_scores < quantiles[1])
            ).flatten()
        )
    else:
        idx = int(np.random.uniform(0, all_scores.shape[0]))

    ds_idx = ds.isel(idx_img=idx)
    print(
        f"nwbce: {all_scores[idx]}\nidx: {idx}\norigin: {ds_idx.point_source.values}\ntime: {ds_idx.time.values}\nemiss: {ds_idx.emiss.values}\n"
    )
    return [idx, ds_idx]


def get_summary_histo_inversion(
    model: tf.keras.Model, data: Data_eval, dir_save: str = "None", bins=30
) -> None:
    """Get various histograms summing up the inversion results."""
    inv_eval = InversionModelEvaluation(model=model, data=data)
    metrics = inv_eval.get_metrics(["mae", "mape", "raw_diff"])
    y = data.y.eval[:, -1]

    df_mae = pd.DataFrame({"loss": metrics["mae"], "method": "CNN"})
    df_mape = pd.DataFrame({"loss": metrics["mape"], "method": "CNN"})
    df_raw_diff = pd.DataFrame({"loss": metrics["raw_diff"], "method": "CNN"})

    pred = np.squeeze(model.predict(tf.convert_to_tensor(data.x.eval, np.float32)))
    df_emiss = pd.concat(
        [
            pd.DataFrame({"emiss": y, "origin": "truth"}),
            pd.DataFrame({"emiss": pred, "origin": "prediction"}),
        ]
    )

    N_rows = 2
    N_cols = 2
    mympf.setMatplotlibParam()
    plt.viridis()
    axs = mympf.set_figure_axs(
        N_rows,
        N_cols,
        wratio=0.35,
        hratio=0.75,
        pad_w_ext_left=0.25,
        pad_w_ext_right=0.25,
        pad_w_int=0.3,
        pad_h_ext=0.3,
        pad_h_int=0.35,
    )

    sns.histplot(
        data=df_mae,
        x="loss",
        hue="method",
        kde=True,
        color="firebrick",
        alpha=0.2,
        bins=bins,
        ax=axs[0],
    )
    axs[0].set_xlabel("Mean absolute error")
    sns.histplot(
        data=df_mape,
        x="loss",
        hue="method",
        kde=True,
        color="firebrick",
        bins=bins,
        alpha=0.2,
        ax=axs[1],
    )
    axs[1].set_xlabel("Mean absolute percentage error")
    sns.histplot(
        data=df_emiss,
        x="emiss",
        hue="origin",
        kde=True,
        color="firebrick",
        bins=bins,
        alpha=0.2,
        ax=axs[2],
    )
    axs[2].set_xlabel("Emission rate")
    sns.histplot(
        data=df_raw_diff,
        x="loss",
        hue="method",
        kde=True,
        color="firebrick",
        bins=bins,
        alpha=0.2,
        ax=axs[3],
    )
    axs[3].axvline(0, color="black", lw=0.5)
    axs[3].set_xlabel("Raw difference")

    for i_ax, ax in enumerate(axs):
        ax.set_yticklabels([])

    if dir_save != "None":
        plt.savefig(os.path.join(dir_save, "summary_inv.png"))


@dataclass
class InversionPlotter:
    inv_eval: InversionModelEvaluation
    cols_to_plot: list[int] = field(default_factory=lambda: [0, 1])
    winds_as_input: bool = True
    col_u_wind: int = 2
    col_v_wind: int = 1
    N_rows: int = 4

    def plot_examples(self, list_idx=[], list_ds_idx=[], interval=[]):
        list_idx, list_ds_idx = self._setup_indices(list_idx, list_ds_idx, interval)
        self.axs = self._configure_plotting_parameters(list_idx)
        self._plot_data(list_idx)
        self._add_annotations(list_idx, list_ds_idx)

    def _setup_indices(self, list_idx=[], list_ds_idx=[], interval=[0, 1]):
        assert self.inv_eval.data is not None
        ic(list_idx)
        if not np.array(list_idx).size:
            list_idx, list_ds_idx = zip(
                *[
                    draw_idx(
                        self.inv_eval.results["mae"],
                        self.inv_eval.data.ds,
                        interval=interval,
                    )
                    for _ in range(self.N_rows)
                ]
            )
        else:
            list_ds_idx = []
            for idx in list_idx:
                list_ds_idx.append(self.inv_eval.data.ds.isel(idx_img=idx))
        return list_idx, list_ds_idx

    def _configure_plotting_parameters(self, list_idx):
        assert self.inv_eval.data is not None
        if np.max(self.cols_to_plot) > self.inv_eval.data.x.eval.shape[0]:
            raise ValueError("Maximum column index exceeds the data shape.")

        mympf.setMatplotlibParam()
        plt.viridis()
        return mympf.set_figure_axs(
            len(list_idx),
            len(self.cols_to_plot),
            wratio=0.35,
            hratio=0.75,
            pad_w_ext_left=0.25,
            pad_w_ext_right=0.25,
            pad_w_int=0.001,
            pad_h_ext=0.2,
            pad_h_int=0.25,
        )

    def _plot_data(self, list_idx):
        assert self.inv_eval.data is not None
        self.N_wind_points = 8
        self.Ny = int(self.inv_eval.data.x.eval.shape[1])
        self.Nx = int(self.inv_eval.data.x.eval.shape[2])
        self.ims = [None] * (len(list_idx) * len(self.cols_to_plot))
        self.caxs = [None] * (len(list_idx) * len(self.cols_to_plot))
        self.cbars = [None] * (len(list_idx) * len(self.cols_to_plot))

        for ax in self.axs:
            ax.set_xticks([int(self.Ny / 4), int(self.Ny / 2), int(self.Ny * 3 / 4)])
            ax.set_yticks([int(self.Nx / 4), int(self.Nx / 2), int(self.Nx * 3 / 4)])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        for i, idx in enumerate(list_idx):
            x_idx = self.inv_eval.data.x.eval[idx]
            for i_col in range(len(self.cols_to_plot)):
                self._plot_column(i, i_col, x_idx)

    def _plot_column(self, i, i_col, x_idx):
        i_ax = i * len(self.cols_to_plot) + i_col
        if i_col == 1 and self.winds_as_input:
            self._plot_wind_vector(i_ax, x_idx)
        else:
            self._plot_image_data(i_ax, x_idx, i_col)

    def _plot_wind_vector(self, i_ax, x_idx):
        u, v = self._extract_wind_components(x_idx)
        self.ims[i_ax] = self.axs[i_ax].quiver(
            np.arange(0, self.Nx, self.N_wind_points),
            np.arange(0, self.Ny, self.N_wind_points),
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=0.1,
        )

    def _extract_wind_components(self, x_idx):
        u = np.squeeze(x_idx[:, :, self.col_u_wind])
        u = u[np.arange(0, u.shape[0], self.N_wind_points), :][
            :, np.arange(0, u.shape[1], self.N_wind_points)
        ]
        v = np.squeeze(x_idx[:, :, self.col_v_wind])
        v = v[np.arange(0, v.shape[0], self.N_wind_points), :][
            :, np.arange(0, v.shape[1], self.N_wind_points)
        ]
        return u, v

    def _plot_image_data(self, i_ax, x_idx, i_col):
        self.ims[i_ax] = self.axs[i_ax].imshow(
            np.squeeze(x_idx[:, :, self.cols_to_plot[i_col]]),
            origin="lower",
        )
        self._add_colorbar(i_ax)

    def _add_colorbar(self, i_ax):
        self.caxs[i_ax] = self.axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        self.cbars[i_ax] = plt.colorbar(
            self.ims[i_ax], self.caxs[i_ax], orientation="vertical"
        )

    def _add_annotations(self, list_idx, list_ds_idx):
        assert self.inv_eval.data is not None
        assert self.inv_eval.pred is not None
        for i, idx in enumerate(list_idx):
            pd_t_idx = pd.Timestamp(list_ds_idx[i].time.values)
            self.axs[i * len(self.cols_to_plot)].set_ylabel(
                f"[{pd_t_idx.month:02d}-{pd_t_idx.day:02d} {pd_t_idx.hour:02d}:00]"
            )
            mae_value = self.inv_eval.results["mae"][idx].numpy()
            mape_value = self.inv_eval.results["mape"][idx].numpy()
            truth_value = self.inv_eval.data.y.eval[idx][0]
            pred_value = self.inv_eval.pred[idx][0]
            self.axs[1 + i * len(self.cols_to_plot)].set_title(
                f"time: {pd_t_idx},   "
                f"mae: {mae_value:.3f},   "
                f"mape: {mape_value:.3f},   "
                f"truth: {truth_value:.3f},   "
                f"pred: {pred_value:.3f},   "
            )


def channel_permutation_importance(dir_model: str, path_eval_nc: str, size_max_combi=1):
    """Get importances of each channel for a model and a dataset at path."""
    data = get_data_for_inversion(
        dir_model,
        path_eval_nc,
    )

    model = get_inversion_model(dir_model, name_w="w_best.h5")
    X = data.x.eval
    y = data.y.eval
    loss_function = tf.keras.losses.MeanAbsolutePercentageError()

    baseline_predictions = model(X)
    baseline_loss = loss_function(y, baseline_predictions).numpy()
    print("baseline", baseline_loss)

    my_list = [i for i in range(X.shape[-1])]
    combinations = []
    for r in range(1, size_max_combi + 1):
        for combo in itertools.combinations(my_list, r):
            combinations.append(combo)

    importances = np.zeros(len(combinations))

    for i, perm in enumerate(combinations):
        X_permuted = np.copy(X)
        for p in perm:
            X_permuted[:, :, :, p] = np.random.permutation(X[:, :, :, p])

        # Compute predictions with permuted channel
        permuted_predictions = model(X_permuted)
        permuted_loss = loss_function(y, permuted_predictions).numpy()
        print("perm", perm, "permuted_loss", permuted_loss)

        # Compute feature importance for channel
        importances[i] = baseline_loss - permuted_loss

    return importances


def build_df_perf_inv(metrics):
    """Build dataframe of inversion performances."""

    keys = ["none", "seg_pred_no2", "no2", "cs"]
    labels = ["No additional input", "Segmentation", "NO2", "CSF"]

    mape_col, mae_col, rel_col, raw_col = (
        "Absolute relative error (%)",
        "Absolute error (Mt/yr)",
        "Relative error (%)",
        "Raw difference (Mt/yr)",
    )
    second_col = "Add. input:"
    df_mape, df_mae, df_rel, df_raw = [], [], [], []

    for key, label in zip(keys, labels):
        for df, col, key_df in zip(
            [df_mape, df_mae, df_rel, df_raw],
            [mape_col, mae_col, rel_col, raw_col],
            ["mape", "mae", "raw_relative_diff", "raw_diff"],
        ):
            df.append(pd.DataFrame({col: metrics[key][key_df], second_col: label}))

    df_mape = pd.concat(df_mape)
    df_mae = pd.concat(df_mae)
    df_rel = pd.concat(df_rel)
    df_raw = pd.concat(df_raw)

    df_mape[mape_col] = df_mape[mape_col].apply(lambda x: 200 if x > 200 else x)
    df_mae[mae_col] = df_mae[mae_col].apply(lambda x: 30 if x > 30 else x)
    df_rel[rel_col] = df_rel[rel_col].apply(lambda x: x * 100)
    df_rel[rel_col] = df_rel[rel_col].apply(lambda x: -150 if x < -150 else x)
    df_rel[rel_col] = df_rel[rel_col].apply(lambda x: 150 if x > 150 else x)

    def group_and_describe(df, col):
        df_groupby = df.groupby(second_col).describe()
        df_groupby = df_groupby.drop(
            [(col, s) for s in ["count", "std", "min", "mean", "max"]], axis=1
        )
        return df_groupby.loc[labels]

    desc_mape = group_and_describe(df_mape, mape_col)
    desc_mae = group_and_describe(df_mae, mae_col)
    desc_rel = group_and_describe(df_rel, rel_col)
    desc_raw = group_and_describe(df_raw, raw_col)

    return {
        "res": desc_mape.join(desc_mae).join(desc_rel).join(desc_raw),
        "df_mae": df_mae,
        "df_mape": df_mape,
        "df_rel": df_rel,
        "df_raw": df_raw,
    }


def integrated_gradients(model, img_tensor, baseline_tensor, num_steps=100):
    # Define the path from baseline to input as a straight line
    alphas = tf.linspace(start=0.0, stop=1.0, num=num_steps + 1)

    # Compute the gradients of the model's output with respect to the input
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
    grads = tape.gradient(predictions, img_tensor)

    # Compute the gradient at each point along the path
    interpolated_inputs = [
        (baseline_tensor + alpha * (img_tensor - baseline_tensor)) for alpha in alphas
    ]
    interpolated_inputs = tf.stack(interpolated_inputs)
    interpolated_inputs = tf.reshape(
        interpolated_inputs, [-1] + list(img_tensor.shape[1:])
    )
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        interpolated_predictions = model(interpolated_inputs)
    interpolated_grads = tape.gradient(interpolated_predictions, interpolated_inputs)

    # Approximate the integral using the trapezoidal rule
    avg_grads = tf.reduce_mean(interpolated_grads, axis=0)
    integrated_grads = tf.reduce_sum(avg_grads * (img_tensor - baseline_tensor), axis=0)
    return integrated_grads
