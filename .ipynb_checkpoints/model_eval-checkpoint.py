# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import xarray as xr
from cmcrameri import cm
from hydra import compose, initialize
from omegaconf import OmegaConf
from scipy.optimize import differential_evolution
from sklearn import preprocessing

import nb_eval.matplotlib_functions as mympf
from cfg.convert_cfg_to_yaml import save_myyaml_from_mycfg
from Data import Data_eval
from include.loss import pixel_weighted_cross_entropy


# Segmentation
## Get functions
def get_scaler(
    dir_res: str, name_scaler: str = "scaler.save"
) -> preprocessing.StandardScaler:
    """Get scaler (fit on training data) for evaluation."""
    scaler = joblib.load(os.path.join(dir_res, name_scaler))
    return scaler


def get_data_for_segmentation(
    dir_res: str,
    path_eval_nc: str,
    ds_inds: dict = dict(),
    region_extrapol: bool = True,
) -> Data_eval:
    """Prepare Data object with name_dataset, scaler, and train or test mode."""

    if not os.path.exists(os.path.join(dir_res, "config.yaml")):
        save_myyaml_from_mycfg(dir_res)
    cfg = OmegaConf.load(os.path.join(dir_res, "config.yaml"))

    data = Data_eval(path_eval_nc, ds_inds, region_extrapol)
    data.prepare_input(
        get_scaler(dir_res),
        cfg.data.input.chan_0,
        cfg.data.input.chan_1,
        cfg.data.input.chan_2,
        cfg.data.input.chan_3,
        cfg.data.input.chan_4,
    )
    data.prepare_output_segmentation(
        curve=cfg.data.output.curve,
        min_w=cfg.data.output.min_w,
        max_w=cfg.data.output.max_w,
        param_curve=cfg.data.output.param_curve,
    )
    return data


def get_segmentation_model(
    dir_res: str,
    name_w: str = "weights_cp_best.h5",
    optimiser: str = "adam",
    loss=pixel_weighted_cross_entropy,
):
    """Get segmentation neural network model and compile it with pixel_weighted_cross_entropy loss."""
    model = tf.keras.models.load_model(os.path.join(dir_res, name_w), compile=False)
    model.compile(optimiser, loss=loss)
    return model


def get_wbce(y_test: tf.Tensor, pred_test: tf.Tensor) -> np.ndarray:
    """Get wbce given y_test and pred_test."""
    all_wbce = pixel_weighted_cross_entropy(y_test, pred_test, reduction=False)
    all_wbce = np.mean(all_wbce, axis=(1, 2))
    return all_wbce


def get_wbce_model_on_data(model: tf.keras.Model, data: Data_eval) -> np.ndarray:
    """Get wbce scores by segmentation model applied on data."""
    x = tf.convert_to_tensor(data.x.eval, np.float32)
    pred = tf.convert_to_tensor(model.predict(x), np.float32)
    y = tf.convert_to_tensor(data.y.eval, np.float32)
    all_wbce = get_wbce(y, pred)
    return all_wbce


def get_nwbce_model_on_data(model: tf.keras.Model, data: Data_eval) -> np.ndarray:
    """Get nwbce scores by segmentation model applied on data."""
    all_cnn_wbce = get_wbce_model_on_data(model, data)
    b1_all_wbce = get_b1_seg_wbce(tf.convert_to_tensor(data.y.eval, np.float32))
    all_cnn_nwbce = all_cnn_wbce / b1_all_wbce
    return all_cnn_nwbce


## neutral baseline functions


def get_mean_loss(params, y_test: tf.Tensor, pred_test: tf.Tensor) -> float:
    """Get mean wbce between y_test and pred_test given shift_to_proba with params."""
    proba_min, proba_max = params
    current_pred_test = shift_to_proba(pred_test, proba_max, proba_min)
    wbce = get_wbce(y_test, current_pred_test)
    return np.mean(wbce)


def shift_to_proba(y_pred, proba_max: np.float32, proba_min: np.float32):
    """Shift from a boolean to a probability map: 1 to proba_max, 0 to proba_min."""
    y_pred = np.where(y_pred == 1, proba_max, proba_min)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    return y_pred


def get_b1_seg_pred(y: tf.Tensor):
    """Get neutral reference/baseline (b1) segmentation predictions."""
    b1_pred = 0.1 * tf.ones(shape=y.shape)
    res = differential_evolution(
        get_mean_loss, args=(y, b1_pred), bounds=[[0, 1], [0, 1]]
    )
    [proba_min, proba_max] = res["x"]
    shifted_b1_pred = shift_to_proba(b1_pred, proba_max, proba_min)
    return shifted_b1_pred


def get_b1_seg_wbce(y: tf.Tensor) -> np.ndarray:
    """Get wbce for y and neutral reference/baseline (b1) segmentation predictions."""
    b1_pred = get_b1_seg_pred(y)
    wbce = get_wbce(y, b1_pred)
    return wbce


## plot functions
def draw_idx(
    cnn_nwbce: np.ndarray, ds_test: xr.Dataset, interval: list = [], idx: int = -1
):
    """Draw a specific field/plume index to plot given potential interval."""
    if idx > -1:
        idx = idx
    else:
        if interval:
            z = np.random.choice(
                cnn_nwbce[(interval[0] < cnn_nwbce) & (cnn_nwbce < interval[1])]
            )
            idx = np.where(cnn_nwbce == z)[0][0]
        else:
            idx = int(np.random.uniform(0, cnn_nwbce.shape[0]))

    ds_idx = ds_test.isel(idx_img=idx)
    print("nwbce:", cnn_nwbce[idx])
    print("idx", idx)
    print("origin:", ds_idx.point_source.values)
    print("time:", ds_idx.time.values)
    print("emiss:", ds_idx.emiss.values)

    return [idx, ds_idx]


def plot_segmentation_examples(
    data: Data_eval,
    cnn_nwbce: np.ndarray,
    model: tf.keras.Model,
    list_idx: list = [],
    list_ds_idx: list = [],
    proba_plume: float = 0,
    no2=False,
):
    """Plot examples of {input / truth / output} of the CNN model."""

    scaler = data.x.scaler

    if not list_idx:
        [idx0, ds_idx0] = draw_idx(cnn_nwbce, data.ds)
        [idx1, ds_idx1] = draw_idx(cnn_nwbce, data.ds)
        [idx2, ds_idx2] = draw_idx(cnn_nwbce, data.ds)
        list_idx = [idx0, idx1, idx2]
        list_ds_idx = [ds_idx0, ds_idx1, ds_idx2]

    N_idx = len(list_idx)

    N_cols = 3
    if proba_plume > 0:
        N_cols = N_cols + 1
    if no2:
        N_cols = N_cols + 1

    mympf.setMatplotlibParam()
    plt.viridis()
    axs = mympf.set_figure_axs(
        N_idx,
        N_cols,
        wratio=0.35,
        hratio=0.75,
        pad_w_ext_left=0.25,
        pad_w_ext_right=0.25,
        pad_w_int=0.001,
        pad_h_ext=0.2,
        pad_h_int=0.15,
    )

    ims = [None] * (N_idx * N_cols)
    caxs = [None] * (N_idx * N_cols)
    cbars = [None] * (N_idx * N_cols)

    Ny = int(data.x.eval.shape[1])
    Nx = int(data.x.eval.shape[2])
    for ax in axs:
        ax.set_xticks([0, int(Ny / 4), int(Ny / 2), int(Ny * 3 / 4), Ny])
        ax.set_yticks([0, int(Nx / 4), int(Nx / 2), int(Nx * 3 / 4), Nx])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for i, idx in enumerate(list_idx):

        x_idx = data.x.eval[idx]

        cur_row = 0

        i_ax = cur_row + i * N_cols
        cur_row += 1
        xco2_inv_scaled = scaler.inverse_transform(
            x_idx.reshape(-1, x_idx.shape[-1])
        ).reshape(x_idx.shape)[:, :, 0]
        ims[i_ax] = axs[i_ax].imshow(np.squeeze(xco2_inv_scaled), origin="lower")
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

        i_ax = cur_row + i * N_cols
        cur_row += 1
        ims[i_ax] = axs[i_ax].imshow(np.squeeze(data.y.eval[idx]), origin="lower")
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

        i_ax = cur_row + i * N_cols
        cur_row += 1
        ims[i_ax] = axs[i_ax].imshow(
            np.squeeze(model(tf.expand_dims(data.x.eval[idx], 0))[0]),
            vmin=0,
            vmax=1,
            cmap=cm.cork,
            origin="lower",
        )
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

        if no2:
            i_ax = cur_row + i * N_cols
            cur_row += 1
            no2_inv_scaled = scaler.inverse_transform(
                x_idx.reshape(-1, x_idx.shape[-1])
            ).reshape(x_idx.shape)[:, :, 1]
            ims[i_ax] = axs[i_ax].imshow(np.squeeze(no2_inv_scaled), origin="lower")
            caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
            cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

        if proba_plume > 0:
            i_ax = cur_row + i * N_cols
            cur_row += 1
            ims[i_ax] = axs[i_ax].imshow(
                np.where(
                    np.squeeze(model(tf.expand_dims(data.x.eval[idx], 0))[0])
                    > proba_plume,
                    1,
                    0,
                ),
                origin="lower",
            )
            caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
            cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

    list_pd_t_idx = []
    list_cnn_nwbce = []
    for idx, ds_idx in enumerate(list_ds_idx):
        list_pd_t_idx.append(pd.Timestamp(ds_idx.time.values))
        list_cnn_nwbce.append(cnn_nwbce[list_idx[idx]])

    for i, (pd_t_idx, loss_idx) in enumerate(
        zip(
            list_pd_t_idx,
            list_cnn_nwbce,
        )
    ):
        axs[i * N_cols].set_ylabel(
            f"[{pd_t_idx.month:02d}-{pd_t_idx.day:02d} {pd_t_idx.hour:02d}:00], n_wbce={loss_idx: .3f}"
        )

    axs[0].set_title("XCO2 field")
    axs[1].set_title("Targetted plume")
    axs[2].set_title("CNN segmentation")

    cbars[0].ax.set_title("[ppmv]")
    cbars[1].ax.set_title("[weight. bool.]")
    cbars[2].ax.set_title("[proba.]")


# Inversion
## Get functions


def get_data_for_inversion(
    dir_res: str,
    path_eval_nc: str,
) -> Data_eval:
    """Prepare Data_eval object with name_dataset, scaler."""

    cfg = OmegaConf.load(os.path.join(dir_res, "config.yaml"))

    data = Data_eval(path_eval_nc)
    data.prepare_input(
        cfg.data.input.chan_0,
        cfg.data.input.chan_1,
        cfg.data.input.chan_2,
        cfg.data.input.chan_3,
        cfg.data.input.chan_4,
    )
    data.prepare_output_inversion(cfg.data.output.N_emissions)
    return data


def get_inversion_model(
    dir_res: str,
    name_w: str = "w_best.h5",
    optimiser: str = "adam",
    loss=tf.keras.losses.MeanAbsoluteError(),
):
    """Get inversion neural network model."""
    from efficientnet.tfkeras import (
        EfficientNetB4,
    )  # to import "FixedDropout" and fix keras issues

    model = tf.keras.models.load_model(os.path.join(dir_res, name_w), compile=False)
    model.compile(optimiser, loss=loss)
    return model


def get_inv_metrics_model_on_data(model: tf.keras.Model, data: Data_eval) -> dict:
    """Get inversion scores by segmentation model applied on data."""
    x = tf.convert_to_tensor(data.x.eval, np.float32)
    pred = tf.convert_to_tensor(model.predict(x), np.float32)
    y = tf.convert_to_tensor(data.y.eval, np.float32)
    f_mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.losses.Reduction.NONE)
    all_mae = f_mae(y, pred)
    f_mape = tf.keras.losses.MeanAbsolutePercentageError(
        reduction=tf.losses.Reduction.NONE
    )
    all_mape = f_mape(y, pred)
    return {"mae": all_mae, "mape": all_mape}


def get_inv_mean_loss(data: Data_eval) -> dict:
    """Get mean inventory for inversion between y and pred."""
    y = tf.convert_to_tensor(data.y.eval, np.float32)
    pred = tf.math.reduce_mean(y) * tf.ones_like(y, np.float32)
    f_mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.losses.Reduction.NONE)
    all_mae = f_mae(y, pred)
    f_mape = tf.keras.losses.MeanAbsolutePercentageError(
        reduction=tf.losses.Reduction.NONE
    )
    all_mape = f_mape(y, pred)
    return {"mae": all_mae, "mape": all_mape}


def get_inv_metrics_m(model: tf.keras.Model, data: Data_eval) -> dict:
    """Get inversion scores by segmentation model applied on data."""
    x = tf.convert_to_tensor(data.x.eval, np.float32)
    pred = tf.convert_to_tensor(model.predict(x), np.float32)
    y = tf.convert_to_tensor(data.y.eval, np.float32)
    f_mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.losses.Reduction.NONE)
    all_mae = f_mae(y, pred)
    f_mape = tf.keras.losses.MeanAbsolutePercentageError(
        reduction=tf.losses.Reduction.NONE
    )
    all_mape = f_mape(y, pred)
    return {"mae": all_mae, "mape": all_mape}


## plot functions
def draw_idx(
    all_scores: np.ndarray, ds: xr.Dataset, interval: list = [], idx: int = -1
):
    """Draw a specific field/plume/emiss index to plot given potential interval."""
    if idx > -1:
        idx = idx
    else:
        if interval:
            z = np.random.choice(
                all_scores[(interval[0] < all_scores) & (all_scores < interval[1])]
            )
            idx = np.where(all_scores == z)[0][0]
        else:
            idx = int(np.random.uniform(0, all_scores.shape[0]))

    ds_idx = ds.isel(idx_img=idx)
    print("nwbce:", all_scores[idx])
    print("idx", idx)
    print("origin:", ds_idx.point_source.values)
    print("time:", ds_idx.time.values)
    print("emiss:", ds_idx.emiss.values)

    return [idx, ds_idx]


def get_summary_histo_inversion(
    model: tf.keras.Model, data: Data_eval, dir_save: str = "None"
) -> None:
    """Get various histograms summing up the inversion results."""
    metrics = get_inv_metrics_model_on_data(model, data)
    mean_metrics = get_inv_mean_loss(data)

    df_mae_1 = pd.DataFrame({"loss": metrics["mae"], "method": "CNN"})
    df_mae_2 = pd.DataFrame({"loss": mean_metrics["mae"], "method": "mean"})
    df_mae = pd.concat([df_mae_1, df_mae_2])

    df_mape_1 = pd.DataFrame({"loss": metrics["mape"], "method": "CNN"})
    df_mape_2 = pd.DataFrame({"loss": mean_metrics["mape"], "method": "mean"})
    df_mape = pd.concat([df_mape_1, df_mape_2])

    pred = np.squeeze(model.predict(tf.convert_to_tensor(data.x.eval, np.float32)))
    y = data.y.eval[:, -1]
    df_emiss_1 = pd.DataFrame({"emiss": y, "origin": "truth"})
    df_emiss_2 = pd.DataFrame({"emiss": pred, "origin": "prediction"})
    df_emiss = pd.concat([df_emiss_1, df_emiss_2])

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

    sns.kdeplot(
        data=df_mae,
        x="loss",
        common_norm=True,
        hue="method",
        color="firebrick",
        fill=True,
        alpha=0.2,
        ax=axs[0],
    )
    sns.kdeplot(
        data=df_mape,
        x="loss",
        common_norm=True,
        hue="method",
        color="firebrick",
        fill=True,
        alpha=0.2,
        ax=axs[1],
    )
    sns.kdeplot(
        data=df_emiss,
        x="emiss",
        common_norm=True,
        hue="origin",
        color="firebrick",
        fill=True,
        alpha=0.2,
        ax=axs[2],
    )
    sns.kdeplot(pred / y, color="firebrick", fill=True, alpha=0.2, ax=axs[3])

    titles = [
        "Mean absolute error",
        "Mean absolute percentage error",
        "Emission rate",
        "Prediction/Truth",
    ]

    for i_ax, ax in enumerate(axs):
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_xlabel(titles[i_ax])

    if dir_save != "None":
        plt.savefig(os.path.join(dir_save, "summary_inv.png"))


def plot_inversion_examples(
    data: Data_eval,
    all_mae: np.ndarray,
    all_mape: np.ndarray,
    model: tf.keras.Model,
    list_idx: list = [],
    list_ds_idx: list = [],
    proba_plume: float = 0,
    no2=False,
):
    """Plot examples of {input / truth / output} of the CNN model."""

    if not list_idx:
        [idx0, ds_idx0] = draw_idx(all_mae, data.ds)
        [idx1, ds_idx1] = draw_idx(all_mae, data.ds)
        [idx2, ds_idx2] = draw_idx(all_mae, data.ds)
        list_idx = [idx0, idx1, idx2]
        list_ds_idx = [ds_idx0, ds_idx1, ds_idx2]

    N_idx = len(list_idx)

    N_cols = 3
    if proba_plume > 0:
        N_cols = N_cols + 1
    if no2:
        N_cols = N_cols + 1

    mympf.setMatplotlibParam()
    plt.viridis()
    axs = mympf.set_figure_axs(
        N_idx,
        N_cols,
        wratio=0.35,
        hratio=0.75,
        pad_w_ext_left=0.25,
        pad_w_ext_right=0.25,
        pad_w_int=0.001,
        pad_h_ext=0.2,
        pad_h_int=0.25,
    )

    ims = [None] * (N_idx * N_cols)
    caxs = [None] * (N_idx * N_cols)
    cbars = [None] * (N_idx * N_cols)

    Ny = int(data.x.eval.shape[1])
    Nx = int(data.x.eval.shape[2])
    for ax in axs:
        ax.set_xticks([0, int(Ny / 4), int(Ny / 2), int(Ny * 3 / 4), Ny])
        ax.set_yticks([0, int(Nx / 4), int(Nx / 2), int(Nx * 3 / 4), Nx])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for i, idx in enumerate(list_idx):

        x_idx = data.x.eval[idx]

        cur_row = 0

        i_ax = cur_row + i * N_cols
        cur_row += 1
        ims[i_ax] = axs[i_ax].imshow(
            np.squeeze(data.x.eval[idx, :, :, 0]), origin="lower"
        )
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

        i_ax = cur_row + i * N_cols
        cur_row += 1
        ims[i_ax] = axs[i_ax].imshow(
            np.squeeze(data.x.eval[idx, :, :, 1]), origin="lower"
        )
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

        i_ax = cur_row + i * N_cols
        cur_row += 1
        ims[i_ax] = axs[i_ax].imshow(
            np.squeeze(data.x.eval[idx, :, :, 2]), origin="lower"
        )
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

    list_pd_t_idx = []
    list_all_scores = []
    for idx, ds_idx in enumerate(list_ds_idx):
        list_pd_t_idx.append(pd.Timestamp(ds_idx.time.values))
        list_all_scores.append(all_mae[list_idx[idx]])

    for i, (pd_t_idx, loss_idx) in enumerate(
        zip(
            list_pd_t_idx,
            list_all_scores,
        )
    ):
        axs[i * N_cols].set_ylabel(
            f"[{pd_t_idx.month:02d}-{pd_t_idx.day:02d} {pd_t_idx.hour:02d}:00]"
        )
    axs[-3].set_xlabel("Chan0")
    axs[-2].set_xlabel("Chan1")
    axs[-1].set_xlabel("Chan2")

    for i, idx in enumerate(list_idx):
        axs[1 + i * 3].set_title(
            f"time: {pd.Timestamp(list_ds_idx[i].time.values)},   "
            f"mae: {all_mae[idx].numpy():.3f},   "
            f"mape: {all_mape[idx].numpy():.3f},   "
            f"truth: {data.y.eval[idx][0]:.3f},   "
            f"pred: {model.predict(data.x.eval[idx:idx+1])[0][0]:.3f},   "
        )
