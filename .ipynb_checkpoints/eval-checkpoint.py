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
import matplotlib_functions
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import xarray as xr
from cmcrameri import cm
from hydra import compose, initialize
from omegaconf import OmegaConf
from scipy.optimize import differential_evolution
from treeconfigparser import TreeConfigParser

from Data import Data_eval, get_scaler
from include.loss import pixel_weighted_cross_entropy
from cfg.convert_cfg_to_yaml import save_myyaml_from_mycfg

def get_scaler(dir_res: str):
    scaler = joblib.load(os.path.join(dir_res, "scaler.save"))
    return scaler

def get_data_for_segmentation(dir_res: str, path_eval_nc: str, ds_inds: dict = dict(), region_extrapol: bool = False):
    """Prepare Data object with name_dataset, scaler, and train or test mode."""
    
    if not os.path.exists(os.path.join(dir_res, "config.yaml")):
        save_myyaml_from_mycfg(dir_res)        
    cfg = OmegaConf.load(os.path.join(dir_w, "config.yaml"))
    
    data = Data_eval(path_eval_nc, ds_inds, region_extrapol)
    data.prepare_input(get_scaler(dir_res), cfg.data.input.chan_0.type, cfg.data.input.chan_1)
    data.prepare_output_segmentation(
        curve=data.output.curve, 
        min_w=cfg.data.output.min_w, 
        max_w=cfg.data.output.max_w,
        param_curve=data.output.param_curve,
    )    
    return data


def get_segmentation_model(dir_res: str, name_w: str="weights_cp_best.h5", optimiser: str="adam", loss=pixel_weighted_cross_entropy):
    """Get segmentation neural network model and compile it with pixel_weighted_cross_entropy loss."""
    model = keras.models.load_model(
        os.path.join(dir_res, name_w), compile=False
    )
    model.compile(optimiser, loss=loss)
    return model


def shift_to_proba(y_pred, proba_max: np.float32, proba_min: np.float32):
    """Shift to a probability map from a boolean map: 1 to proba_max, 0 to proba_min."""
    y_pred = np.where(y_pred == 1, proba_max, proba_min)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    return y_pred


def get_wbce(y_test: tf.Tensor, pred_test: tf.Tensor):
    """Get wbce given y_test and pred_test."""
    all_wbce = pixel_weighted_cross_entropy(y_test, pred_test, reduction=False)
    all_wbce = np.mean(all_wbce, axis=(1, 2))
    return all_wbce


def get_mean_loss(params, y_test: tf.Tensor, pred_test: tf.Tensor):
    """Get mean wbce between y_test and pred_test given shift_to_proba with params."""
    proba_min, proba_max = params
    current_pred_test = shift_to_proba(pred_test, proba_max, proba_min)
    wbce = get_wbce(y_test, current_pred_test)
    return np.mean(wbce)


def get_neutral_baseline_wbce(y_test: tf.Tensor):
    """Get wbce for y_test and *neutral baseline* predictions."""
    neutral_pred_test = 0.0 * tf.ones(shape=y_test.shape)
    res = differential_evolution(
        get_mean_loss, args=(y_test, neutral_pred_test), bounds=[[0, 1], [0, 1]]
    )
    [proba_min, proba_max] = res["x"]
    shifted_neutral_pred_test = shift_to_proba(neutral_pred_test, proba_max, proba_min)
    neutral_wbce = get_wbce(y_test, shifted_neutral_pred_test)
    return neutral_wbce


def draw_idx(
    cnn_nwbce: np.ndarray, ds_test: xr.Dataset, interval: list = None, idx: int = None
):
    """Draw a specific field/plume index to plot given potential interval."""
    if idx:
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


def plot_examples(
    data: Data,
    cnn_nwbce: np.ndarray,
    scaler,
    model: keras.Model,
    list_idx: list,
    list_ds_idx: list,
):
    """Plot examples of {input / truth / output} of the CNN model."""
    N_idx = len(list_idx)
    N_cols = 3
    matplotlib_functions.setMatplotlibParam()
    plt.viridis()
    axs = matplotlib_functions.set_figure_axs(
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

    for ax in axs:
        ax.set_xticks([0, 40, 80, 120, 160])
        ax.set_yticks([0, 40, 80, 120, 160])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for i, idx in enumerate(list_idx):
        i_ax = 0 + i * N_cols
        ims[i_ax] = axs[i_ax].imshow(
            scaler.inverse_transform(np.squeeze(data.x.test[idx])), origin="lower"
        )
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

        i_ax = 1 + i * N_cols
        ims[i_ax] = axs[i_ax].imshow(np.squeeze(data.y.test[idx]), origin="lower")
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

        i_ax = 2 + i * N_cols
        ims[i_ax] = axs[i_ax].imshow(
            np.squeeze(model(tf.expand_dims(data.x.test[idx], 0))[0]),
            vmin=0,
            vmax=1,
            cmap=cm.cork,
            origin="lower",
        )
        caxs[i_ax] = axs[i_ax].inset_axes((1.02, 0, 0.035, 1))
        cbars[i_ax] = plt.colorbar(ims[i_ax], caxs[i_ax], orientation="vertical")

    pd_t_idx0 = pd.Timestamp(list_ds_idx[0].time.values)
    pd_t_idx1 = pd.Timestamp(list_ds_idx[1].time.values)
    pd_t_idx2 = pd.Timestamp(list_ds_idx[2].time.values)
    pd_t_idx3 = pd.Timestamp(list_ds_idx[3].time.values)

    for i, (pd_t_idx, loss_idx) in enumerate(
        zip(
            [pd_t_idx0, pd_t_idx1, pd_t_idx2, pd_t_idx3],
            [
                cnn_nwbce[list_idx[0]],
                cnn_nwbce[list_idx[1]],
                cnn_nwbce[list_idx[2]],
                cnn_nwbce[list_idx[3]],
            ],
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
