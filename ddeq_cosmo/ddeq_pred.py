# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import numpy as np
import sys
import pandas as pd
import xarray as xr
import joblib
import pickle

import scipy.ndimage
import scipy.stats
import shapely
import skimage.measure
import xarray
import dplume
import ddeq

import build_ds


def apply_dplume(data, sources, noisy=True, filter_size=0.5):
    """Build data with sources, then apply ddeq dplume algorithm."""
    if noisy:
        data["xco2"] = (["y", "x"], data["xco2_noisy"].values)
    data = dplume.detect_plumes(data, sources, filter_size=filter_size)
    ddeq_plume = np.sum(data.detected_plume, axis=-1)
    return ddeq_plume


def compute_ddeq_plumes(ds_pred, sources, noisy=True, filter_size=0.5):
    """Compute ddeq plumes for ds_pred and sources."""

    all_detections = np.empty(
        (len(ds_pred.idx_img), ds_pred.dims["y"], ds_pred.dims["x"])
    )
    for i, idx_img in enumerate(ds_pred.idx_img.values):
        # print(i, "/", len(ds_pred.idx_img.values))
        all_detections[i] = apply_dplume(
            ds_pred.sel(idx_img=idx_img), sources, noisy, filter_size=filter_size
        )

    ds_pred["ddeq_plumes"] = (["idx_img", "y", "x"], all_detections)
    return ds_pred


def get_ddeq_predictions_for_PS(
    PS,
    selection,
    selected_idx=np.arange(0, 30, 1),
    N_pred=30,
    noisy=True,
    filter_size=0.5,
):
    """Get ddeq predictions for specific PS: build ds, sources, then compute ddeq plumes."""

    if PS == "Paris":
        sources = xr.open_dataset("/libre/dumontj/coco2/raw_data/lsce/xco2/sources.nc")
    else:
        sources = ddeq.misc.read_point_sources()

    ds = build_ds.prepare_and_get_dataset(PS)

    if selection == "random":
        selected_idx = np.random.choice(len(ds.idx_img), size=N_pred, replace=False)
        ds_pred = ds.sel(idx_img=selected_idx)
    elif selection == "specific":
        ds_pred = ds.sel(idx_img=selected_idx)
    elif selection == "all":
        ds_pred = ds

    ds_pred = compute_ddeq_plumes(ds_pred, sources, noisy, filter_size)
    return ds_pred


def get_ddeq_predictions_forall_PS():
    dir_eval_ddeq = "/libre/dumontj/coco2/ddeq/eval"
    # for PS in ["Boxberg", "Lippendorf", "Janschwalde", "Berlin"]:
    for PS in ["Paris"]:
        print(PS)
        if PS in ["Berlin", "Paris"]:
            filter_size = 3
        elif PS in ["Boxberg", "Lippendorf", "Janschwalde"]:
            filter_size = 1
        else:
            sys.exit()
        ds_pred = get_ddeq_predictions_for_PS(
            PS, selection="all", filter_size=filter_size
        )
        save_dataset(
            os.path.join(dir_eval_ddeq, build_ds.get_abbrev_source(PS), "res_ddeq.nc"),
            ds_pred,
        )


def save_dataset(path_ds: str, ds: xr.Dataset) -> None:
    """Save dataset in path_ds."""
    if not os.path.exists(os.path.dirname(path_ds)):
        os.makedirs(os.path.dirname(path_ds))
    if os.path.exists(path_ds):
        os.remove(path_ds)
    ds.to_netcdf(path_ds, mode="w")
