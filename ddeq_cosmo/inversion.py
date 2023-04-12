# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import pickle
import sys

import build_ds

sys.path.append("/cerea_raid/users/dumontj/dev/coco2/dl/ddeq_cosmo/cross_sectional/src")
import ddeq
import dplume
import joblib
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.stats
import shapely
import skimage.measure
import xarray
import xarray as xr


def apply_dplume(data, sources, noisy=True, filter_size=0.5):
    """Build data with sources, then apply ddeq dplume algorithm."""
    if noisy:
        data["xco2"] = (["y", "x"], data["xco2_noisy"].values)
    data = dplume.detect_plumes(data, sources, filter_size=filter_size)
    data = dplume.detect_plumes(
        data,
        sources,
        variable="no2",
        variable_std="no2_std",
        filter_type="gaussian",
        filter_size=filter_size,
    )
    print(list(data.keys()))
    return data


def get_Nsources(ds_pred, sources):
    """Get number of sources considering a data object with lon_o, lat_o, images."""
    data = ds_pred.isel(idx_img=0)
    N_sources = len(
        np.where(
            dplume.overlaps_with_sources(
                data.lon,
                data.lat,
                sources["lon_o"],
                sources["lat_o"],
                sources["radius"],
            )
        )[0]
    )
    return N_sources


def compute_ddeq_plumes(ds_pred, sources, noisy=True, filter_size=0.5):
    """Compute ddeq plumes for ds_pred and sources."""

    N_img = len(ds_pred.idx_img)
    Ny = ds_pred.dims["y"]
    Nx = ds_pred.dims["x"]
    N_sources = get_Nsources(ds_pred, sources)

    all_detections = np.empty(
        (
            N_img,
            Ny,
            Nx,
            N_sources,
        ),
        dtype=bool,
    )
    all_is_hits = np.empty((N_img, Ny, Nx))
    all_labels = np.empty((N_img, Ny, Nx))
    all_local_CO2_mean = np.empty((N_img, Ny, Nx))
    all_local_CO2_median = np.empty((N_img, Ny, Nx))
    all_local_NO2_mean = np.empty((N_img, Ny, Nx))
    all_local_NO2_median = np.empty((N_img, Ny, Nx))
    all_lon_o = np.empty((N_img, N_sources))
    all_lat_o = np.empty((N_img, N_sources))
    all_sources = np.empty((N_img, N_sources), dtype=object)

    for i, idx_img in enumerate(ds_pred.idx_img.values):
        data = apply_dplume(
            ds_pred.sel(idx_img=idx_img), sources, noisy, filter_size=filter_size
        )
        all_detections[i] = data.detected_plume
        all_is_hits[i] = data.is_hit
        all_labels[i] = data.labels
        all_lon_o[i] = data.lon_o
        all_lat_o[i] = data.lat_o
        all_sources[i] = data.source
        all_local_CO2_mean[i] = data.local_xco2_mean
        all_local_CO2_median[i] = data.xco2_local_median
        all_local_NO2_mean[i] = data.local_no2_mean
        all_local_NO2_median[i] = data.no2_local_median

    ds_pred["detected_plume"] = (["idx_img", "y", "x", "sources"], all_detections)
    ds_pred["is_hit"] = (["idx_img", "y", "x"], all_is_hits.astype(bool))
    ds_pred["labels"] = (["idx_img", "y", "x"], all_labels)
    ds_pred["local_CO2_mean"] = (["idx_img", "y", "x"], all_local_CO2_mean)
    ds_pred["CO2_local_median"] = (["idx_img", "y", "x"], all_local_CO2_median)
    ds_pred["local_NO2_mean"] = (["idx_img", "y", "x"], all_local_NO2_mean)
    ds_pred["NO2_local_median"] = (["idx_img", "y", "x"], all_local_NO2_median)
    ds_pred["lon_o"] = (["idx_img", "sources"], all_lon_o)
    ds_pred["lat_o"] = (["idx_img", "sources"], all_lat_o)
    ds_pred["sources"] = all_sources[0]

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
