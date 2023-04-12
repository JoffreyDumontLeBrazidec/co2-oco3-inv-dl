# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append("/cerea_raid/users/dumontj/dev/coco2/data_build")
import coco2_data_config
import coco2_data_tools
import ddeq


def get_coord_around_source(name_source: str, Ny: int, Nx: int) -> np.ndarray:
    """Get SMARTCARB coordinates around a specified source."""
    if name_source in [
        "Berlin",
        "Janschwalde",
        "Boxberg",
        "Lippendorf",
        "Turow",
        "Schwarze Pumpe",
    ]:
        dir_raw_smartcarb_xco2_data = os.path.join(
            coco2_data_config.raw_data_dir, "smartcarb", "cosmo2D"
        )

        # get lons, lats from random cosmo_2d_file
        cosmo_2d_file = os.path.join(
            dir_raw_smartcarb_xco2_data, f"cosmo_2d_{2015:01}{2:02}{15:02}{10:02}.nc"
        )
        ds = xr.open_dataset(
            cosmo_2d_file,
            decode_times=False,
        )
        lons = ds.lon.values
        lats = ds.lat.values

        # get source lon, lat in lons, lats
        sources = ddeq.misc.read_point_sources()
        source = sources.sel(source=name_source)
        lon_o = float(source["lon_o"])
        lat_o = float(source["lat_o"])
        dist_to_source = np.abs(lons - lon_o) + np.abs(lats - lat_o)
        ilat_o, ilon_o = np.unravel_index(dist_to_source.argmin(), dist_to_source.shape)

        source_lons = lons[
            ilat_o - int(Ny / 2) : ilat_o + int(Ny / 2),
            ilon_o - int(Nx / 2) : ilon_o + int(Nx / 2),
        ]

        source_lats = lats[
            ilat_o - int(Ny / 2) : ilat_o + int(Ny / 2),
            ilon_o - int(Nx / 2) : ilon_o + int(Nx / 2),
        ]

    elif name_source == "Paris":
        source_lats, source_lons = coco2_data_tools.get_grid_coordinates("lsce", Ny, Nx)
        lat_o = np.median(source_lats)
        lon_o = np.median(source_lons)

    coords_dict = {
        "lon_o": lon_o,
        "lat_o": lat_o,
        "source_lons": source_lons,
        "source_lats": source_lats,
    }

    return coords_dict


def get_segmentation_weighted(ds, min_w, max_w):
    plume = np.array(ds.plume.values, dtype=np.float32)
    threshold_min = 0.05
    N_data = ds.N_img

    y_min = np.repeat([threshold_min], N_data).reshape(N_data, 1, 1)
    y_max = np.quantile(plume, q=0.99, axis=(1, 2)).reshape(N_data, 1, 1)
    weight_min = np.repeat([min_w], N_data).reshape(N_data, 1, 1)
    weight_max = np.repeat([max_w], N_data).reshape(N_data, 1, 1)
    pente = (weight_max - weight_min) / (y_max - y_min)
    b = weight_min - pente * y_min

    y_data = pente * plume + b * np.where(plume > 0, 1, 0)
    y_data = np.where(y_data < max_w, y_data, max_w)

    ds["weight_bool_plume"] = (["idx_img", "y", "x"], y_data)
    return ds


def get_noisy_xco2(ds):
    noise_level = 0.7
    xco2 = ds["xco2"].values
    xco2_noisy = xco2 + noise_level * np.random.randn(*xco2.shape).astype(xco2.dtype)
    xco2_std = np.full(shape=xco2.shape, fill_value=noise_level)
    ds["xco2_std"] = (["idx_img", "y", "x"], xco2_std)
    ds["xco2_noisy"] = (["idx_img", "y", "x"], xco2_noisy)
    return ds


def get_noisy_no2(ds):
    noise_level = 2e15
    no2 = ds["no2"].values
    no2_noisy = no2 + noise_level * np.random.randn(*no2.shape).astype(no2.dtype)
    no2_std = np.full(shape=no2.shape, fill_value=noise_level)
    ds["no2_std"] = (["idx_img", "y", "x"], no2_std)
    ds["no2_noisy"] = (["idx_img", "y", "x"], no2_noisy)
    return ds


def get_lon_lat(ds, PS):
    Ny = ds.dims["y"]
    Nx = ds.dims["x"]

    coord_dict = get_coord_around_source(PS, Ny, Nx)
    lon = np.tile(
        coord_dict["source_lons"].reshape(1, Ny, Nx), (ds.attrs["N_img"], 1, 1)
    )
    lat = np.tile(
        coord_dict["source_lats"].reshape(1, Ny, Nx), (ds.attrs["N_img"], 1, 1)
    )
    ds["lon"] = (["idx_img", "y", "x"], lon)
    ds["lat"] = (["idx_img", "y", "x"], lat)
    return ds


def prepare_and_get_dataset(PS):

    ds = xr.open_dataset(
        f"/libre/dumontj/coco2/dl-input/{get_abbrev_source(PS)}/valid_dataset.nc"
    )
    ds = get_segmentation_weighted(ds, min_w=0.01, max_w=4)
    ds = get_noisy_xco2(ds)
    ds = get_noisy_no2(ds)
    ds = get_lon_lat(ds, PS)

    return ds


def get_abbrev_source(PS):
    cases = {
        "Boxberg": "2km_Box",
        "Lippendorf": "2km_Lip",
        "Janschwalde": "2km_Jan",
        "Berlin": "2km_Ber",
    }
    return cases[PS]
