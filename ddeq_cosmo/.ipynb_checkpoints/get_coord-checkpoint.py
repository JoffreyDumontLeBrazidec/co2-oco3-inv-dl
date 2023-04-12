# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import numpy as np
import xarray as xr
import pandas as pd
import sys

sys.path.append("/cerea_raid/users/dumontj/dev/coco2/data_build")
import ddeq
import coco2_data_config
import coco2_data_tools

def get_coord_around_source(
    name_source: str, Ny: int, Nx: int
) -> np.ndarray:
    """Get SMARTCARB coordinates around a specified source."""
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

    coords_dict = {"lon_o": lon_o, "lat_o": lat_o, "source_lons": source_lons, "source_lats": source_lats}
    
    return coords_dict