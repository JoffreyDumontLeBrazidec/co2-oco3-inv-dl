import build_ds
import cartopy.crs as ccrs
import ddeq
import inversion
import numpy as np
import pandas as pd

PS = "Lippendorf"

ds = build_ds.prepare_and_get_dataset(PS)


data = inversion.get_ddeq_predictions_for_PS(PS, selection="random", N_pred=1)

data = data.rename({"xco2": "CO2", "sources": "source"})
data = data.isel(idx_img=0)
data.CO2.attrs["units"] = "ppm"
data.attrs["noise"] = 0.7
data.attrs["trace_gas"] = "CO2"

data["psurf"] = data["xco2_noisy"]
data = ddeq.emissions.prepare_data(data, "CO2")
CRS = ccrs.UTM(32)
data = data.rename({"y": "nobs", "x": "nrows"})

data, curves = ddeq.plume_coords.compute_plume_line_and_coords(data, crs=CRS)

