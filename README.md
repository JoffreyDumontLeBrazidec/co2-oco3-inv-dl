# Inversion of CO2 plumes - OCO3-SAM and synthetic data

Official repository for the paper "Quantification of CO2 hotspot emissions from OCO-3 SAM CO2 satellite images using deep learning methods" submitted to "Geoscientific Model Development".

%$$$$$[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10100338.svg)](https://doi.org/10.5281/zenodo.10100338)
 
This project introduces the application of deep learning CO2

This project presents the development and application of a deep learning-based method for inverting CO2 atmospheric plumes
from power plants using satellite imagery of the CO2 total column mixing ratios (XCO2).

Our scripts and modules are written in Python, using Tensorflow as the deep learning framework.

To employ these scripts and train CNN models, download the datasets of fields and plumes from [inv-zenodo](https://doi.org/10.5281/zenodo.12788520).
Note that the data generation scripts are not part of this repository, but can be provided upon request.

Weights of trained models can be obtained from [inv-zenodo-weights](https://doi.org/10.5281/zenodo.12788520).

After data collection/generation, 'main.py` is used to train the CNN with config constructed from hydra.

For any queries, do not hesitate to reach out.

## Acknowledgements and Authors

This project is funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement 958927 (Prototype system for a Copernicus CO2 service). 
CEREA is a proud member of Institut Pierre Simon Laplace (IPSL).

## Support

Feel free to contact: joffrey.dumont@ecmwf.int