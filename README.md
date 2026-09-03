# Example scripts for Open University Programming Task

## Example scripts are located under /COLIBRE-Research/Scripts

The first script is "SwiftGalaxy_Ebind_Example.ipynb", which demonstrates multiple ways of loading and reducing data from the COLIBRE simulations.
The use-case highlights how COLIBRE data can be used to identify correlations between the scatter in SMBH mass for a fixed halo binding energy at z=0.
It shows how we can take data directly from SOAP, or we can manually calculate properties from the snapshots using tools such as SWIFTGalaxy.

The second script is "LSTM_TimeSHAP_Example.ipynb", which shows how deep learning models can be trained on time-series data of galaxy properties, to predict their eventual black hole masses at z=0 for a fixed halo binding energy. The full timeline of data reduction and scaling, hyperparameter tuning, model training and validating, testing on a held-out set, reporting accuracy metrics, and further analysis using TimeSHAP, is shown.
Briefly, TimeSHAP is a feature attribution framework that allows for the relative importance of individual features, time steps, or a combination of features at a given time step, in the model's predictions for target variable, in our case SMBH mass at fixed binding energy at z=0.

## The Util directory under /COLIBRE-Research/Util just contains some additional code that is imported into the two main scripts, and is not directly relevant to the task, they are simply auxiliary files
