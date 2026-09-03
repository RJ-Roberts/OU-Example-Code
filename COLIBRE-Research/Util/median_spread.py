import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


def median_spread(x, y, lower_frac=0.25, upper_frac=0.75, window=50):

    # Compute rolling 10th and 90th percentiles
    val_lower = y.rolling(window, center=True).quantile(lower_frac)
    val_upper = y.rolling(window, center=True).quantile(upper_frac)
    val_median = y.rolling(window, center=True).quantile(0.5)

    # Compute LOWESS smoothed curves
    val_lowess_lower = lowess(val_lower, x, frac=0.2, return_sorted=False)
    val_lowess_upper = lowess(val_upper, x, frac=0.2, return_sorted=False)
    val_lowess_median = lowess(val_median, x, frac=0.2, return_sorted=False)
    
    return val_lowess_median, val_lowess_lower, val_lowess_upper