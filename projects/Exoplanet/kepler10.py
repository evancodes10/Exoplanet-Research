"""

Kepler-10 Exoplanet Analysis

Section 1 - Light Curve Implementation
Section 2 - Stellar & Planet Parameters
Section 3 - Radial Velocity
Section 4 - Spectral / Catalog Data
Section 5 - Derived Physical Quantities
Seciton 6 - Habitability & Environment

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import stats
from scipy.optimize import curve_fit
from astropy import units as u
from astropy import constants as const
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk

import warnings

warnings.filterwarnings("ignore")

"""

Constants

"""

Grav_const = const.G.value
R_sun = const.R_sun.value
M_sun = const.M_sun.value
R_earth = const.R_earth.value
M_earth = const.M_earth.value
L_sun = const.L_sun.value
Astrom_unit = const.au.value
sigma_SB = const.sigma_sb.value
pc = 3.0857e16

"""

Section 1: Function Implementation for Light Curve Data Analysis.

First gets the light curve data, normalizes it and removes NAN values, and then gets the flux values and 
runs a BLS periodogram to determine potential exoplanets. Finally, a folded light curve will be done to measure
exact changes in brightness, with a full light curve plot being created as well.


"""

def light_curve_analysis():
    
    print("Starting data extraction process.")
    
    search = lk.search_lightcurve("Kepler-10", mission="Kepler")
    print(search)
    
    lc_collection = search.download_all()
    lc = lc_collection.stitch().remove_nans().normalize()
    
    print(lc)
    
    print(f"Total Data Points: {len(lc)}")
    print(f"Time Baseline: {lc.time.value.min(): .2f} - {lc.time.value.max(): .2f} BKJD")
    print(f"Median Flux: {np.median(lc.flux.value): .6f}")
    print(f"Flux Standard Deviation: {np.std(lc.flux.value)*1e6: .1f} ppm")
    
    #BLS Implementation
    
    