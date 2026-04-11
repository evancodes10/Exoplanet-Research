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

Section 0: Pandas Dataframes

Returns the full dataset or any subsection as a clean pandas DataFrame.
Each sub-dataframe can be used independently or merged into one master frame.

Usage
-----
    dfs = kepler10_dataframe()              # all sections
    dfs = kepler10_dataframe("lightcurve")  # just light curve
    dfs = kepler10_dataframe("stellar")     # just stellar params
    dfs = kepler10_dataframe("planets")     # just planet params
    dfs = kepler10_dataframe("derived")     # just derived quantities
    dfs = kepler10_dataframe("habitability")# just habitability metrics
    dfs = kepler10_dataframe("full")        # single merged master dataframe

"""

def kepler10_dataframe(subset="all"):
    """
    Builds pandas DataFrames for the Kepler-10 dataset.

    Parameters
    ----------
    subset : str
        Which dataframe(s) to return. Options:
            "all"         — dict of all dataframes (default)
            "lightcurve"  — time-series flux data
            "stellar"     — stellar catalog parameters
            "planets"     — planet catalog parameters
            "derived"     — computed physical quantities
            "habitability"— habitability & environment metrics
            "full"        — single wide DataFrame (all sections merged)

    Returns
    -------
    dict of DataFrames  (if subset="all")
    single DataFrame    (if a specific subset or "full" is requested)
    """

    import lightkurve as lk
    import numpy as np
    import pandas as pd
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
    from astropy import units as u
    from astropy import constants as const

    # ── Constants ─────────────────────────────────────────────────────────────
    G        = const.G.value
    R_sun    = const.R_sun.value
    M_sun    = const.M_sun.value
    R_earth  = const.R_earth.value
    M_earth  = const.M_earth.value
    L_sun    = const.L_sun.value
    AU       = const.au.value
    sigma_SB = const.sigma_sb.value

    # ── Helper ────────────────────────────────────────────────────────────────
    def safe(table, row_idx, col, default=np.nan):
        try:
            v = table[row_idx][col]
            return float(v) if str(v) not in ("--", "nan", "") else default
        except Exception:
            return default

    # =========================================================================
    # LIGHTCURVE DATAFRAME
    # time | flux | flux_err | centroid_col | centroid_row | quality | ...
    # =========================================================================
    
    def build_lightcurve_df():
        print("  [lightcurve] Downloading all Kepler quarters...")
        lc_collection = lk.search_lightcurve(
            "Kepler-10", mission="Kepler"
        ).download_all()
        lc = lc_collection.stitch().remove_nans().normalize()
        df = lc.to_pandas().reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.insert(0, "target", "Kepler-10")
        print(f"  [lightcurve] {len(df)} rows  x  {len(df.columns)} columns")
        return df

    # =========================================================================
    # STELLAR DATAFRAME
    # one row = Kepler-10 star  |  cols = all archive stellar fields
    # =========================================================================
    
    def build_stellar_df():
        print("  [stellar] Querying NASA Exoplanet Archive...")
        planets = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            where="hostname='Kepler-10'"
        )
        stellar_cols = [
            c for c in planets.colnames
            if c.startswith("st_") or c.startswith("sy_")
            or c in ("hostname", "ra", "dec")
        ]
        row = {col: safe(planets, 0, col) for col in stellar_cols}
        row["hostname"] = "Kepler-10"
        df = pd.DataFrame([row])
        print(f"  [stellar] {len(df)} row  x  {len(df.columns)} columns")
        return df

    # =========================================================================
    # PLANETS DATAFRAME
    # one row per planet  |  cols = all archive planet fields
    # =========================================================================
    
    def build_planets_df():
        print("  [planets] Querying NASA Exoplanet Archive...")
        planets = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            where="hostname='Kepler-10'"
        )
        planet_cols = [
            c for c in planets.colnames
            if c.startswith("pl_") or c == "pl_name"
        ]
        rows = []
        for i, row in enumerate(planets):
            entry = {"pl_name": str(row["pl_name"])}
            for col in planet_cols:
                entry[col] = safe(planets, i, col)
            rows.append(entry)
        df = pd.DataFrame(rows)
        print(f"  [planets] {len(df)} rows  x  {len(df.columns)} columns")
        return df

    # =========================================================================
    # DERIVED DATAFRAME
    # computed quantities for star + both planets in one tidy frame
    # =========================================================================
    
    def build_derived_df():
        print("  [derived] Computing physical quantities...")
        planets = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            where="hostname='Kepler-10'"
        )

        R_star_m  = safe(planets, 0, "st_rad",  1.065) * R_sun
        M_star_kg = safe(planets, 0, "st_mass", 0.913) * M_sun
        T_star_K  = safe(planets, 0, "st_teff", 5627.0)
        L_star_W  = safe(planets, 0, "st_lum",  0.585) * L_sun

        def planet_row(name, P_days, R_Re, M_Me):
            P_s    = P_days * 86400
            R_m    = R_Re * R_earth
            M_kg   = M_Me * M_earth
            a_m    = (G * M_star_kg * P_s**2 / (4 * np.pi**2)) ** (1/3)
            v_orb  = 2 * np.pi * a_m / P_s
            rho    = M_kg / ((4/3) * np.pi * R_m**3)
            g_surf = G * M_kg / R_m**2
            v_esc  = np.sqrt(2 * G * M_kg / R_m)
            T_eq   = T_star_K * np.sqrt(R_star_m / (2 * a_m)) * (1 - 0.30)**0.25
            Rp_Rs  = R_m / R_star_m
            R_hill = a_m * (M_kg / (3 * M_star_kg))**(1/3)
            S_ins  = (L_star_W / (4 * np.pi * a_m**2)) / 1361.0   # S_Earth
            return {
                "name":              name,
                "period_days":       P_days,
                "sma_au":            a_m / AU,
                "sma_over_rstar":    a_m / R_star_m,
                "v_orbital_kms":     v_orb / 1000,
                "radius_rearth":     R_Re,
                "mass_mearth":       M_Me,
                "density_gcc":       rho / 1000,
                "surface_gravity_g": g_surf / 9.81,
                "escape_vel_kms":    v_esc / 1000,
                "teq_K":             T_eq,
                "teq_C":             T_eq - 273.15,
                "transit_depth_ppm": Rp_Rs**2 * 1e6,
                "rp_over_rs":        Rp_Rs,
                "hill_sphere_rearth":R_hill / R_earth,
                "insolation_searth": S_ins,
            }

        def get_planet(planets, fragment, P_default, R_default, M_default):
            for i, row in enumerate(planets):
                if fragment in str(row["pl_name"]):
                    return (
                        safe(planets, i, "pl_orbper", P_default),
                        safe(planets, i, "pl_rade",   R_default),
                        safe(planets, i, "pl_masse",  M_default),
                    )
            return P_default, R_default, M_default

        Pb, Rb, Mb = get_planet(planets, "10 b", 0.8374907, 1.47, 3.72)
        Pc, Rc, Mc = get_planet(planets, "10 c", 45.29485,  2.35, 17.2)

        df = pd.DataFrame([
            planet_row("Kepler-10b", Pb, Rb, Mb),
            planet_row("Kepler-10c", Pc, Rc, Mc),
        ])
        print(f"  [derived] {len(df)} rows  x  {len(df.columns)} columns")
        return df

    # =========================================================================
    # HABITABILITY DATAFRAME
    # one row per planet with HZ, TSM, tidal locking, insolation
    # =========================================================================
    
    def build_habitability_df():
        print("  [habitability] Computing habitability metrics...")
        planets = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            where="hostname='Kepler-10'"
        )

        L_lsun   = safe(planets, 0, "st_lum",  0.585)
        T_star_K = safe(planets, 0, "st_teff", 5627.0)
        R_star_m = safe(planets, 0, "st_rad",  1.065) * R_sun
        M_star_kg= safe(planets, 0, "st_mass", 0.913) * M_sun
        R_Rs     = safe(planets, 0, "st_rad",  1.065)

        T_rel = T_star_K - 5780
        def hz(S0, c):
            S = S0 + c[0]*T_rel + c[1]*T_rel**2 + c[2]*T_rel**3
            return np.sqrt(L_lsun / S)
        hz_inner = hz(1.7665, [1.3351e-4, 3.1515e-9, -3.3488e-12])
        hz_outer = hz(0.3240, [5.3221e-5, 1.4288e-9, -1.1049e-12])

        def hab_row(name, P_days, R_Re, M_Me, fragment, i_idx):
            P_s   = P_days * 86400
            R_m   = R_Re * R_earth
            M_kg  = M_Me * M_earth
            a_m   = (G * M_star_kg * P_s**2 / (4 * np.pi**2)) ** (1/3)
            a_AU  = a_m / AU
            L_W   = L_lsun * L_sun
            T_eq  = T_star_K * np.sqrt(R_star_m / (2 * a_m)) * (1-0.30)**0.25
            S_ins = L_lsun / a_AU**2
            sf    = 0.190 if R_Re < 1.5 else (1.26 if R_Re < 2.75 else 1.15)
            TSM   = sf * (R_Re**3 * T_eq) / (M_Me * R_Rs**2) * 10**(-9.0/5)
            return {
                "name":               name,
                "hz_inner_AU":        hz_inner,
                "hz_outer_AU":        hz_outer,
                "sma_AU":             a_AU,
                "in_habitable_zone":  hz_inner <= a_AU <= hz_outer,
                "teq_K":              T_eq,
                "insolation_searth":  S_ins,
                "flux_at_planet_Wm2": L_W / (4 * np.pi * a_m**2),
                "transit_prob_pct":   R_star_m / a_m * 100,
                "tidally_locked":     P_days < 10,
                "TSM_jwst":           TSM,
                "good_jwst_target":   TSM > 10,
            }

        def get_planet(planets, fragment, P_def, R_def, M_def):
            for i, row in enumerate(planets):
                if fragment in str(row["pl_name"]):
                    return (
                        safe(planets, i, "pl_orbper", P_def),
                        safe(planets, i, "pl_rade",   R_def),
                        safe(planets, i, "pl_masse",  M_def),
                        i,
                    )
            return P_def, R_def, M_def, 0

        Pb, Rb, Mb, ib = get_planet(planets, "10 b", 0.8374907, 1.47, 3.72)
        Pc, Rc, Mc, ic = get_planet(planets, "10 c", 45.29485,  2.35, 17.2)

        df = pd.DataFrame([
            hab_row("Kepler-10b", Pb, Rb, Mb, "10 b", ib),
            hab_row("Kepler-10c", Pc, Rc, Mc, "10 c", ic),
        ])
        print(f"  [habitability] {len(df)} rows  x  {len(df.columns)} columns")
        return df

    # =========================================================================
    # DISPATCH
    # =========================================================================
    builders = {
        "lightcurve":   build_lightcurve_df,
        "stellar":      build_stellar_df,
        "planets":      build_planets_df,
        "derived":      build_derived_df,
        "habitability": build_habitability_df,
    }

    if subset in builders:
        return builders[subset]()

    # Build all
    print("[kepler10_dataframe] Building all subsections...")
    dfs = {name: fn() for name, fn in builders.items()}

    if subset == "full":
        # Merge derived + habitability (planet-level, 2 rows)
        planet_df = dfs["derived"].merge(
            dfs["habitability"].drop(columns=["hz_inner_AU","hz_outer_AU",
                                               "sma_AU","teq_K","insolation_searth"],
                                     errors="ignore"),
            on="name", how="left"
        )
        # Add stellar scalars as repeated columns
        stellar_scalar = dfs["stellar"].iloc[0].to_dict()
        for k, v in stellar_scalar.items():
            planet_df[f"star_{k}"] = v

        print(f"\n  [full] {len(planet_df)} rows  x  {len(planet_df.columns)} columns")
        return planet_df

    return dfs

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
    
    