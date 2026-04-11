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
    
    search = lk.search_lightcurve("Kepler-10", mission="Kepler", cadence="long", author="Kepler")
    print(search)
    
    lc_collection = search.download_all()
    lc_collection = lk.LightCurveCollection([lc_q.select_flux("pdcsap_flux") for lc_q in lc_collection])
    lc = lc_collection.stitch().remove_nans().normalize()
    lc_flat = lc.flatten(window_length=401)
    
    print(lc)
    
    print(f"Total Data Points: {len(lc)}")
    print(f"Time Baseline: {lc.time.value.min(): .2f} - {lc.time.value.max(): .2f} BKJD")
    print(f"Median Flux: {np.median(lc.flux.value): .6f}")
    print(f"Flux Standard Deviation: {np.std(lc.flux.value)*1e6: .1f} ppm")
    
    #BLS Implementation for Kepler-10b 
    
    print("Running BLS for Kepler-10b over days 0.5-5")
    
    pg_b_coarse = lc_flat.to_periodogram(method="bls", period=np.arange(0.5,5,0.005), duration=np.arange(0.02, 0.15, 0.02))
    rough_b = pg_b_coarse.period_at_max_power.value
    
    print(f"Kepler-10b Coarse Best Period: {rough_b: .4f} d")
    
    pg_b = lc_flat.to_periodogram(method="bls", period=np.linspace(rough_b * 0.98, rough_b * 1.02, 500), duration=np.arange(0.01,0.15,0.005))
    
    
    #BLS Implementation for Kepler-10c
    
    print("Running BLS for Kepler-10c over days 40-50.")
    pg_c_coarse = lc_flat.to_periodogram(method="bls", period=np.arange(40,50,0.05), duration=np.arange(0.1, 0.35, 0.05))
    rough_c = pg_c_coarse.period_at_max_power.value
    
    print(f"Kepler-10c Coarse Best Period {rough_c: .4f} d")

    pg_c = lc_flat.to_periodogram(method="bls", period = np.linspace(rough_c * 0.98, rough_c * 1.02, 500), duration = np.arange(0.05, 0.35, 0.01))
    
    #Dictonary Creation
    transit_paramaters_dictonary = {
        
        "b": {
            "period":   pg_b.period_at_max_power.value,
            "t0":       pg_b.transit_time_at_max_power.value,
            "duration": pg_b.duration_at_max_power.value,
            "depth":    pg_b.depth_at_max_power,
        },
        
        "c": {
            "period":   pg_c.period_at_max_power.value,
            "t0":       pg_c.transit_time_at_max_power.value,
            "duration": pg_c.duration_at_max_power.value,
            "depth":    pg_c.depth_at_max_power,
        },
    }
    
    print(f'Kepler-10b: Duration={transit_paramaters_dictonary["b"]["duration"]:.6f} d, Period={transit_paramaters_dictonary["b"]["period"]:.6f} d, Depth={transit_paramaters_dictonary["b"]["depth"]*1e6:.4f} ppm')
    print(f'Kepler-10c: Duration={transit_paramaters_dictonary["c"]["duration"]:.6f} d, Period={transit_paramaters_dictonary["c"]["period"]:.6f} d, Depth={transit_paramaters_dictonary["c"]["depth"]*1e6:.4f} ppm')
    
    
    #Plots 
    
    """

    Section 1b: Research-Grade Plots for Light Curve Analysis

    Produces publication-quality figures matching the style of Kepler discovery papers.
    All figures saved as high-DPI PNG and PDF (vector) for journal submission.

    Usage:
        lc, lc_flat, pg_b, pg_c, transit_params = light_curve_analysis()
        plot_light_curve_analysis(lc, lc_flat, pg_b, pg_c, transit_params)

    """

    def plot_light_curve_analysis(lc, lc_flat, pg_b, pg_c, transit_params):
        """
        Generates 6 publication-quality figures for the light curve section.

        Figures produced
        ----------------
        Fig 1  — Full PDC_SAP light curve with quality flags
        Fig 2  — BLS power spectra for both planets (side by side)
        Fig 3  — Phase-folded transit of Kepler-10b with binned model overlay
        Fig 4  — Phase-folded transit of Kepler-10c with binned model overlay
        Fig 5  — Odd/even transit comparison (false-positive check)
        Fig 6  — Transit timing: individual transit depths per epoch

        Parameters
        ----------
        lc             : raw normalized LightCurve from light_curve_analysis()
        lc_flat        : flattened LightCurve from light_curve_analysis()
        pg_b           : BLS periodogram for Kepler-10b
        pg_c           : BLS periodogram for Kepler-10c
        transit_params : dict with period, t0, duration, depth for b and c
        """

        # ── Global style — matches ApJ / A&A publication standards ───────────────
        plt.rcParams.update({
            "font.family":        "serif",
            "font.serif":         ["Times New Roman", "DejaVu Serif"],
            "font.size":          11,
            "axes.titlesize":     12,
            "axes.labelsize":     11,
            "xtick.labelsize":    10,
            "ytick.labelsize":    10,
            "xtick.direction":    "in",
            "ytick.direction":    "in",
            "xtick.top":          True,
            "ytick.right":        True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.linewidth":     1.0,
            "lines.linewidth":    1.0,
            "legend.fontsize":    9,
            "legend.framealpha":  0.9,
            "legend.edgecolor":   "0.8",
            "figure.dpi":         150,
            "savefig.dpi":        300,
            "savefig.bbox":       "tight",
            "savefig.pad_inches": 0.05,
        })

        P_b   = transit_params["b"]["period"]
        t0_b  = transit_params["b"]["t0"]
        dur_b = transit_params["b"]["duration"]
        dep_b = transit_params["b"]["depth"]

        P_c   = transit_params["c"]["period"]
        t0_c  = transit_params["c"]["t0"]
        dur_c = transit_params["c"]["duration"]
        dep_c = transit_params["c"]["depth"]

        time  = lc_flat.time.value
        flux  = lc_flat.flux.value

        # =========================================================================
        # FIGURE 1 — Full light curve
        # =========================================================================
        fig1, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

        # Top panel: flux
        axes[0].scatter(time, flux, s=0.8, c="0.3", alpha=0.5, linewidths=0, rasterized=True)
        axes[0].set_ylabel("Normalized Flux")
        axes[0].set_ylim(np.percentile(flux, 0.1), np.percentile(flux, 99.9))
        axes[0].set_title("Kepler-10  —  PDC_SAP Long-Cadence Light Curve (Q0–Q17)", pad=8)

        # Mark transit times for both planets
        for epoch in np.arange(t0_b, time.max(), P_b):
            if time.min() < epoch < time.max():
                axes[0].axvline(epoch, color="#2166AC", alpha=0.15, lw=0.5)
        for epoch in np.arange(t0_c, time.max(), P_c):
            if time.min() < epoch < time.max():
                axes[0].axvline(epoch, color="#D6604D", alpha=0.25, lw=0.8)

        from matplotlib.lines import Line2D
        axes[0].legend(
            handles=[
                Line2D([0], [0], color="#2166AC", alpha=0.6, lw=1.5, label=f"Kepler-10b transits  (P = {P_b:.5f} d)"),
                Line2D([0], [0], color="#D6604D", alpha=0.6, lw=1.5, label=f"Kepler-10c transits  (P = {P_c:.4f} d)"),
            ],
            loc="lower right", markerscale=2
        )

        # Bottom panel: residuals from median (noise floor check)
        rolling_med = pd.Series(flux).rolling(window=13, center=True).median().values
        residuals   = flux - rolling_med
        axes[1].scatter(time, residuals * 1e6, s=0.8, c="0.5", alpha=0.4,
                        linewidths=0, rasterized=True)
        axes[1].axhline(0, color="k", lw=0.8, ls="--")
        axes[1].set_ylabel("Residuals\n(ppm)")
        axes[1].set_xlabel("Time (BKJD)")
        axes[1].set_ylim(-3000, 3000)

        fig1.align_ylabels(axes)
        plt.savefig("fig1_full_lightcurve.png")
        plt.savefig("fig1_full_lightcurve.pdf")
        plt.show()
        print("  Saved: fig1_full_lightcurve.png / .pdf")

        # =========================================================================
        # FIGURE 2 — BLS power spectra
        # =========================================================================
        
        fig2, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(10, 4))

        # Planet b
        pg_b.plot(ax=ax_b, color="0.3", lw=0.8)
        ax_b.axvline(P_b, color="#2166AC", lw=1.5, ls="--",
                    label=f"P = {P_b:.5f} d")
        # Mark harmonics
        for harmonic in [P_b * 2, P_b * 3, P_b / 2]:
            if pg_b.period.value.min() < harmonic < pg_b.period.value.max():
                ax_b.axvline(harmonic, color="#2166AC", lw=0.8, ls=":", alpha=0.5)
        ax_b.set_title("BLS Periodogram — Kepler-10b")
        ax_b.set_xlabel("Period (days)")
        ax_b.set_ylabel("BLS Power")
        ax_b.legend()

        # Planet c
        pg_c.plot(ax=ax_c, color="0.3", lw=0.8)
        ax_c.axvline(P_c, color="#D6604D", lw=1.5, ls="--",
                    label=f"P = {P_c:.4f} d")
        ax_c.set_title("BLS Periodogram — Kepler-10c")
        ax_c.set_xlabel("Period (days)")
        ax_c.set_ylabel("BLS Power")
        ax_c.legend()

        fig2.suptitle("Box Least Squares Periodograms", y=1.01)
        plt.tight_layout()
        plt.savefig("fig2_bls_periodograms.png")
        plt.savefig("fig2_bls_periodograms.pdf")
        plt.show()
        print("  Saved: fig2_bls_periodograms.png / .pdf")

        # =========================================================================
        # FIGURE 3 — Phase-folded transit: Kepler-10b
        # =========================================================================
        
        def phase_fold_plot(ax, lc_flat, period, t0, duration, depth, label, color):
            folded   = lc_flat.fold(period=period, epoch_time=t0)
            binned   = folded.bin(bins=200)
            ph       = folded.phase.value
            fl       = folded.flux.value
            ph_b     = binned.phase.value
            fl_b     = binned.flux.value
            fl_b_err = binned.flux_err.value

            # Raw scatter
            ax.scatter(ph * period * 24, fl, s=1.5, c="0.75", alpha=0.3,
                    linewidths=0, rasterized=True, label="Individual cadences")

            # Binned points with error bars
            ax.errorbar(ph_b * period * 24, fl_b, yerr=fl_b_err,
                        fmt="o", ms=3, color=color, ecolor=color,
                        elinewidth=0.8, capsize=2, zorder=5, label="30-min bins")

            # Transit duration marker
            ax.axvspan(-duration * 24 / 2, duration * 24 / 2,
                    alpha=0.08, color=color, label=f"Transit duration ({duration*24:.2f} h)")
            ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
            ax.axhline(1.0 - depth, color=color, lw=1.0, ls=":",
                    alpha=0.7, label=f"Transit depth ({depth*1e6:.0f} ppm)")

            ax.set_xlabel("Time from mid-transit (hours)")
            ax.set_ylabel("Normalized Flux")
            ax.set_title(f"Phase-folded Transit — {label}")
            ax.legend(loc="lower right", markerscale=2)

            # Zoom to ±3x transit duration
            ax.set_xlim(-duration * 24 * 3, duration * 24 * 3)
            ax.set_ylim(
                min(np.percentile(fl, 0.5), 1 - depth * 2),
                np.percentile(fl, 99.8)
            )

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        phase_fold_plot(ax3, lc_flat, P_b, t0_b, dur_b, dep_b,
                        f"Kepler-10b  (P = {P_b:.5f} d)", "#2166AC")
        plt.tight_layout()
        plt.savefig("fig3_transit_kepler10b.png")
        plt.savefig("fig3_transit_kepler10b.pdf")
        plt.show()
        print("  Saved: fig3_transit_kepler10b.png / .pdf")

        # =========================================================================
        # FIGURE 4 — Phase-folded transit: Kepler-10c
        # =========================================================================
        
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        phase_fold_plot(ax4, lc_flat, P_c, t0_c, dur_c, dep_c, f"Kepler-10c  (P = {P_c:.4f} d)", "#D6604D")
        plt.tight_layout()
        plt.savefig("fig4_transit_kepler10c.png")
        plt.savefig("fig4_transit_kepler10c.pdf")
        plt.show()
        print("  Saved: fig4_transit_kepler10c.png / .pdf")

        # =========================================================================
        # FIGURE 5 — Odd/even transit comparison (eclipsing binary false-positive check)
        # A real planet gives identical odd and even transits.
        # A background eclipsing binary gives alternating depths.
        # =========================================================================
        
        fig5, axes5 = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for planet_label, period, t0, duration, color, axs in [
            ("Kepler-10b", P_b, t0_b, dur_b, "#2166AC", axes5[0]),
            ("Kepler-10c", P_c, t0_c, dur_c, "#D6604D", axes5[1]),
        ]:
            epochs    = np.arange(t0, time.max(), period)
            odd_flux  = []
            even_flux = []
            odd_time  = []
            even_time = []

            half_dur = duration / 2 * 1.5

            for i, epoch in enumerate(epochs):
                mask = np.abs(time - epoch) < half_dur
                if mask.sum() < 2:
                    continue
                t_centered = (time[mask] - epoch) * 24
                f_centered = flux[mask]
                if i % 2 == 0:
                    odd_time.extend(t_centered)
                    odd_flux.extend(f_centered)
                else:
                    even_time.extend(t_centered)
                    even_flux.extend(f_centered)

            axs.scatter(odd_time,  odd_flux,  s=4, alpha=0.5, color=color,
                        label="Odd transits",  zorder=3)
            axs.scatter(even_time, even_flux, s=4, alpha=0.5, color="0.5",
                        marker="s", label="Even transits", zorder=3)
            axs.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
            axs.set_xlabel("Time from mid-transit (hours)")
            axs.set_title(f"{planet_label} — Odd/Even Check")
            axs.legend(markerscale=2)
            axs.set_xlim(-duration * 24 * 2.5, duration * 24 * 2.5)

        axes5[0].set_ylabel("Normalized Flux")
        fig5.suptitle("Odd/Even Transit Comparison  (eclipsing binary false-positive test)",
                    y=1.01)
        plt.tight_layout()
        plt.savefig("fig5_odd_even_check.png")
        plt.savefig("fig5_odd_even_check.pdf")
        plt.show()
        print("  Saved: fig5_odd_even_check.png / .pdf")

        # =========================================================================
        # FIGURE 6 — Per-epoch transit depths (transit timing variation proxy)
        # Measures the depth of each individual transit to check for TTVs or
        # stellar variability corrupting specific transits.
        # =========================================================================
        
        fig6, axes6 = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        for i, (planet_label, period, t0, duration, color, ax) in enumerate([
            ("Kepler-10b", P_b, t0_b, dur_b, "#2166AC", axes6[0]),
            ("Kepler-10c", P_c, t0_c, dur_c, "#D6604D", axes6[1]),
        ]):
            epochs       = np.arange(t0, time.max(), period)
            epoch_nums   = []
            depths_ppm   = []
            depth_errs   = []

            half_dur = duration / 2 * 1.8

            for n, epoch in enumerate(epochs):
                in_transit  = np.abs(time - epoch) < half_dur / 2
                out_transit = (np.abs(time - epoch) > half_dur / 2) & \
                            (np.abs(time - epoch) < half_dur * 2)
                if in_transit.sum() < 2 or out_transit.sum() < 3:
                    continue
                depth_val = (np.median(flux[out_transit]) - np.median(flux[in_transit])) * 1e6
                depth_err = np.std(flux[out_transit]) / np.sqrt(out_transit.sum()) * 1e6
                if depth_val > 0:
                    epoch_nums.append(n)
                    depths_ppm.append(depth_val)
                    depth_errs.append(depth_err)

            if epoch_nums:
                mean_depth = np.mean(depths_ppm)
                ax.errorbar(epoch_nums, depths_ppm, yerr=depth_errs,
                            fmt="o", ms=3, color=color, ecolor=color,
                            elinewidth=0.8, capsize=2, label="Individual transit depth")
                ax.axhline(mean_depth, color="k", lw=1.0, ls="--",
                        label=f"Mean = {mean_depth:.0f} ppm")
                ax.fill_between(
                    [min(epoch_nums), max(epoch_nums)],
                    mean_depth - np.std(depths_ppm),
                    mean_depth + np.std(depths_ppm),
                    alpha=0.12, color=color, label=f"±1σ = {np.std(depths_ppm):.0f} ppm"
                )
                ax.set_ylabel("Transit Depth (ppm)")
                ax.set_xlabel("Transit Epoch Number")
                ax.set_title(f"{planet_label} — Per-epoch Transit Depths")
                ax.legend()

        plt.tight_layout()
        plt.savefig("fig6_per_epoch_depths.png")
        plt.savefig("fig6_per_epoch_depths.pdf")
        plt.show()
        print("  Saved: fig6_per_epoch_depths.png / .pdf")

        print("\n  All figures saved as PNG (150 dpi screen) and PDF (300 dpi vector, journal-ready).")
        
    plot_light_curve_analysis(lc, lc_flat, pg_b, pg_c, transit_paramaters_dictonary)
    
    return lc, lc_flat, pg_b, pg_c, transit_paramaters_dictonary