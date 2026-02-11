#%%
import os
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import matplotlib as mpl


BASE_PATH   = r"F:/S1_CarbonIAV/Landis_Result_new_ffff"
ECO_PATH    = r"D:/ForestC/30_calibration/99_calibration_data1_ini수정/ecoregions_F.tif"
PREFIX      = "LandisANPP_AG_NPP-"
CACHE_PARQUET   = r"D:/ForestC/C1_decade_carbon.parquet"
FORCE_RECOMPUTE = True
YEARS_ALL   = np.arange(2001, 2101)

REGION_IDS     = [1, 2, 3, 4, 5, 6]
REGION_LABELS  = [f"Band {i}" for i in REGION_IDS]
ALL_REGIONS    = ["National-level"] + REGION_LABELS
SCENARIOS   = ["SSP126", "SSP370", "SSP585"]
MODELS      = ["canesm5", "cnrm-cm6-1", "cnrm-esm2-1", "ec-earth3"]

DECADES = {
    "2000s": np.arange(2001, 2011),
    "2010s": np.arange(2011, 2021),
    "2020s": np.arange(2021, 2031),
    "2030s": np.arange(2031, 2041),
    "2040s": np.arange(2041, 2051),
    "2050s": np.arange(2051, 2061),
    "2060s": np.arange(2061, 2071),
    "2070s": np.arange(2071, 2081),
    "2080s": np.arange(2081, 2091),
    "2090s": np.arange(2091, 2101),
}
PERIODS = list(DECADES.keys())

BAR_WIDTH_HIST = 0.45
BAR_WIDTH_SSP  = 0.2
BAR_GAP_SSP    = 0.01
HIST_MS        = 5

HIST_BAR_ALPHA = 0.18
SSP_BAR_ALPHA  = 0.28
BAR_EDGE_COLOR = "0.35"
BAR_EDGE_LW    = 0.4
GRID_ALPHA     = 0.18

TITLE_FS  = 12
TICK_FS   = 10
LABEL_FS  = 12
LEGEND_FS = 10

Y_LIM_NPP = (400, 1000)
Y_LIM_STD = (0, 150)

SCENARIO_COLORS  = {"SSP126": "green", "SSP370": "gold", "SSP585": "red"}
SCENARIO_MARKERS = {"SSP126": "o",     "SSP370": "s",    "SSP585": "D"}
HIST_COLOR       = "k"
HIST_MARKER      = "o"

SCENARIO_TITLES = {
    "SSP126": "(a) SSP1–2.6",
    "SSP370": "(b) SSP3–7.0",
    "SSP585": "(c) SSP5–8.5",
}

MODEL_HATCHES = {
    "canesm5":      "////",
    "cnrm-cm6-1":   "\\\\\\\\",
    "cnrm-esm2-1":  "....",
    "ec-earth3":    "xxxx",
}

mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

with rasterio.open(ECO_PATH) as eco_src:
    ECO_MASK = eco_src.read(1)

def region_mask(region: str) -> np.ndarray:
    if region == "National-level":
        return np.isin(ECO_MASK, REGION_IDS)
    rid = int(region.split()[-1])
    return ECO_MASK == rid

def load_stack(folder: str, years: np.ndarray) -> np.ndarray:
    arrs = []
    for y in years:
        idx = y - 2000
        path = os.path.join(folder, f"{PREFIX}{idx}.tif")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raster not found: {path}")
        with rasterio.open(path) as src:
            a = src.read(1).astype(float)
            nodata = src.nodata
        if nodata is not None:
            a[a == nodata] = np.nan
        arrs.append(a)
    return np.stack(arrs)

def _grid_detrended_std(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    Y = np.asarray(Y, float)

    valid = np.isfinite(Y)
    n = valid.sum(axis=0).astype(float)

    out = np.full(Y.shape[1], np.nan, dtype=float)
    ok = n >= 2
    if not np.any(ok):
        return out

    x2 = x[:, None]
    sum_x  = np.sum(np.where(valid, x2, 0.0), axis=0)
    sum_y  = np.sum(np.where(valid, Y,  0.0), axis=0)

    x_mean = np.zeros_like(sum_x)
    y_mean = np.zeros_like(sum_y)
    x_mean[ok] = sum_x[ok] / n[ok]
    y_mean[ok] = sum_y[ok] / n[ok]

    xc = x2 - x_mean[None, :]
    yc = Y  - y_mean[None, :]

    cov_xy = np.sum(np.where(valid, xc * yc, 0.0), axis=0)
    var_x  = np.sum(np.where(valid, xc * xc, 0.0), axis=0)

    a = np.full_like(sum_x, np.nan, dtype=float)
    b = np.full_like(sum_x, np.nan, dtype=float)

    has_var = ok & (var_x > 0)
    a[has_var] = cov_xy[has_var] / var_x[has_var]
    b[has_var] = y_mean[has_var] - a[has_var] * x_mean[has_var]

    trend = a[None, :] * x2 + b[None, :]
    resid = Y - trend
    resid[~valid] = np.nan

    out[has_var] = np.nanstd(resid[:, has_var], axis=0, ddof=0)
    return out

def decadal_stats_from_stack_grid(years: np.ndarray, stack: np.ndarray, mask: np.ndarray):
    res = {}
    years = np.asarray(years)
    Y_all = stack[:, mask]

    for p, ys in DECADES.items():
        idx = np.isin(years, ys)
        x = years[idx].astype(float)
        Y = Y_all[idx, :].astype(float)

        mean_region = float(np.nanmean(np.nanmean(Y, axis=0)))
        std_region  = float(np.nanmean(_grid_detrended_std(x, Y)))

        res[p] = (mean_region, std_region)
    return res

def compute_all_to_df() -> pd.DataFrame:
    rows = []
    for region in ALL_REGIONS:
        mask = region_mask(region)

        for sc in SCENARIOS:
            npp_models, std_models = [], []

            for m in MODELS:
                folder = os.path.join(BASE_PATH, m, f"Landis_Result_{sc}", "20_result")
                stack_all = load_stack(folder, YEARS_ALL)
                dec = decadal_stats_from_stack_grid(YEARS_ALL, stack_all, mask)

                npp_vec = np.array([dec[p][0] for p in PERIODS], dtype=float)
                std_vec = np.array([dec[p][1] for p in PERIODS], dtype=float)

                npp_models.append(npp_vec)
                std_models.append(std_vec)

                for pi, p in enumerate(PERIODS):
                    rows.append((region, sc, m, p, "npp", float(npp_vec[pi])))
                    rows.append((region, sc, m, p, "std", float(std_vec[pi])))

            npp_mmm = np.nanmean(np.vstack(npp_models), axis=0)
            std_mmm = np.nanmean(np.vstack(std_models), axis=0)

            for pi, p in enumerate(PERIODS):
                rows.append((region, sc, "MMM", p, "npp", float(npp_mmm[pi])))
                rows.append((region, sc, "MMM", p, "std", float(std_mmm[pi])))

    return pd.DataFrame(rows, columns=["region", "scenario", "model", "period", "metric", "value"])

def load_or_compute_cache(cache_path: str, force: bool = False) -> pd.DataFrame:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if (not force) and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    df = compute_all_to_df()
    df.to_parquet(cache_path, index=False)
    print(f"[Saved cache] {cache_path}")
    return df

def _safe_period_vector(df_sub: pd.DataFrame, periods: list) -> np.ndarray:
    if df_sub is None or df_sub.empty:
        return np.full(len(periods), np.nan, dtype=float)
    s = df_sub.groupby("period")["value"].mean()
    v = s.reindex(periods).to_numpy(dtype=float).reshape(-1)
    if v.size != len(periods):
        raise ValueError(f"[vector size mismatch] got {v.size}, expected {len(periods)}")
    return v

def df_to_results(df: pd.DataFrame):
    results = {r: {sc: {} for sc in SCENARIOS} for r in ALL_REGIONS}

    for region in ALL_REGIONS:
        for sc in SCENARIOS:
            npp_mat, std_mat = [], []

            for m in MODELS:
                sub_npp = df[(df.region == region) & (df.scenario == sc) & (df.model == m) & (df.metric == "npp")]
                sub_std = df[(df.region == region) & (df.scenario == sc) & (df.model == m) & (df.metric == "std")]
                npp_mat.append(_safe_period_vector(sub_npp, PERIODS))
                std_mat.append(_safe_period_vector(sub_std, PERIODS))

            npp_mat = np.vstack(npp_mat).astype(float)
            std_mat = np.vstack(std_mat).astype(float)

            sub_mmm_npp = df[(df.region == region) & (df.scenario == sc) & (df.model == "MMM") & (df.metric == "npp")]
            sub_mmm_std = df[(df.region == region) & (df.scenario == sc) & (df.model == "MMM") & (df.metric == "std")]

            results[region][sc] = {
                "npp_models_dec": npp_mat,
                "std_models_dec": std_mat,
                "npp_mean": _safe_period_vector(sub_mmm_npp, PERIODS),
                "std_mean": _safe_period_vector(sub_mmm_std, PERIODS),
            }

    return results

def _plot_region_metric_one_scenario(
    ax, rd, scenario, region_name,
    metric_base, y_label, y_lim,
    show_ylabel=True,
    hide_xlabels_only=False
):
    idx_hist = np.array([0, 1], dtype=int)
    idx_ssp  = np.arange(2, len(PERIODS))

    x = np.arange(len(PERIODS))
    sc_color  = SCENARIO_COLORS[scenario]
    sc_marker = SCENARIO_MARKERS[scenario]

    models_dec = np.asarray(rd[f"{metric_base}_models_dec"], dtype=float)
    mean_vals  = np.asarray(rd[f"{metric_base}_mean"], dtype=float)

    if models_dec.shape != (len(MODELS), len(PERIODS)):
        raise ValueError(f"[PLOT] models_dec shape error: {region_name}-{scenario}-{metric_base} {models_dec.shape}")
    if mean_vals.shape != (len(PERIODS),):
        raise ValueError(f"[PLOT] mean_vals shape error: {region_name}-{scenario}-{metric_base} {mean_vals.shape}")

    offsets_ssp = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2.0) * (BAR_WIDTH_SSP + BAR_GAP_SSP)

    for i, m in enumerate(MODELS):
        ax.bar(
            x[idx_ssp] + offsets_ssp[i],
            models_dec[i, idx_ssp],
            width=BAR_WIDTH_SSP,
            color=sc_color, alpha=SSP_BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_LW,
            hatch=MODEL_HATCHES.get(m, None), zorder=2
        )

    ax.bar(
        x[idx_hist], mean_vals[idx_hist],
        width=BAR_WIDTH_HIST,
        color=HIST_COLOR, alpha=HIST_BAR_ALPHA,
        edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_LW, zorder=3
    )

    ax.plot(x[idx_hist], mean_vals[idx_hist], color=HIST_COLOR, marker=HIST_MARKER,
            linewidth=2.0, markersize=HIST_MS, zorder=6)

    ax.plot(x[1:3], mean_vals[1:3], color=sc_color, linewidth=2.0, zorder=7)
    ax.plot(x[2:], mean_vals[2:], color=sc_color, marker=sc_marker,
            linewidth=2.0, markersize=HIST_MS, zorder=5)

    ax.plot([x[1]], [mean_vals[1]], color=HIST_COLOR, marker=HIST_MARKER,
            linestyle="None", markersize=HIST_MS, zorder=12)

    ax.set_xticks(x)
    ax.set_xticklabels([] if hide_xlabels_only else PERIODS, rotation=60, ha="right")

    ax.set_ylim(*y_lim)
    if metric_base == "std":
        ax.yaxis.set_major_locator(MultipleLocator(25))

    ax.set_title(region_name, loc="left", fontsize=TITLE_FS)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    ax.tick_params(labelsize=TICK_FS)

    ax.set_ylabel(y_label if show_ylabel else "", fontsize=LABEL_FS)

def _set_band_y_ticklabels(ax, show_right=False, pad=10):
    if show_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", labelleft=False, labelright=True, pad=pad)
    else:
        ax.tick_params(axis="y", labelleft=False, labelright=False)

def plot_one_figure_npp_and_std(results, figsize=(22, 14), out_png=None, dpi=300, label_color="k"):
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(3, 3, width_ratios=[1, 0.18, 1], wspace=0.0, hspace=0.22)

    row_bboxes = []

    for r, sc in enumerate(SCENARIOS):
        inner_anpp = outer[r, 0].subgridspec(2, 4, height_ratios=[1, 1], width_ratios=[1.4, 1, 1, 1],
                                             wspace=0.1, hspace=0.32)

        show_xlabels = (r == 2)

        ax_nat_anpp = fig.add_subplot(inner_anpp[:, 0])
        _plot_region_metric_one_scenario(
            ax_nat_anpp, results["National-level"][sc], sc, "National-level",
            "npp", "ANPP (gC m$^{-2}$ yr$^{-1}$)", Y_LIM_NPP,
            show_ylabel=True, hide_xlabels_only=(not show_xlabels)
        )

        axes_row = [ax_nat_anpp]
        for idx, region in enumerate(REGION_LABELS):
            rr = idx // 3
            cc = idx % 3 + 1
            ax = fig.add_subplot(inner_anpp[rr, cc])

            hide_xt = (rr == 0) or (not show_xlabels)
            _plot_region_metric_one_scenario(
                ax, results[region][sc], sc, region,
                "npp", "ANPP (gC m$^{-2}$ yr$^{-1}$)", Y_LIM_NPP,
                show_ylabel=False, hide_xlabels_only=hide_xt
            )
            _set_band_y_ticklabels(ax, show_right=(region in ["Band 3", "Band 6"]), pad=12)
            axes_row.append(ax)

        inner_std = outer[r, 2].subgridspec(2, 4, height_ratios=[1, 1], width_ratios=[1.4, 1, 1, 1],
                                            wspace=0.1, hspace=0.32)

        ax_nat_std = fig.add_subplot(inner_std[:, 0])
        _plot_region_metric_one_scenario(
            ax_nat_std, results["National-level"][sc], sc, "National-level",
            "std", "IAV (gC m$^{-2}$ yr$^{-1}$)", Y_LIM_STD,
            show_ylabel=True, hide_xlabels_only=(not show_xlabels)
        )
        axes_row.append(ax_nat_std)

        for idx, region in enumerate(REGION_LABELS):
            rr = idx // 3
            cc = idx % 3 + 1
            ax = fig.add_subplot(inner_std[rr, cc])

            hide_xt = (rr == 0) or (not show_xlabels)
            _plot_region_metric_one_scenario(
                ax, results[region][sc], sc, region,
                "std", "IAV (gC m$^{-2}$ yr$^{-1}$)", Y_LIM_STD,
                show_ylabel=False, hide_xlabels_only=hide_xt
            )
            _set_band_y_ticklabels(ax, show_right=(region in ["Band 3", "Band 6"]), pad=12)
            axes_row.append(ax)

        x0 = min(a.get_position().x0 for a in axes_row)
        y0 = min(a.get_position().y0 for a in axes_row)
        x1 = max(a.get_position().x1 for a in axes_row)
        y1 = max(a.get_position().y1 for a in axes_row)
        row_bboxes.append((x0, y0, x1, y1))

    for r, sc in enumerate(SCENARIOS):
        x0, _, _, y1 = row_bboxes[r]
        fig.text(x0 - 0.02, y1 + 0.022, SCENARIO_TITLES[sc],
                 ha="left", va="bottom", fontsize=15, fontweight="bold", color=label_color)

    legend_handles = [
        mlines.Line2D([], [], color=HIST_COLOR, marker=HIST_MARKER, linestyle='-',
                      linewidth=2.2, markersize=HIST_MS, label="HIST")
    ]
    legend_handles += [
        mlines.Line2D([], [], color=SCENARIO_COLORS[s], marker=SCENARIO_MARKERS[s],
                      linestyle='-', linewidth=2.0, markersize=HIST_MS,
                      label=SCENARIO_TITLES[s].replace("(a) ", "").replace("(b) ", "").replace("(c) ", ""))
        for s in SCENARIOS
    ]
    legend_handles += [
        Patch(facecolor="white", edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_LW,
              hatch=MODEL_HATCHES[m], label=m)
        for m in MODELS
    ]

    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=len(legend_handles),
               frameon=False, fontsize=LEGEND_FS)

    plt.tight_layout(rect=[0.04, 0.06, 0.995, 0.92])

    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        print(f"[Saved] {out_png}")

    plt.show()

def main():
    df = load_or_compute_cache(CACHE_PARQUET, force=FORCE_RECOMPUTE)
    results = df_to_results(df)
    out = r"D:/ForestC/Figure1.png"
    plot_one_figure_npp_and_std(results, figsize=(22, 14), out_png=out, dpi=600)

if __name__ == "__main__":
    main()
