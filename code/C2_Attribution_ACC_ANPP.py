#%%
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

base_path = r"F:\S1_CarbonIAV\Landis_Result_new_ffff_KMA"
eco_path  = r"D:/ForestC/30_calibration/99_calibration_data1_ini수정/ecoregions_F.tif"
OUT_DIR = r"."
os.makedirs(OUT_DIR, exist_ok=True)

FIG_PATH = os.path.join(
    OUT_DIR,
    "Figure2.png"
)

region_ids    = [1, 2, 3, 4, 5, 6]
region_labels = ["National-level"] + [f"Band {i}" for i in region_ids]
model_names = ["canesm5", "cnrm-cm6-1", "cnrm-esm2-1", "ec-earth3"]

linestyle_map = {
    "canesm5":     (0, (6, 2)),
    "cnrm-cm6-1":  (0, (3, 2)),
    "cnrm-esm2-1": (0, (1, 2)),
    "ec-earth3":   (0, (6, 2, 1, 2)),
}

SSP_LIST = ["SSP1-2.6", "SSP3-7.0", "SSP5-8.5"]
SCEN_MAP = {
    "SSP1-2.6": "Landis_Result_SSP126",
    "SSP3-7.0": "Landis_Result_SSP370",
    "SSP5-8.5": "Landis_Result_SSP585",
}

SSP_COLORS = {
    "SSP1-2.6": "#2ca02c",
    "SSP3-7.0": "gold",
    "SSP5-8.5": "#d62728",
}

prefix = "LandisANPP_AG_NPP-"
all_years = np.arange(2001, 2101)

FONT = 12
XFONT = 10
YFONT = 10.5
plt.rcParams.update({
    "font.size": FONT,
    "axes.labelsize": FONT,
    "xtick.labelsize": XFONT,
    "ytick.labelsize": YFONT,
    "legend.fontsize": FONT,
})

Y_LIM  = (-65, 25)
Y_STEP = 10
DPI = 600

MODEL_ALPHA_HIST = 0.18
MODEL_ALPHA_FUT  = 0.60
HIST_COLOR = "black"
HIST_LW_MODEL = 1.4
FUT_LW_MODEL  = 1.5
MMM_LW = 3.0
MARKER_SIZE = 18

REGION_TEXT_X = 0.06
REGION_TEXT_Y = 0.08

def decadal_bins():
    bins = {
        "2000s": np.arange(2001, 2011),
        "2010s": np.arange(2011, 2021),
    }
    for s in range(2021, 2101, 10):
        e = min(s + 9, 2100)
        bins[f"{(s//10)*10}s"] = np.arange(s, e + 1)
    return bins

DECADES   = decadal_bins()
HIST_LABS = ["2000s", "2010s"]
SSP_LABS  = [k for k in DECADES.keys() if k not in HIST_LABS]
COL_LABS  = HIST_LABS + SSP_LABS
xs = np.arange(len(COL_LABS))

def _read_raster(path, ref_shape=None):
    if not os.path.exists(path):
        if ref_shape is None:
            raise FileNotFoundError(path)
        return np.full(ref_shape, np.nan, dtype=float)

    with rasterio.open(path) as src:
        a = src.read(1).astype(float)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        return a

def _get_ref_shape_from_any(model):
    candidates = [
        os.path.join(base_path, model, "Landis_Result_PIC", "20_result"),
        os.path.join(base_path, model, "Landis_Result_SSP126", "20_result"),
        os.path.join(base_path, model, "Landis_Result_SSP370", "20_result"),
        os.path.join(base_path, model, "Landis_Result_SSP585", "20_result"),
    ]
    for folder in candidates:
        for y in all_years:
            f = os.path.join(folder, f"{prefix}{y-2000}.tif")
            if os.path.exists(f):
                with rasterio.open(f) as src:
                    return src.read(1).shape
    raise FileNotFoundError(f"[{model}] No tif found under base_path candidates.")

def region_mask_from_eco(eco, region_name):
    if region_name == "National-level":
        return np.isin(eco, region_ids)
    rid = int(region_name.split()[-1])
    return eco == rid

def spatial_mean_2d(arr2d, mask2d):
    return float(np.nanmean(np.where(mask2d, arr2d, np.nan)))

def factual_folder_ssp_only(model, ssp):
    return os.path.join(base_path, model, SCEN_MAP[ssp], "20_result")

def pic_folder(model):
    return os.path.join(base_path, model, "Landis_Result_PIC", "20_result")

def tif_path(folder, year):
    return os.path.join(folder, f"{prefix}{year-2000}.tif")

with rasterio.open(eco_path) as src:
    eco_mask = src.read(1)

region_masks = {r: region_mask_from_eco(eco_mask, r) for r in region_labels}

acc_npp = {r: {ssp: {} for ssp in SSP_LIST} for r in region_labels}

for m in model_names:
    ref_shape = _get_ref_shape_from_any(m)
    pic_dir = pic_folder(m)
    
    for ssp in SSP_LIST:
        fac_dir = factual_folder_ssp_only(m, ssp)

        series_by_region = {r: [] for r in region_labels}

        for lab in COL_LABS:
            yrs = DECADES[lab]
            acc_yearly = []
            for y in yrs:
                F = _read_raster(tif_path(fac_dir, y), ref_shape=ref_shape)
                P = _read_raster(tif_path(pic_dir, y), ref_shape=ref_shape)

                with np.errstate(divide="ignore", invalid="ignore"):
                    acc = (F - P) / P * 100.0
                    acc[~np.isfinite(acc)] = np.nan
                acc_yearly.append(acc)

            acc_dec_raster = np.nanmean(np.stack(acc_yearly, axis=0), axis=0)

            for r in region_labels:
                val = spatial_mean_2d(acc_dec_raster, region_masks[r])
                series_by_region[r].append(val)

        for r in region_labels:
            acc_npp[r][ssp][m] = np.array(series_by_region[r], dtype=float)

for r in region_labels:
    for ssp in SSP_LIST:
        arrs = [acc_npp[r][ssp][m] for m in model_names]
        acc_npp[r][ssp]["MMM"] = np.nanmean(np.vstack(arrs), axis=0)

nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 10), sharex=False, sharey=True)
axes = axes.flatten()

hist_n  = len(HIST_LABS)
idx_2010 = 1
idx_2020 = hist_n

for i, rgn in enumerate(region_labels):
    ax = axes[i]

    for ssp in SSP_LIST:
        color = SSP_COLORS[ssp]

        for m in model_names:
            y = acc_npp[rgn][ssp][m]

            ax.plot(xs[:hist_n], y[:hist_n],
                    color=HIST_COLOR,
                    linestyle=linestyle_map[m],
                    linewidth=HIST_LW_MODEL,
                    alpha=MODEL_ALPHA_HIST,
                    zorder=2)

            ax.plot(xs[idx_2010:idx_2020+1], y[idx_2010:idx_2020+1],
                    color=color,
                    linestyle=linestyle_map[m],
                    linewidth=FUT_LW_MODEL,
                    alpha=MODEL_ALPHA_FUT,
                    zorder=3)

            ax.plot(xs[idx_2020:], y[idx_2020:],
                    color=color,
                    linestyle=linestyle_map[m],
                    linewidth=FUT_LW_MODEL,
                    alpha=MODEL_ALPHA_FUT,
                    zorder=2)

        y_mmm = acc_npp[rgn][ssp]["MMM"]

        ax.plot(xs[:hist_n], y_mmm[:hist_n],
                color=HIST_COLOR, linestyle="-", linewidth=MMM_LW, alpha=1.0, zorder=4)

        ax.plot(xs[idx_2010:idx_2020+1], y_mmm[idx_2010:idx_2020+1],
                color=color, linestyle="-", linewidth=MMM_LW, alpha=1.0, zorder=5)

        ax.plot(xs[idx_2020:], y_mmm[idx_2020:],
                color=color, linestyle="-", linewidth=MMM_LW, alpha=1.0, zorder=4)

        ax.scatter(xs[:hist_n], y_mmm[:hist_n], color="black", s=MARKER_SIZE,
                   zorder=6, edgecolors="none")
        ax.scatter(xs[idx_2020:], y_mmm[idx_2020:], color=color, s=MARKER_SIZE,
                   zorder=6, edgecolors="none")

    ax.axvline(hist_n - 0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=1)

    ax.set_ylim(*Y_LIM)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(Y_STEP))
    ax.grid(axis="y", alpha=0.25)

    ax.text(REGION_TEXT_X, REGION_TEXT_Y, rgn, transform=ax.transAxes,
            va="bottom", ha="left", fontsize=FONT, fontweight="bold")

for ax in axes[:len(region_labels)]:
    ax.set_xticks(xs)
    ax.set_xticklabels(COL_LABS, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", labelbottom=True, labelsize=XFONT, length=3)

for i, ax in enumerate(axes[:len(region_labels)]):
    if i % ncols != 0:
        ax.set_ylabel("")

for j in range(len(region_labels), nrows * ncols):
    axes[j].axis("off")

ssp_handles   = [Patch(facecolor=SSP_COLORS[ssp], edgecolor="none", label=ssp) for ssp in SSP_LIST]
hist_handle   = Patch(facecolor="black", edgecolor="none", label="HIST")
mmm_handle    = Line2D([], [], color="gray", linestyle="-", linewidth=3.0, label="MMM")
model_handles = [Line2D([], [], color="gray", linestyle=linestyle_map[m], linewidth=2.0, label=m)
                 for m in model_names]

legend_ax = axes[7]
legend_ax.axis("off")
legend_ax.legend(
    handles=[hist_handle] + ssp_handles + [mmm_handle] + model_handles,
    loc="lower center",
    bbox_to_anchor=(0.25, -0.1),
    frameon=False,
    fontsize=10.5,
    ncol=1
)

fig.text(0.01, 0.5, "ACC of ANPP (%)", rotation=90, va="center", ha="center", fontsize=FONT)
plt.tight_layout(rect=[0.02, 0.04, 1, 0.98])
fig.savefig(FIG_PATH, dpi=DPI)
plt.show()