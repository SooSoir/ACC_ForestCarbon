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
    "Figure3.png"
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

Y_LIM  = (-100, 240)
Y_STEP = 40
DPI = 600

MODEL_ALPHA_HIST = 0.18
MODEL_ALPHA_FUT  = 0.60
HIST_COLOR = "black"
HIST_LW_MODEL = 1.4
FUT_LW_MODEL  = 1.5
MMM_LW = 3.0
MARKER_SIZE = 18
REGION_TEXT_X = 0.06
REGION_TEXT_Y = 0.92

def decade_windows():
    wins = []
    wins.append(("2000s", np.arange(2001, 2011)))
    wins.append(("2010s", np.arange(2011, 2021)))
    for s in range(2021, 2101, 10):
        e = min(s + 9, 2100)
        wins.append((f"{s//10*10}s", np.arange(s, e + 1)))
    return wins

DECADE_WINS = decade_windows()
COL_LABS = [lab for lab, _ in DECADE_WINS]
xs = np.arange(len(COL_LABS))

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        profile = src.profile
    return arr, profile

def stack_decade(folder, years, ref_profile=None):
    arrays = []
    profile = ref_profile
    for y in years:
        f = os.path.join(folder, f"{prefix}{y-2000}.tif")
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing raster: {f}")
        a, prof = read_raster(f)
        if profile is None:
            profile = prof
        arrays.append(a)
    return np.stack(arrays, axis=0), profile

def detrended_sd_per_cell(years, y3d):
    x = years.astype(float)
    x_mean = np.mean(x)
    x_c = x - x_mean
    var_x = np.mean(x_c**2)

    y_mean = np.nanmean(y3d, axis=0)
    cov = np.nanmean(x_c[:, None, None] * (y3d - y_mean[None, :, :]), axis=0)

    slope = cov / var_x
    intercept = y_mean - slope * x_mean

    trend = slope[None, :, :] * x[:, None, None] + intercept[None, :, :]
    y_detr = y3d - trend + y_mean[None, :, :]

    return np.nanstd(y_detr, axis=0)

def region_mean_from_raster(arr2d, eco, region_name):
    if region_name == "National-level":
        mask = np.isin(eco, region_ids)
    else:
        rid = int(region_name.split()[-1])
        mask = (eco == rid)
    return np.nanmean(np.where(mask, arr2d, np.nan))

def folder_pic(base_path, model):
    return os.path.join(base_path, model, "Landis_Result_PIC", "20_result")

def folder_ssp(base_path, model, ssp):
    return os.path.join(base_path, model, SCEN_MAP[ssp], "20_result")

with rasterio.open(eco_path) as s:
    eco_mask = s.read(1)

acc_ts = {r: {ssp: {} for ssp in SSP_LIST} for r in region_labels}
pic_sd_cache = {m: {} for m in model_names}
fact_sd_cache = {m: {ssp: {} for ssp in SSP_LIST} for m in model_names}

for m in model_names:
    pic_fold = folder_pic(base_path, m)
    ref_profile = None
    for lab, yrs in DECADE_WINS:
        y3d, ref_profile = stack_decade(pic_fold, yrs, ref_profile=ref_profile)
        pic_sd_cache[m][lab] = detrended_sd_per_cell(yrs, y3d)

    for ssp in SSP_LIST:
        fact_fold = folder_ssp(base_path, m, ssp)
        for lab, yrs in DECADE_WINS:
            y3d, _ = stack_decade(fact_fold, yrs, ref_profile=ref_profile)
            fact_sd_cache[m][ssp][lab] = detrended_sd_per_cell(yrs, y3d)

for ssp in SSP_LIST:
    for m in model_names:
        series_by_region = {r: [] for r in region_labels}

        for lab, _yrs in DECADE_WINS:
            fact_sd = fact_sd_cache[m][ssp][lab]
            pic_sd  = pic_sd_cache[m][lab]

            acc2d = np.where(
                np.isfinite(pic_sd) & (pic_sd != 0) & np.isfinite(fact_sd),
                (fact_sd - pic_sd) / pic_sd * 100.0,
                np.nan
            )

            for r in region_labels:
                series_by_region[r].append(region_mean_from_raster(acc2d, eco_mask, r))

        for r in region_labels:
            acc_ts[r][ssp][m] = np.array(series_by_region[r], dtype=float)

for r in region_labels:
    for ssp in SSP_LIST:
        mat = np.vstack([acc_ts[r][ssp][m] for m in model_names])
        acc_ts[r][ssp]["MMM"] = np.nanmean(mat, axis=0)

nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 10), sharex=False, sharey=True)
axes = axes.flatten()

hist_n = 2
idx_2010 = 1
idx_2020 = hist_n

for i, r in enumerate(region_labels):
    ax = axes[i]

    for ssp in SSP_LIST:
        color = SSP_COLORS[ssp]

        for m in model_names:
            y = acc_ts[r][ssp][m]

            ax.plot(xs[:hist_n], y[:hist_n],
                    color=HIST_COLOR, linestyle=linestyle_map[m],
                    linewidth=HIST_LW_MODEL, alpha=MODEL_ALPHA_HIST, zorder=2)

            ax.plot(xs[idx_2010:idx_2020+1], y[idx_2010:idx_2020+1],
                    color=color, linestyle=linestyle_map[m],
                    linewidth=FUT_LW_MODEL, alpha=MODEL_ALPHA_FUT, zorder=3)

            ax.plot(xs[idx_2020:], y[idx_2020:],
                    color=color, linestyle=linestyle_map[m],
                    linewidth=FUT_LW_MODEL, alpha=MODEL_ALPHA_FUT, zorder=2)

        y_mmm = acc_ts[r][ssp]["MMM"]

        ax.plot(xs[:hist_n], y_mmm[:hist_n],
                color=HIST_COLOR, linestyle="-", linewidth=MMM_LW, alpha=1.0, zorder=4)

        ax.plot(xs[idx_2010:idx_2020+1], y_mmm[idx_2010:idx_2020+1],
                color=color, linestyle="-", linewidth=MMM_LW, alpha=1.0, zorder=5)

        ax.plot(xs[idx_2020:], y_mmm[idx_2020:],
                color=color, linestyle="-", linewidth=MMM_LW, alpha=1.0, zorder=4)

        ax.scatter(xs[:hist_n], y_mmm[:hist_n], color="black", s=MARKER_SIZE, zorder=6, edgecolors="none")
        ax.scatter(xs[idx_2020:], y_mmm[idx_2020:], color=color, s=MARKER_SIZE, zorder=6, edgecolors="none")

    ax.axvline(hist_n - 0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=1)

    ax.set_ylim(*Y_LIM)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(Y_STEP))
    ax.grid(axis="y", alpha=0.25)

    ax.text(REGION_TEXT_X, REGION_TEXT_Y, r, transform=ax.transAxes,
            va="top", ha="left", fontsize=FONT, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(COL_LABS, rotation=45, ha="right")
    ax.tick_params(axis="x", which="both", labelbottom=True, labelsize=XFONT, length=3)

for i, ax in enumerate(axes[:len(region_labels)]):
    if i % ncols != 0:
        ax.set_ylabel("")

for j in range(len(region_labels), nrows * ncols):
    axes[j].axis("off")

ssp_handles = [Patch(facecolor=SSP_COLORS[ssp], edgecolor="none", label=ssp) for ssp in SSP_LIST]
hist_handle = Patch(facecolor="black", edgecolor="none", label="HIST")
mmm_handle  = Line2D([], [], color="gray", linestyle="-", linewidth=3.0, label="MMM")
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

fig.text(0.01, 0.5, "ACC of IAV (%)", rotation=90, va="center", ha="center", fontsize=FONT)

plt.tight_layout(rect=[0.02, 0.04, 1, 0.98])
fig.savefig(FIG_PATH, dpi=DPI)
plt.show()
