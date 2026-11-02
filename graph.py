r"""
make_figures.py  --  regenerates every non-text item in the midterm report.

Outputs (PDF, vector/hi-res) go to FIGS_DIR and are included in main.tex with
    \includegraphics[...]{figs/<name>.pdf}
    fig_dataset_examples.jpg   Figure 1  (figure*, full width)
    fig_class_balance.pdf      Figure 2  (single column)
    fig_val_curves.pdf         Figure 3  (single column)
    fig_qualitative.png        Figure 4  (figure*, full width)
    results_table.tex          body of Table 2 (\input it inside the table*)
"""
import os, re, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")                       # remove this line if you want plt.show() windows
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================ PART 0: SETTINGS ===============================
DATA_ROOT = "processed_data"   # folder that contains  val/inputs  and  val/outputs
SPLIT     = "val"
FIGS_DIR  = "figs"
os.makedirs(FIGS_DIR, exist_ok=True)

CLASS_NAMES = ["Background", "SRF", "IRF", "PED", "SHRM", "IS/OS"]
PALETTE = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0],
                    [0, 0, 255], [255, 255, 0], [255, 0, 255]], dtype=np.int32)
HEX = ["#000000", "#ff0000", "#00e000", "#0000ff", "#f2e600", "#ff00ff"]   # legend colours
LEGEND = [Patch(facecolor=HEX[i], edgecolor="gray", label=CLASS_NAMES[i]) for i in range(6)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- helpers shared by several figures (same logic as your AMDDataset class) ----
def rgb_to_class(mask_rgb):
    """RGB mask -> integer labels 0..5 by nearest palette colour."""
    m = np.ascontiguousarray(mask_rgb, dtype=np.int32)
    d = np.sum((m[:, :, None, :] - PALETTE[None, None, :, :]) ** 2, axis=-1)
    return np.argmin(d, axis=-1)

def labels_to_rgb(labels):
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for c in range(6):
        out[labels == c] = PALETTE[c]
    return out

def load_pair(stem_file, size_hw):
    """Load one processed (input, mask) pair. stem_file e.g. '100_11_input.png'."""
    H, W = size_hw
    img = cv2.imread(os.path.join(DATA_ROOT, SPLIT, "inputs", stem_file), cv2.IMREAD_GRAYSCALE)
    m_path = os.path.join(DATA_ROOT, SPLIT, "outputs", stem_file.replace("_input", "_output"))
    mask = cv2.cvtColor(cv2.imread(m_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return img, rgb_to_class(mask)

# ===================== PART 1: FIGURE 1 - dataset examples ====================
def fig_dataset_examples(indices=(0, 25, 50), titles=("(a)", "(b)", "(c)"),
                         size_hw=(384, 576), out="fig_dataset_examples.jpg"):
    files = sorted(f for f in os.listdir(os.path.join(DATA_ROOT, SPLIT, "inputs"))
                   if f.endswith(".png") and not f.startswith("."))
    fig, axs = plt.subplots(2, len(indices), figsize=(7.0, 3.35))
    for c, idx in enumerate(indices):
        img, lab = load_pair(files[idx], size_hw)
        axs[0, c].imshow(img, cmap="gray"); axs[0, c].axis("off")
        axs[1, c].imshow(labels_to_rgb(lab)); axs[1, c].axis("off")
        axs[0, c].set_title(titles[c], fontsize=8.5, fontweight="bold")   # edit titles here
    axs[0, 0].text(-0.02, 0.5, "OCT B-scan", transform=axs[0, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=8.5)
    axs[1, 0].text(-0.02, 0.5, "Expert mask", transform=axs[1, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=8.5)
    fig.legend(handles=LEGEND, loc="lower center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    plt.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.09, wspace=0.03, hspace=0.04)
    plt.savefig(os.path.join(FIGS_DIR, out), dpi=200, pil_kwargs={"quality": 92}); plt.close()   # JPEG: small + fast to compile

# ===================== PART 2: FIGURE 2 - class balance =======================
# Ground-truth pixel counts of the 703 validation slices at 512x512
# (row sums of Model 5's confusion matrix). Replace if your split/size changes.
GT_COUNTS = {"Background": 177799633, "SRF": 1591059, "IRF": 409650,
             "PED": 1374972, "SHRM": 2120770, "IS/OS": 991148}

def count_pixels_from_masks(size_hw=(512, 512)):
    """Optional: recompute GT_COUNTS directly from the mask files."""
    H, W = size_hw
    counts = np.zeros(6, dtype=np.int64)
    d = os.path.join(DATA_ROOT, SPLIT, "outputs")
    for f in sorted(os.listdir(d)):
        if not f.endswith(".png") or f.startswith("."):
            continue
        m = cv2.cvtColor(cv2.imread(os.path.join(d, f)), cv2.COLOR_BGR2RGB)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        counts += np.bincount(rgb_to_class(m).ravel(), minlength=6)
    return dict(zip(CLASS_NAMES, counts.tolist()))

def fig_class_balance(counts=GT_COUNTS, out="fig_class_balance.pdf"):
    tot = sum(counts.values())
    les = {k: v for k, v in counts.items() if k != "Background"}
    ls = sum(les.values())
    b = [100 * counts["Background"] / tot, 100 * ls / tot]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.25),
                            gridspec_kw={"width_ratios": [0.5, 1], "wspace": 0.6})
    axs[0].bar(["Bkg", "Lesion"], b, color=["#555555", "#C8333B"], width=0.7)
    for i, v in enumerate(b):
        axs[0].text(i, v + 2.5, f"{v:.1f}%", ha="center", fontsize=7)
    axs[0].set_xlim(-0.7, 1.7); axs[0].set_ylim(0, 118)
    axs[0].set_yticks([0, 25, 50, 75, 100]); axs[0].set_ylabel("% of all pixels")
    axs[0].tick_params(axis="x", labelsize=7)
    names = list(les); vals = [100 * les[k] / ls for k in names]
    axs[1].bar(names, vals, color=[HEX[CLASS_NAMES.index(k)] for k in names],
               edgecolor="gray", width=0.7)
    for i, v in enumerate(vals):
        axs[1].text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=7)
    axs[1].set_ylim(0, 40); axs[1].set_ylabel("% of lesion pixels")
    axs[1].tick_params(axis="x", labelsize=7)
    plt.setp(axs[1].get_xticklabels(), rotation=35, ha="right")      # keeps SHRM / IS/OS from touching
    plt.savefig(os.path.join(FIGS_DIR, out), bbox_inches="tight"); plt.close()
    return b, dict(zip(names, vals))

# ===================== PART 3: FIGURE 3 - validation curves ===================
# Validation mean Dice (%) per epoch, copied from each notebook's training log.
VAL_DICE = {
 "Baseline U-Net": [51.39,57.9,51.3,54.69,55.04,54.41,58.09,51.14,61.54,61.57,60.12,66.2,62.38,67.32,60.81,59.88,63.68,59.53,57.13,64.12,64.9,64.38,65.39,65.19,65.57,65.53,66.5,65.69,66.79,66.02,65.27,66.21,65.67,65.62,65.39,64.52,64.7,64.98,64.86,65.13],
 "Model 1": [45.79,55.66,48.57,49.29,53.94,55.13,57.83,60.9,64.08,59.57,63.09,64.87,61.93,64.92,63.29,65.06,63.7,66.06,61.6,65.97,65.98,66.87,67.1,66.7,67.26,66.74,65.49,66.94,65.51,65.72,66.16,66.42,65.75,66.51,66.6],
 "Model 2": [34.38,46.47,64.35,62.02,70.89,73.4,74.63,71.86,76.07,70.03,75.84,76.57,74.01,76.08,77.76,79.06,76.07,78.05,78.02,77.96],
 "Model 3": [52.95,54.46,74.42,72.77,73.53,74.33,76.54,74.34,76.91,77.88,77.36,76.74,78.04,77.72,77.45,77.48,77.01,77.91,78.25,78.08],
 "Model 4": [56.1,75.93,75.57,75.66,74.35,75.73,77.56,78.07,78.84,79.15,78.8,79.48,78.97,79.26,78.81,78.46,79.29,79.46,79.17,79.14],
 "Model 5": [4.33,9.42,24.67,46.14,54.28,53.94,56.93,66.89,74.45,72.96,76.56,75.68,76.75,75.96,76.05,75.73,76.72,76.55,78.04,77.08,76.95,77.86,78.47,79.14,78.37,78.0,78.63,79.17,78.59,78.34,79.61,78.7,79.37,79.13,79.24],
}
LINE_COLORS = {"Baseline U-Net": "#7f7f7f", "Model 1": "#bcbd22", "Model 2": "#1f77b4",
               "Model 3": "#2ca02c", "Model 4": "#d62728", "Model 5": "#9467bd"}

def parse_curves_from_notebooks(paths):
    """Optional: rebuild VAL_DICE from the saved outputs of your .ipynb files.
       paths = {"Model 5": "model_5_complete.ipynb", ...}"""
    out = {}
    for name, p in paths.items():
        with open(p) as fh:
            nb = json.load(fh)
        txt = ""
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    txt += "".join(o["text"])
        res = {}
        for chunk in re.split(r"(?=(?:---|===)\s*Epoch\s+\d+)", txt):
            m = re.match(r"(?:---|===)\s*Epoch\s+(\d+)", chunk)
            d = re.search(r"Mean Dice[^0-9\n]*([\d.]+)\s*%", chunk)
            if m and d:
                res[int(m.group(1))] = float(d.group(1))
        out[name] = [res[e] for e in sorted(res)]
    return out

def fig_val_curves(curves=VAL_DICE, ref=78.55, ref_label="AMD-SD U-Net\n(78.55)",
                   out="fig_val_curves.pdf"):
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for k, v in curves.items():
        ax.plot(range(1, len(v) + 1), v, color=LINE_COLORS.get(k), lw=1.3, label=k)
    ax.axhline(ref, color="k", ls=":", lw=0.9)
    ax.text(40.3, ref + 0.4, ref_label, fontsize=6.5, ha="right", va="bottom")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation mean Dice (%)")
    ax.set_ylim(0, 88); ax.set_xlim(1, 40)
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="lower right")
    plt.savefig(os.path.join(FIGS_DIR, out), bbox_inches="tight"); plt.close()

# ===================== PART 4: FIGURE 4 - qualitative example =================
def fig_qualitative(img, gt_labels, pred_labels, pred_title="Model 5 prediction",
                    out="fig_qualitative.png"):
    """img: HxW uint8 (grayscale) | gt_labels, pred_labels: HxW ints in 0..5"""
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55))
    panels = [(img, "Input OCT B-scan", "gray"),
              (labels_to_rgb(gt_labels), "Expert mask", None),
              (labels_to_rgb(pred_labels), pred_title, None)]
    for ax, (im, t, cmap) in zip(axs, panels):
        ax.imshow(im, cmap=cmap); ax.axis("off")
        ax.set_title(t, fontsize=8.5, fontweight="bold")
    fig.legend(handles=LEGEND, loc="lower center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.1, wspace=0.03)
    plt.savefig(os.path.join(FIGS_DIR, out), dpi=200); plt.close()

# --- how to call it: paste this inside your Model 5 notebook (model + data exist there) ---
# import torch
# img_t, mask_t = val_ds[IDX][0], val_ds[IDX][1]          # (1,H,W) float in [0,1], (H,W) long
# with torch.no_grad():
#     logits = model(img_t.unsqueeze(0).to(device))[0]    # dual-head models return (multiclass, binary)
# pred = logits.argmax(1).squeeze(0).cpu().numpy()
# fig_qualitative((img_t[0].numpy()*255).astype("uint8"), mask_t.numpy(), pred)

# ===================== PART 5: TABLE 2 - results (LaTeX) ======================
# name, params (M) or None, Dice, IoU, SRF, IRF, PED, SHRM, IS/OS   (all in %)
ROWS = [
 ("AMD-SD U-Net (paper)",   None,   78.55, 64.48, 78.65, 78.66, 79.62, 72.04, 83.76),
 ("AMD-SD ResUNet (paper)", None,   79.81, 66.58, 78.86, 81.16, 81.88, 72.97, 84.19),
 "---",   # horizontal rule between published rows and your models
 ("Baseline U-Net",              7.76, 67.32, 59.59, 67.40, 79.44, 55.39, 59.53, 74.83),
 ("Model 1: dual-head U-Net",    7.79, 67.26, 59.47, 65.86, 79.19, 58.54, 57.79, 74.94),
 ("Model 2: cascaded U-Nets",   62.07, 79.06, 65.51, 78.57, 77.87, 81.22, 73.63, 84.03),
 ("Model 3: dual-decoder U-Net",10.90, 78.25, 64.45, 78.60, 79.06, 78.59, 71.19, 83.80),
 ("Model 4: TransUNet",        105.92, 79.48, 66.09, 81.05, 77.30, 81.81, 73.69, 83.55),
 ("Model 5: state-space U-Net", 7.67, 79.61, 66.32, 80.45, 81.11, 80.73, 71.59, 84.16),
]
HEADER = ["Model", "Params (M)", "Dice", "IoU", "SRF", "IRF", "PED", "SHRM", "IS/OS"]

def make_results_table(rows=ROWS, out="results_table.tex"):
    """Writes the tabular; bold = best of YOUR models (rows after the '---' separator)."""
    data = [r for r in rows if r != "---"]
    ours = rows[rows.index("---") + 1:] if "---" in rows else data
    best = [max(r[j] for r in ours) for j in range(2, 9)]        # best Dice..IS/OS
    L = ["\\begin{tabular}{lcccccccc}", "\\toprule",
         " & ".join(f"\\textbf{{{h}}}" for h in HEADER) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "---":
            L.append("\\midrule"); continue
        cells = [r[0], "--" if r[1] is None else f"{r[1]:.2f}"]
        for j, v in enumerate(r[2:]):
            s = f"{v:.2f}"
            cells.append(f"\\textbf{{{s}}}" if (r in ours and v == best[j]) else s)
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(FIGS_DIR, out), "w").write("\n".join(L) + "\n")

# =============================== RUN EVERYTHING ===============================
if __name__ == "__main__":
    fig_dataset_examples()            # needs processed_data/val/{inputs,outputs}
    fig_class_balance()               # uses GT_COUNTS (or pass count_pixels_from_masks())
    fig_val_curves()                  # uses VAL_DICE
    make_results_table()              # writes figs/results_table.tex
    # fig_qualitative(...) is called from the Model 5 notebook (see PART 4)
    print("Done ->", os.path.abspath(FIGS_DIR))
