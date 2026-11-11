"""Draw the reference-plate comparison from the supplied four-second trace.

From the repository root run:
    python paper_figures.py 9
NumPy and Matplotlib are required. The first native frame and the target-only
half-second integration cells are retained inputs; no attitude or phase is reset.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

GAIN = 10.0 / 3.0
DARK = 0.7754788379076724
READ = 3.3
FLAT_LIGHT = 5000.
N_FIELD = 3428**2
X_BEGIN, Y_CENTRE = 32.25, 32.25
X_END = X_BEGIN + 4 * 939.56668616 / 15.75001479


def static_detector(height, width, cropped_width):
    # Same physical maps and fixed map seed as the native v27 sequence.
    rng = np.random.default_rng(8202)
    full = (height, width)
    prnu = rng.normal(1., .01, full)
    log_sigma = np.sqrt(np.log1p(.32**2))
    dsnu = DARK * np.exp(rng.normal(-log_sigma**2/2, log_sigma, full))
    offset = rng.normal(128 * GAIN, 14.6, full)
    return tuple(a[:, :cropped_width] for a in (prnu, dsnu, offset))


def digitise(expected, offset, rng):
    charge = rng.poisson(expected) + rng.normal(0., READ, expected.shape) + offset
    return np.clip(np.rint(charge / GAIN), 0, 4095).astype(np.uint16)


def short_stack(templates, sky, maps):
    flat, dark, offset = maps
    duration = .5
    # A single predeclared realization. The short frames share these duration-matched masters.
    children = np.random.SeedSequence(20261111).spawn(10)

    def master(index, light, count):
        rng = np.random.default_rng(children[index])
        expectation = dark * duration + flat * light
        total = np.zeros_like(expectation)
        for _ in range(count):
            total += GAIN * digitise(expectation, offset, rng)
        return total / count

    d_hat = master(0, 0., 64)
    f_hat = (master(1, FLAT_LIGHT, 32) - d_hat) / FLAT_LIGHT
    assert np.all(f_hat > 0)
    summed = np.zeros_like(flat)
    for stamp, seed in zip(templates, children[2:]):
        science = flat * (stamp + sky * duration) + dark * duration
        dn = digitise(science, offset, np.random.default_rng(seed))
        assert dn.min() > 0 and dn.max() < 4095
        summed += (GAIN * dn - d_hat) / f_hat
    return summed


def profile_sigma(target, sky, reads):
    """One standard deviation per transverse row from the paper's first-order model.

    Equal weights along the shown track and zero outside it. The same finite
    master pair is shared by all reads and the full-field normalisation term
    uses the full detector area. The band is ensemble scatter around the
    expected sky-subtracted profile, not the error of an average of 8 images.
    """
    duration = 4. / reads
    optical = target + 4. * sky
    r2 = READ**2 + GAIN**2 / 12.
    vd = (DARK * duration + r2) / 64.
    vu = (FLAT_LIGHT + DARK * duration + r2) / 32. + vd
    science_variance = np.sum(optical + 4 * DARK + reads * r2, axis=1)
    nx = optical.shape[1]
    u_sum = optical.sum(axis=1)
    dark_variance = reads**2 * vd * nx
    flat_variance = vu / FLAT_LIGHT**2 * ((optical**2).sum(axis=1) - u_sum**2 / N_FIELD)
    cross = -2 * vd / FLAT_LIGHT * reads * u_sum * (1 - nx / N_FIELD)
    return np.sqrt(science_variance + dark_variance + flat_variance + cross)


def draw(data, output):
    plate_file = data / "reference_plate_readout.npz"
    if not plate_file.is_file():
        plate_file = Path(__file__).with_name("reference_plate_readout.npz")
    with np.load(plate_file) as record:
        sub = record["target_half_e"].astype(float)
        sky = float(record["sky_rate"])
        native = record["long_calibrated_e"]
        maps = static_detector(*[native.shape[0], int(record["calibration_width"]), native.shape[1]])
        assert abs(float(record["distance_km"]) - 329.5232806914059) < 1.e-7
    source = sub.sum(axis=0)
    assert abs(source.sum() - 1027.4862328351974) < 1.e-4
    stacked = short_stack(sub, sky, maps)
    with (Path(__file__).resolve().parents[2] / "data/pseudo4_comparison.csv").open(newline="", encoding="utf-8") as f:
        rows = {r["scheme"]:r for r in csv.DictReader(f)
                if r["band"] == "E5" and r["case"] == "weak_date"}
    schemes = ["native_4s","coadd_2x2s","coadd_8x0p5s","coadd_40x0p1s"]
    snr = [float(rows[k]["snr_sum_fixed"]) for k in schemes]
    assert np.allclose(snr, [10., 7.90, 4.23, 1.63], atol=.005)
    assert abs(float(rows["coadd_8x0p5s"]["snr_frames_optimal"]) - 11.33) < .005
    assert all(abs(float(rows[k]["source_e"]) - source.sum()) < 1.e-4 for k in schemes)

    # Integrate each row over the modelled endpoints with a four-pixel PSF margin.
    columns = slice(int(np.floor(X_BEGIN - 4)), int(np.ceil(X_END + 4)) + 1)
    y = np.arange(native.shape[0]) - Y_CENTRE
    shown = (y >= -10) & (y <= 10)
    mean = source[:, columns].sum(axis=1)
    panels = []
    for image, reads in [(native, 1), (stacked, 8)]:
        observed = (image[:, columns] - 4 * sky).sum(axis=1)
        sigma = profile_sigma(source[:, columns], sky, reads)
        panels.append((observed, sigma))
    lo = min(min(profile[shown].min(), (mean - sig)[shown].min()) for profile,sig in panels)
    hi = max(max(profile[shown].max(), (mean + sig)[shown].max()) for profile,sig in panels)
    plt.rcParams.update({"font.size":8.3, "axes.labelsize":8, "axes.titlesize":9,
                         "axes.spines.right":False,"axes.spines.top":False})
    fig = plt.figure(figsize=(7.9,4.55))
    grid = fig.add_gridspec(3,2,height_ratios=[1.12,.56,1.04],hspace=.76,wspace=.20,
                            left=.10,right=.93,top=.94,bottom=.14)
    ax=fig.add_subplot(grid[0,:])
    x=np.arange(4)
    bars=ax.bar(x,snr,width=.40,color=["#1d6571","#578792","#779da5","#a1b9be"])
    ax.axhline(10,color="#a64a21",ls="--",lw=.9)
    ax.text(3.51,10.15,"Threshold 10",ha="right",va="bottom",fontsize=7.8,color="#88441f")
    for bar,value in zip(bars,snr):
        ax.text(bar.get_x()+bar.get_width()/2,value+.20,f"{value:.2f}",ha="center",weight="bold")
    ax.set(title="(a) Expected SNR for the same four-second trajectory",ylabel="Expected SNR",
           xlim=(-.55,3.55),ylim=(0,12),yticks=[0,5,10],xticks=x,
           xticklabels=["1 × 4 s","2 × 2 s","8 × 0.5 s","40 × 0.1 s"])
    ax.title.set_ha("left");ax.title.set_x(0)
    green="#39e780";teal="#086777"
    y0,y1=20,45
    for j,(image,title) in enumerate([(native,"(b) Single 4 s exposure"),(stacked,"(c) Sum of eight 0.5 s exposures")]):
        ax=fig.add_subplot(grid[1,j])
        im=ax.imshow(image[y0:y1],origin="lower",extent=(-.5,image.shape[1]-.5,y0-.5,y1-.5),
                     cmap="gray",vmin=-20,vmax=26,interpolation="nearest",aspect="auto")
        ax.add_patch(Rectangle((X_BEGIN-4,Y_CENTRE-5),X_END-X_BEGIN+8,10,
                               fill=False,ec=green,lw=1.05))
        ax.set_title(title,loc="left",pad=3);ax.set_axis_off()
        ax=fig.add_subplot(grid[2,j])
        observed,sigma=panels[j]
        ax.fill_between(y[shown],(mean-sigma)[shown],(mean+sigma)[shown],color="#b2d1d5",alpha=.85)
        ax.plot(y[shown],observed[shown],color=".19",lw=1.0,marker=".",ms=3.2)
        ax.plot(y[shown],mean[shown],color=teal,lw=1.35,ls="--")
        ax.axhline(0,color=".5",lw=.55)
        ax.set(xlabel="Transverse offset (pixels)",ylim=(lo-.05*(hi-lo),hi+.06*(hi-lo)),xlim=(-10,10))
        if j==0:ax.set_ylabel("Sky-subtracted row sum (e⁻)")
        else:ax.tick_params(labelleft=False)
    # The single colourbar belongs to both strips.
    barax=fig.add_axes([.942,.479,.009,.088])
    fig.colorbar(im,cax=barax,ticks=[-20,0,26])
    barax.tick_params(labelsize=6.8,length=2)
    barax.set_title("e⁻/px",fontsize=6.9,pad=2)
    handles=[Line2D([],[],color=".19",lw=1,marker=".",label="One realization"),
             Line2D([],[],color=teal,lw=1.35,ls="--",label="Expected profile"),
             Patch(facecolor="#b2d1d5",label="Expected ±1 SD")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.51,.005),
               ncol=3,frameon=False,fontsize=7.8)
    fig.savefig(output,dpi=310,bbox_inches="tight",pad_inches=.04)
    plt.close(fig)
    print("Peak expected row:",mean.max(),"; native and summed model SD at peak:",
          *[sigma[mean.argmax()] for _,sigma in panels])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path(__file__).resolve().parents[1]/"supplement")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    draw(args.data,args.output)
