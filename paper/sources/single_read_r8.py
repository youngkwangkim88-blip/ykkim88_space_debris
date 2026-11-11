"""Reproduce the supplement's single-read 0.5 s E12 near-threshold pair.

Pass the unpacked public A1 reproduction release (v30) using --release.
Use --plot-only to render the retained calibrated native pixels without scipy or POPPY.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUP = Path(__file__).resolve().parents[1] / "supplement"
ARCHIVE = SUP / "single_read_r8.npz"
EXPOSURE = 0.5


def simulate(release, archive=ARCHIVE):
    sys.path.insert(0,str(release.resolve()))
    from scipy.constants import astronomical_unit
    import figure32
    import fragment as model
    import sky

    state = model.setup("E12")
    state.initial_attitudes = state.initial_attitudes[:1]
    state.spin_axes = state.spin_axes[:1]
    times,q = model.attitudes(state,"plate",10.)
    keep = times < EXPOSURE
    scenes = sky.sightlines()
    aug = scenes["weak"]
    rate = model.factor(state,q[:,keep],"plate","grey",aug["target_phase_far_sun_deg"])
    rate *= (astronomical_unit/1000/aug["sun_distance_km"])**2
    intrinsic = model.render_fft(state,rate,times[keep],EXPOSURE,model.PAPER_SPEED).reshape(65,-1)
    moments = [np.sum(intrinsic**n) for n in range(1,5)]
    backgrounds = np.array([sum(sky.background(scenes[k],"E12").values()) for k in ("weak","strong")])
    distance,scale = model.threshold(moments,100.,EXPOSURE,backgrounds[0],model.NATIVE_PIXELS)
    source = intrinsic*scale
    snr = np.array([model.calibrated_snr(moments,EXPOSURE,level,scale) for level in backgrounds])
    predicted = np.array([distance,source.sum(),*snr])
    error = predicted - [117.448,444.81,10.,9.676]
    if np.any(abs(error) > [.0006,.006,1e-7,.0006]):
        raise AssertionError(f"Single-exposure optical calculation: {predicted}")
    # Draw one spatial response and the 0.5 s shared dark and flat masters.
    rng = np.random.default_rng(8202)
    shape = source.shape
    prnu = rng.normal(1,.01,shape)
    sigma = np.sqrt(np.log1p(.32**2))
    dark_rate = figure32.DARK_E_PER_PIXEL_S * np.exp(rng.normal(-sigma**2/2,sigma,shape))
    bias = rng.normal(128*figure32.GAIN_E_PER_DN,14.6,shape)
    def master(n,lamp):
        expected = prnu*lamp+dark_rate*EXPOSURE
        sum_dn = np.zeros(shape)
        for _ in range(n):
            sum_dn += figure32.read_detector(rng,expected,bias)
        return sum_dn*figure32.GAIN_E_PER_DN/n
    dark = master(64,0.)
    flat = (master(32,figure32.FLAT_E)-dark)/figure32.FLAT_E
    assert np.all(flat>0)
    raw = []
    seeds = [180211,180212]
    for bg,seed in zip(backgrounds,seeds):
        raw.append(figure32.read_detector(np.random.default_rng(seed),
                   prnu*(source+bg*EXPOSURE)+dark_rate*EXPOSURE,bias))
    raw = np.stack(raw)
    assert np.all((raw>0) & (raw<4095))
    cal = (raw*figure32.GAIN_E_PER_DN-dark)/flat
    xstart=32.25; scale_arcsec=206265*2.8e-6/state.focal_m
    meta=dict(range_km=float(distance),total_source_e=float(source.sum()),
              expected_snr=snr.tolist(), exposure_s=EXPOSURE,sky_e_pixel_s=backgrounds.tolist(),
              plate_scale_arcsec_pixel=scale_arcsec, speed_arcsec_s=model.PAPER_SPEED,phase_deg=aug["target_phase_far_sun_deg"],
              shutter_xy=[[xstart,32.25],[xstart+EXPOSURE*model.PAPER_SPEED/scale_arcsec,32.25]],
              master_seed=8202,science_seeds=seeds, dark_frames=64, flat_frames=32, gain_e_per_dn=figure32.GAIN_E_PER_DN)
    np.savez_compressed(archive,raw_dn=raw,calibrated_e=cal.astype("f4"),target_e=source.astype("f4"),
                        master_dark_e=dark,normalized_flat=flat,metadata_json=np.array(json.dumps(meta)))


def draw(output, archive=ARCHIVE):
    with np.load(archive,allow_pickle=False) as z:
        meta=json.loads(z["metadata_json"].item())
        cal=z["calibrated_e"]
        verify=(z["raw_dn"]*meta["gain_e_per_dn"]-z["master_dark_e"])/z["normalized_flat"]
        np.testing.assert_array_equal(verify.astype("f4"),cal)
        fig,axs=plt.subplots(1,2,figsize=(7.45,2.72),layout="constrained",sharey=True)
        for i,ax in enumerate(axs):
            im=ax.imshow(cal[i],origin="lower",interpolation="nearest",cmap="gray",vmin=-8,vmax=25)
            for x,y in meta["shutter_xy"]:
                ax.plot([x,x],[y-7,y-4],color="#ffb755",lw=1.4)
            label=("Fainter sky","Brighter sky")[i]
            ax.set_title(f"({'ab'[i]})  {label}    SNR = {meta['expected_snr'][i]:.3f}",loc="left",fontsize=10,weight="bold")
            ax.set(xlim=(9.5,84.5),ylim=(15.5,48.5),xticks=[10,30,50,70],yticks=[16,32,48],xlabel="Native column (pixel)")
        axs[0].set_ylabel("Native row (pixel)")
        cbar=fig.colorbar(im,ax=axs,location="bottom",fraction=.055,aspect=44,pad=.1)
        cbar.set_label("Calibrated electrons per native pixel")
        fig.savefig(output,dpi=280,bbox_inches="tight",pad_inches=.03)
        plt.close(fig)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--release",type=Path)
    mode.add_argument("--plot-only",action="store_true")
    parser.add_argument("--output", type=Path, default=SUP.parent / "figures/single_read_r8.png")
    parser.add_argument("--archive", type=Path, default=ARCHIVE, help="input or regenerated frame archive")
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.release:
        simulate(args.release, args.archive)
    draw(args.output, args.archive)
