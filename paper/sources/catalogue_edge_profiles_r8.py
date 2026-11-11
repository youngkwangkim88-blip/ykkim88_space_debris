"""Draw Fig. 10 and verify the calibrated pixels from the compact simulation crops."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "supplement"
EXAMPLES = SUP / "catalogue_edge_pixels_r8.npz"
RESULT = ROOT / "figures" / "catalogue_edge_profiles_r8.png"
SIDE = 81


def draw(output=RESULT):
    with np.load(EXAMPLES,allow_pickle=False) as data:
        meta = json.loads(data["metadata_json"].item())
        fig,axs = plt.subplots(1,2,figsize=(7.0,3.12),layout="constrained",sharey=True)
        # A fixed shared display range includes every calibrated pixel of both crops.
        norm = AsinhNorm(linear_width=6,vmin=-15,vmax=910)
        for i,(ax,rec) in enumerate(zip(axs,meta)):
            calibrated = data[f"calibrated_e_{i}"]
            recomputed = (data[f"raw_dn_{i}"].astype(float)*rec["gain_e_per_dn"]
                          - data[f"master_dark_e_{i}"])/data[f"normalized_flat_{i}"]
            valid = data[f"valid_master_{i}"]
            np.testing.assert_array_equal(recomputed[valid].astype("f4"),calibrated[valid])
            im=ax.imshow(calibrated,origin="lower",interpolation="nearest",cmap="gray",norm=norm)
            xy=data[f"path_relative_xy_{i}"]
            tangent = xy[-1] - xy[0]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1],tangent[0]])
            along = xy @ tangent
            cross = xy @ normal
            amin,amax = along.min()-4,along.max()+4
            cmin,cmax = cross.min()-5,cross.max()+5
            corners = np.array([a*tangent+c*normal for a,c in
                                [(amin,cmin),(amax,cmin),(amax,cmax),(amin,cmax),(amin,cmin)]])
            ax.plot(*corners.T,color="#39e780",linewidth=1.1)
            ax.set(xlim=(-.5,SIDE-.5),ylim=(-.5,SIDE-.5),aspect="equal",xticks=[0,20,40,60,80],
                   yticks=[0,20,40,60,80],xlabel="Local column (pixel)")
            surface = {"CFRP_SOLAR": "solar-cell reflectance model", "TI_STEEL_CU": "metal class, aluminum model"}[rec["material"]]
            side = "Near" if i == 0 else "Far"
            snr_label = f"{rec['snr']:.3f}" if rec["snr"] < 1 else f"{rec['snr']:.1f}"
            title = (f"({'ab'[i]}) {side} virtual fragment · {100 * rec['lc_m']:.1f} cm\n{surface}\n"
                     f"{rec['range_km']:,.1f} km · {rec['trail_px']:,.0f} px\nexpected aperture SNR {snr_label}")
            ax.set_title(title, fontsize=7.8, loc="left", linespacing=1.35, pad=6)
            ax.tick_params(labelsize=8)
        axs[0].set_ylabel("Local row (pixel)",fontsize=9)
        cbar=fig.colorbar(im,ax=axs,shrink=.8,pad=.03)
        cbar.set_ticks([-10,0,10,100,900],labels=["−10","0","10","100","900"])
        cbar.set_label("Calibrated e⁻/pixel",fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        fig.savefig(output,dpi=290,bbox_inches="tight",pad_inches=.03)
        plt.close(fig)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from redraw_figures import STYLE
    with plt.rc_context(STYLE):
        draw(args.output)
