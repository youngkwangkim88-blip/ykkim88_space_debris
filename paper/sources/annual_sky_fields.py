"""Draw supplement Fig. S12 from the retained E12 sky samples (NumPy, Matplotlib, Astropy)."""
from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from astropy import units as u
from astropy.wcs import WCS


def field_wcs(rotation, step, mid):
    x, y, z = rotation
    ra, dec = np.arctan2(z[1],z[0]), np.arcsin(z[2])
    east=np.array([-np.sin(ra),np.cos(ra),0.0])
    north=np.cross(z,east)
    chart=WCS(naxis=2)
    chart.wcs.ctype=["RA---TAN","DEC--TAN"]
    chart.wcs.crpix=[mid+1,mid+1]
    chart.wcs.crval=[np.rad2deg(ra)%360,np.rad2deg(dec)]
    chart.wcs.cd=np.rad2deg(step)*np.array([[east@x,east@y],[north@x,north@y]])
    return chart


def default_data():
    here=Path(__file__).resolve().parent
    path=here/"annual_sky_fields.npz"
    return path if path.is_file() else here.parent/"supplement/annual_sky_fields.npz"


def draw(data, output):
    with np.load(data,allow_pickle=False) as inputs:
        q=inputs["plane_coordinate"]
        step=q[1]-q[0]; mid=len(q)//2
        total={k:sum(inputs[k+"_"+name].astype(float) for name in
                   ("zodiacal","isotropic","dgl","unresolved")) for k in ("faint","bright")}
        declaration=json.loads(inputs["declaration"].item())
        norm=LogNorm(vmin=min(t.min() for t in total.values()),vmax=max(t.max() for t in total.values()))
        fig=plt.figure(figsize=(7.3,4.6))
        grid=fig.add_gridspec(1,2,left=.100,right=.975,bottom=.275,top=.898,wspace=.33)
        for index,(tag,original,date) in enumerate([("faint","weak","4 Aug 2027"),("bright","strong","26 Jul 2027")]):
            rotation=inputs[tag+"_rotation"]
            wcs=field_wcs(rotation,step,mid)
            ax=fig.add_subplot(grid[0,index],projection=wcs)
            img=ax.imshow(total[tag],origin="lower",interpolation="nearest",cmap="viridis",norm=norm)
            ra,dec=ax.coords
            # Set each coordinate's labels on the edge on which it varies most.
            vertical_ra=abs(wcs.wcs.cd[0,1])>abs(wcs.wcs.cd[0,0])
            for coord,label,side in [(ra,"Right ascension","l" if vertical_ra else "b"),
                                     (dec,"Declination","b" if vertical_ra else "l")]:
                coord.set_ticks_position(side)
                coord.set_ticklabel_position(side)
                coord.set_axislabel_position(side)
                coord.set_axislabel(label,fontsize=9,minpad=.35)
                coord.set_format_unit(u.deg)
                coord.set_major_formatter("d")
                coord.set_ticks(spacing=5*u.deg,color="#3b4551",size=3)
                coord.set_ticklabel(size=8,exclude_overlapping=True,pad=3)
            ax.coords.grid(True,color="white",alpha=.3,ls=":",lw=.65)
            ax.set_title(f"({'ab'[index]}) {tag.capitalize()} sky · {date}",loc="left",fontweight="bold",fontsize=10,pad=6)
            ax.plot(mid,mid,"+",mec="black",mew=2.5,ms=9,zorder=5)
            ax.plot(mid,mid,"+",mec="white",mew=1.2,ms=9,zorder=6)
            if tag=="bright":
                ax.contour(inputs["bright_dust"],[25],colors=["#e34c95"],linewidths=1.0)
                ax.contour(inputs["bright_galactic_latitude"],[0],colors=["white"],linestyles=["--"],linewidths=1.25)
                center=rotation@inputs["galactic_center_gcrs"]
                position=center[:2]/center[2]/step+mid
                assert np.all((-.5<position)&(position<len(q)-.5))
                ax.plot(*position,marker="*",markersize=8,markerfacecolor="white",markeredgecolor="black",mew=.7)
            np.testing.assert_allclose(wcs.all_pix2world([[mid,mid]],0)[0],
                                       [np.rad2deg(np.arctan2(rotation[2,1],rotation[2,0]))%360,
                                        np.rad2deg(np.arcsin(rotation[2,2]))],atol=1e-7)
            comps=declaration["cases"][original]["components_scmos_e_px_s"]["E12"]
            np.testing.assert_allclose(total[tag][mid,mid],sum(comps.values()),rtol=2e-7)
        cax=fig.add_axes([.245,.17,.52,.032])
        scale=fig.colorbar(img,cax=cax,orientation="horizontal")
        candidates=np.array([.3,.4,.5,.6,.7,.8,1,1.5,2,3,4,6,8,10,15,20])
        ticks=candidates[(candidates>=norm.vmin)&(candidates<=norm.vmax)]
        if len(ticks)>9: ticks=ticks[::2]
        scale.set_ticks(ticks)
        scale.ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos:f"{val:g}"))
        scale.ax.minorticks_off()
        scale.ax.tick_params(labelsize=8,length=3,pad=2)
        scale.set_label("E12 diffuse sky rate (e⁻ pixel⁻¹ s⁻¹)",fontsize=9,labelpad=3)
        handles=[Line2D([],[],marker="+",ms=7,linestyle="none",color="#34404a",label="Field center"),
                 Line2D([],[],marker="*",ms=8,linestyle="none",color="#34404a",label="Galactic center"),
                 Line2D([],[],lw=1.3,ls="--",color="#34404a",label="Galactic plane"),
                 Line2D([],[],lw=1.2,color="#e34c95",label="Dust cap boundary")]
        fig.legend(handles=handles,loc="lower center",ncol=4,frameon=False,fontsize=8,
                   handlelength=1.4,handletextpad=.4,columnspacing=1.15,bbox_to_anchor=(.52,.005))
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=320,bbox_inches="tight",pad_inches=.04,facecolor="white")
        plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=default_data())
    parser.add_argument("--output",type=Path,default=Path("figures/annual_sky_fields.png"))
    args=parser.parse_args()
    draw(args.data,args.output)
