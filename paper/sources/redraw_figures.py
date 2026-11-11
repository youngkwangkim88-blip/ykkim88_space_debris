"""Plot the v30 paper from retained scientific inputs. Run python paper_figures.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, Polygon, Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "supplement"
OUT = HERE.parent / "figures"
PUBLIC_DATA = HERE.parents[1] / "data"
EXTRA = HERE / "ijass_figure_inputs.npz"
TEAL, ORANGE, GREY = "#006c75", "#ad4c16", "#475764"
STYLE = {"font.family":"DejaVu Sans", "font.size":8.6, "axes.titlesize":9,
         "axes.spines.top":False, "axes.spines.right":False, "axes.axisbelow":True,
         "legend.frameon":False, "legend.fontsize":8, "axes.labelsize":8.6}


def save(fig, name):
    fig.savefig(OUT/(name+".png"),dpi=320,bbox_inches="tight",pad_inches=.04,facecolor="white")
    plt.close(fig)


def geometry():
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.15),gridspec_kw={"width_ratios":[1.24,.9,.94]})
    fig.subplots_adjust(left=.025,right=.985,wspace=.16,top=.86,bottom=.07)
    for a,title in zip(axs,("(a) Observation geometry","(b) Surface geometry","(c) Controlled shapes")):
        a.set_aspect("equal");a.axis("off");a.set_title(title,loc="left",weight="bold",pad=8)
    def arrow(a,start,end,color=GREY):
        a.annotate("",end,start,arrowprops=dict(arrowstyle="-|>",lw=1.05,color=color,shrinkA=0,shrinkB=0))
    def arc(a,origin,radius,lo,hi):
        a.add_patch(Arc(origin,2*radius,2*radius,theta1=lo,theta2=hi,color=GREY,lw=.8))
    a=axs[0];a.set(xlim=(-1.,3.35),ylim=(-.48,2.62))
    a.add_patch(Ellipse((-.12,.05),1.72,1.10,angle=24,fc="none",ec="#aab6bc",lw=.85,ls="--"))
    a.add_patch(Circle((-.24,.05),.39,fc="#d8e9ef",ec="#557a88"));a.text(-.24,.05,"Earth",ha="center",va="center",fontsize=7)
    o=np.array([.64,.34]);d=np.array([2.68,1.45]);sun=np.array([-.37,.94])
    phi=np.degrees(np.arctan2(*(d-o)[::-1]));psi=np.degrees(np.arctan2(*sun[::-1]))
    arrow(a,o,d);arrow(a,o,o+[1.72,0]);arrow(a,d,d+sun,color=ORANGE);arrow(a,o,o+sun*.96,color=ORANGE)
    a.scatter(*o,marker="s",c=GREY,s=15);a.scatter(*d,c=TEAL,s=16)
    a.text(*((o+d)/2+[-.12,.13]),r"$d$",ha="center")
    a.text(*(o+[-.06,-.21]),"O");a.text(*(d+[.11,-.02]),"D")
    a.text(*(o+[1.42,-.05]),"Camera axis",ha="center",va="top",fontsize=7)
    a.text(*(d+sun+[.08,0]),"To Sun",color=ORANGE,va="center",fontsize=7.6)
    arc(a,o,.65,0,phi);a.text(*(o+[.75,.08]),"β")
    arc(a,o,.47,phi,psi);a.text(*(o+[.14,.56]),"ε")
    arc(a,d,.37,psi,180+phi);a.text(*(d+[-.56,.1]),"α")
    b=axs[1];b.set(xlim=(-1.45,1.36),ylim=(-.36,2.15))
    b.add_patch(Polygon([[-1,-.12],[.64,-.12],[.95,.1],[-.66,.1]],fc="#d8e9ef",ec="#557a88"))
    for end,color,label,shift in [((0,1.85),GREY,"n",(.08,.00)),((-.91,1.43),ORANGE,"s",(-.17,.01)),((1.00,1.06),TEAL,"v",(.10,.02))]:
        arrow(b,(0,0),end,color);b.text(end[0]+shift[0],end[1]+shift[1],label,color=color,fontsize=10,style="italic")
    arc(b,(0,0),.79,90,122.5);arc(b,(0,0),.63,46.7,90)
    b.text(-.54,.90,r"$\theta_i$");b.text(.27,.72,r"$\theta_o$")
    c=axs[2];c.set(xlim=(-1.2,1.65),ylim=(-.23,2.34))
    def box(x,y,thick,fill):
        u,v=np.array([.79,0]),np.array([.36,.23])
        bottom=np.array([x,y]);p=np.array([bottom,bottom+u,bottom+u+v,bottom+v]);top=p+[0,thick]
        for face,tint in (([p[0],p[1],top[1],top[0]],fill[0]),([p[1],p[2],top[2],top[1]],fill[1]),(top,fill[2])):
            c.add_patch(Polygon(face,fc=tint,ec=GREY,lw=.7))
    box(-1.05,.67,.79,["#c7d9dd","#9bbec2","#e0ecec"])
    box(.40,.67,.79*.125,["#c7d9dd","#9bbec2","#e0ecec"])
    c.text(-.48,.42,"Cube",ha="center",weight="bold");c.text(.98,.42,"Plate",ha="center",weight="bold")
    c.text(-.48,.18,"10 × 10 × 10 cm",ha="center",fontsize=6.5);c.text(.98,.18,"10 × 10 × 1.25 cm",ha="center",fontsize=6.5)
    save(fig,"geometry")


def specimens(aux):
    fits=pd.read_csv(PUBLIC_DATA/"optics/material_laws.csv").set_index(["material","band"])
    fig,axes=plt.subplots(2,2,figsize=(6.4,3.85),sharex=True)
    fig.subplots_adjust(left=.10,right=.98,top=.94,bottom=.20,hspace=.32,wspace=.28)
    angle=np.linspace(-75,75,451)
    for a,key,title in zip(axes.flat,("alum","mli","solar_cell","spback"),("Aluminum","Multilayer insulation","Solar cell","BR 127 panel backing")):
        for band,color,marker,ls in (("E5",GREY,"o","-"),("E12",ORANGE,"+","--")):
            r=fits.loc[key,band]; m=r["m"]
            k=(m+2)*(m+4)/(8*(m+2**(-m/2)))
            a.plot(angle,r.rho_d+r.rho_s*k*np.cos(np.deg2rad((angle-45)/2))**m,color=color,ls=ls,lw=1.2)
            a.plot(aux[key+"_"+band+"_angle"],aux[key+"_"+band+"_measured"],marker,ms=5.2,markeredgewidth=1.1,mfc="none",color=color)
        a.set(yscale="log",xticks=[-75,-45,0,45,75]);a.set_title(title,loc="left",weight="bold")
        a.grid(axis="y",alpha=.17)
    for ax in axes[:,0]:ax.set_ylabel("Reflectance factor")
    fig.text(.52,.108,"Signed emission angle (deg)",ha="center")
    handles=[Line2D([],[],ls="",marker=m,color=c,mfc="none",label=f"{b} measured") for b,c,m in (("E5",GREY,"o"),("E12",ORANGE,"+"))]
    handles.extend(Line2D([],[],ls=ls,color=c,label=f"{b} fit") for b,c,ls in (("E5",GREY,"-"),("E12",ORANGE,"--")))
    fig.legend(handles=handles,loc="lower center",ncol=4,bbox_to_anchor=(.52,.005),columnspacing=1.3)
    save(fig,"material_refl")


def ranges():
    tab=pd.read_csv(PUBLIC_DATA/"ranges_reference.csv").query("shape=='plate' and material=='grey' and T==4").set_index(["band","case"])
    cases=["weak_date","strong_fixed","lunar_fixed","strong_date","strong_date_lunar"]
    y=np.array([4.6,3.5,2.65,1.32,.46])
    fig,ax=plt.subplots(figsize=(6.7,2.30))
    fig.subplots_adjust(left=.37,right=.99,top=.86,bottom=.23)
    for x,l in [(-.55,"Phase angle"),(-.25,"Sky")]:
        ax.text(x,1.04,l,transform=ax.transAxes,color=GREY,fontsize=7.8)
    for yy,l in [(4.6,"75.7°"),(1.32,"34.7°")]:
        ax.text(-.55,yy,l,transform=ax.get_yaxis_transform(),va="center",weight="bold")
    for yy,l in zip(y,["Faint sky","Bright sky","+ Moon","Bright sky","+ Moon"]):
        x=-.25 if not l.startswith("+") else -.20
        ax.text(x,yy,l,transform=ax.get_yaxis_transform(),va="center",fontsize=8.1)
    for b,off,col,mk in (("E5",-.13,GREY,"s"),("E12",.13,TEAL,"o")):
        for c,yy in zip(cases,y):
            row=tab.loc[b,c];mid=row.p50
            ax.hlines(yy+off,row.p05,row.p95,color=col,lw=1)
            ax.plot(mid,yy+off,mk,color=col,ms=3.7)
            if b=="E12":ax.annotate(f"{mid:.1f}",(mid,yy+off),xytext=(4,2),textcoords="offset points",color=TEAL,fontsize=8)
    ax.legend(handles=[Line2D([],[],color=col,marker=m,label=b,lw=1,ms=4) for b,col,m in (("E5",GREY,"s"),("E12",TEAL,"o"))],loc="lower center",bbox_to_anchor=(.53,.96),ncol=2)
    ax.set(xlim=(125,505),ylim=(-.05,5.1),yticks=[],xlabel="Range at expected SNR 10 (km)")
    ax.spines["left"].set_visible(False);ax.grid(axis="x",alpha=.18)
    save(fig,"paper_ranges_r8")


def crop_sequence():
    tab=pd.read_csv(DATA/"paper_frames.csv").query("band=='E12' and frame==1").set_index("case")
    fig=plt.figure(figsize=(6.5,3.85))
    gs=fig.add_gridspec(4,2,width_ratios=[5,1.1],height_ratios=[1,1,1,1.82],left=.12,right=.98,top=.945,bottom=.13,hspace=.74,wspace=.08)
    profiles=[]
    for i,(case,label) in enumerate([("weak_date","Faint sky · 75.7° phase"),("strong_fixed","Bright sky · 75.7° phase"),("strong_date","Bright sky · 34.7° phase")]):
        with np.load(DATA/"sequences"/f"E12_{case}.npz") as z:
            a=z["calibrated_e"][0];p=z["target_e"][0].sum(axis=0,dtype=float)
        profiles.append(p)
        ax=fig.add_subplot(gs[i,0]);im=ax.imshow(a[18:47],extent=(-.5,a.shape[1]-.5,17.5,46.5),origin="lower",aspect="auto",interpolation="nearest",cmap="gray",vmin=-6,vmax=24)
        x0=32.25
        x1=x0+4*939.56668616/15.75001479
        ax.add_patch(Rectangle((x0-4,32-5),x1-x0+8,10,fill=False,color="#39e780",lw=1.0))
        ax.set_title(f"({'abc'[i]}) {label}",loc="left",fontsize=8.3,pad=1);ax.axis("off")
        note=fig.add_subplot(gs[i,1]);note.axis("off");row=tab.loc[case]
        np.testing.assert_allclose(p.sum(),row.source_e,rtol=1e-7)
        note.text(0,.60,f"SNR  {row.SNR_cal:.2f}",weight="bold");note.text(0,.16,f"{row.source_e:,.0f} e⁻")
    np.testing.assert_array_equal(profiles[0],profiles[1])
    ax=fig.add_subplot(gs[3,0])
    for p,label,col,ls in [(profiles[2],"(c)",ORANGE,"-"),(profiles[0],"(a) = (b)",TEAL,"--")]:ax.plot(p,color=col,ls=ls,label=label,lw=1.4)
    ax.set(xlim=(-.5,len(p)-.5),ylim=(0,None),xlabel="Along-track position (native pixel)",ylabel="Expected target\n(e⁻/column)")
    ax.legend(loc="upper center",ncol=2,borderaxespad=0);ax.grid(axis="y",alpha=.18)
    note=fig.add_subplot(gs[3,1]);note.axis("off")
    fig.colorbar(im,cax=note.inset_axes([.05,.66,.82,.13]),orientation="horizontal",ticks=[-6,9,24]);note.text(.45,.11,"Calibrated\ne⁻/pixel",ha="center")
    save(fig,"first_exposure_E12_r8")


def virtual_speeds():
    speed_file = DATA / "catalogue_speed_review/leo_speeds_arcsec_s.csv"
    if not speed_file.is_file():
        speed_file = HERE / "leo_speeds_arcsec_s.csv"
    speeds = np.loadtxt(speed_file, skiprows=1)
    assert speeds.size == 9396
    levels = (0, 10, 50, 90, 100)
    q = np.percentile(speeds,levels)
    fig, ax = plt.subplots(figsize=(6.25,1.78),layout="constrained")
    ax.set_xscale("log")
    ax.set(xlim=(13,34000),ylim=(-.65,.75),yticks=[],xlabel="Angular speed (arcsec/s)")
    ax.set_xticks([20,100,500,2000,10000],labels=["20","100","500","2,000","10,000"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.spines["left"].set_visible(False)
    ax.hlines(0,q[0],q[-1],color=GREY,lw=.95)
    ax.hlines(0,q[1],q[-2],color=TEAL,lw=4)
    for k,(lev,value) in enumerate(zip(levels,q)):
        ax.vlines(value,-.10,.10,color=TEAL,lw=1)
        ax.annotate(f"P{lev}  {value:,.0f}",(value,0),xytext=(0,7),
                    textcoords="offset points",ha="center",fontsize=8)
    ax.plot(q[2],0,"o",color=ORANGE,ms=5)
    upper=ax.secondary_xaxis("top",functions=(lambda v:v/15.75,lambda v:v*15.75))
    upper.set_xlabel("Virtual camera speed (pixel/s)")
    upper.set_xticks([2,10,50,200,1000],labels=["2","10","50","200","1,000"])
    upper.xaxis.set_minor_locator(plt.NullLocator())
    save(fig,"speed_anchors")


def sky_response_trade():
    data = pd.read_csv(PUBLIC_DATA / "ranges_reference.csv")
    medians = data[data["case"].isin(["weak_date", "strong_fixed"])].pivot(
        index=["shape", "material", "T"], columns=["band", "case"], values="p50"
    )
    surfaces = [
        ("grey", "Gray diffuse", "o"), ("alum", "Aluminum", "^"),
        ("mli", "MLI", "s"), ("solar_cell", "Solar cell", "D"),
        ("spback", "Panel back", "v"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.87), sharex=True, sharey=True)
    fig.subplots_adjust(left=.094, right=.986, bottom=.22, top=.82, wspace=.075)
    for ax, shape, title in zip(axes, ("plate", "cuboid"), ("(a) Plate", "(b) Cube")):
        ax.plot([0, 11], [0, 11], "--", color=GREY, lw=.8, zorder=1)
        for material, _, marker in surfaces:
            for exposure, colour in ((.5, TEAL), (4, ORANGE)):
                row = medians.loc[shape, material, exposure]
                faint = row["E12", "weak_date"]
                bright = row["E12", "strong_fixed"]
                loss = 100 * (1 - bright / faint)
                gain_faint = 100 * (faint / row["E5", "weak_date"] - 1)
                gain_bright = 100 * (bright / row["E5", "strong_fixed"] - 1)
                assert (gain_bright > loss and gain_faint > loss) if exposure == .5 else (
                    gain_bright < loss and gain_faint < loss)
                ax.hlines(loss, gain_bright, gain_faint, color=colour, lw=.85, zorder=2)
                for gain, fill in ((gain_bright, colour), (gain_faint, "white")):
                    ax.plot(gain, loss, marker=marker, ls="none", ms=3.65, mfc=fill,
                            mec=colour, mew=.85, zorder=3)
        ax.set(xlim=(0,11), ylim=(0,11), xticks=range(0,11,2), yticks=range(0,11,2))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, loc="left", weight="bold", pad=3)
        ax.grid(alpha=.14)
    axes[0].set_ylabel("Range loss from the brighter sky (%)")
    fig.text(.55, .13, "Range gain from E5 to E12 (%)", ha="center", fontsize=9)
    shape_key = [Line2D([], [], color=GREY, marker=mark, ls="none", ms=5,
                        mfc="white", label=name) for _,name,mark in surfaces]
    fig.legend(handles=shape_key, ncol=5, loc="upper center", bbox_to_anchor=(.52,1.015),
               columnspacing=1.1, handletextpad=.25, fontsize=8.5, frameon=False)
    keys = [Line2D([], [], color=colour, lw=2, label=label)
            for colour,label in ((TEAL,"0.5 s"),(ORANGE,"4 s"))]
    keys.extend(Line2D([], [], color=GREY, marker="o", ls="none", ms=5, mfc=fill, label=label)
                for fill,label in (("white","Faint sky"),(GREY,"Bright sky")))
    fig.legend(handles=keys, ncol=4, loc="lower center", bbox_to_anchor=(.53,-.008),
               columnspacing=1.55, handletextpad=.45, fontsize=8.3, frameon=False)
    save(fig, "sky_spectral_trade")


def paired_sky_ranges():
    from figure11 import calculate
    _, faint, bright, plate_loss = calculate()
    losses = 100 * (1 - bright / faint)
    lo, median, hi = np.percentile(losses, [5, 50, 95])
    fig, (ax, interval) = plt.subplots(1, 2, figsize=(6.7, 2.8),
                                       gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.10, right=.98, bottom=.20, top=.84, wspace=.72)
    ax.set_xscale("log")
    ax.scatter(faint, losses, s=14, facecolors="none", edgecolors=TEAL, linewidths=.8)
    ax.set(xlabel="Faint sky range (km)", ylabel="Bright sky range loss")
    ax.yaxis.set_major_formatter("{x:g}%")
    ax.set_title("(a) Individual streaks", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.16)
    interval.set_title("(b) Range loss", loc="left", weight="bold")
    interval.hlines(1, lo, hi, color=TEAL, lw=1.6)
    interval.vlines([lo, hi], .94, 1.06, color=TEAL, lw=1.25)
    interval.plot(median, 1, "o", color=TEAL, ms=5)
    interval.plot(plate_loss, 0, "D", color=ORANGE, mfc="none", ms=5)
    for level, value in [(1, median), (0, plate_loss)]:
        interval.text(value, level+.14, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    interval.set(yticks=[1, 0], yticklabels=["68 streaks", "Gray plate"],
                 xlim=(7.8, 10.6), ylim=(-.5, 1.5), xticks=[8, 9, 10], xlabel="Range loss (%)")
    interval.tick_params(axis="y", length=0)
    interval.spines["left"].set_visible(False)
    interval.grid(axis="x", alpha=.16)
    save(fig, "loss")


def exposure_sums():
    details=json.loads((DATA/"production_coadd_example.json").read_text(encoding="utf-8"))
    rows={r["tier"]:r for r in details["rows"]}
    fig,axes=plt.subplots(2,3,figsize=(6.1,3.8))
    fig.subplots_adjust(left=.105,right=.865,bottom=.07,top=.86,wspace=.05,hspace=.12)
    with np.load(DATA/"production_coadd_cutouts.npz") as z:
        aperture,annulus=z["aperture"],z["annulus"];weights=z["weights"]
        assert (int(aperture.sum()),int(annulus.sum()))==(299,261)
        for j,(key,title) in enumerate([("T4P0","1 × 4 s"),("T2P0","2 × 2 s"),("T0P5","8 × 0.5 s")]):
            summed=z[key+"_sum"];exp=z[key+"_expected"];r=rows[key]
            np.testing.assert_allclose([np.sum(weights*summed),np.sum(weights*exp)],[r["measured_fixed_statistic_e"],r["expected_fixed_statistic_e"]])
            for i,a in enumerate((summed-summed[annulus].mean(),exp)):
                ax=axes[i,j];im=ax.imshow(a,origin="lower",cmap="gray",interpolation="nearest",vmin=-15,vmax=200)
                ax.contour(aperture,levels=[.5],colors=["#dd9c2e"],linewidths=.55)
                ax.set(xlim=(17.5,69.5),ylim=(18.5,70.5),xticks=[],yticks=[])
                for sp in ax.spines.values():sp.set_visible(False)
            axes[0,j].set_title(title,weight="bold",pad=5)
    axes[0,0].set_ylabel("Simulated calibrated\nexposure sum",rotation=90);axes[1,0].set_ylabel("Expected target\ncharge",rotation=90)
    cb=fig.colorbar(im,cax=fig.add_axes([.90,.20,.019,.51]),ticks=[0,50,100,150,200],extend="both");cb.set_label("Electrons per pixel")
    save(fig,"production_coadd_example")


def tess(aux):
    with np.load(PUBLIC_DATA/"tess/site2_model_inputs.npz") as z:
        mask=z["stats_mask"];model=z["galaxy"].astype(float);obs=np.full(mask.shape,np.nan);obs[mask]=aux["tess_obs_in_mask"]
    units=pd.read_csv(PUBLIC_DATA/"tess/tess_paired_units.csv").query("population=='selected57'").set_index("unit").loc[aux["unit_ids"]]
    x=aux["unit_abs_beta"];ratio=units.ratio_clamp.to_numpy();median=np.median(ratio)
    rms=np.sqrt(np.mean(units.model_relative_error_clamp**2))
    np.testing.assert_allclose([median,rms],[1.09100648,.09808848],rtol=1e-7)
    fig=plt.figure(figsize=(6.2,4.75))
    gs=fig.add_gridspec(2,4,height_ratios=[1.23,1],width_ratios=[1,1,1,.065],left=.10,right=.95,bottom=.12,top=.845,wspace=.23,hspace=.70)
    mm=obs[mask].mean();gm=model[mask].mean()
    lim=float(np.percentile(abs(obs[mask]-mm),98))
    cmap=plt.get_cmap("RdBu_r").copy();cmap.set_bad("#bcc1c0")
    for i,(values,mean,title) in enumerate([(obs,mm,"Observed"),(model,gm,"Galactic model"),(obs-model,mm-gm,"Observed − model")]):
        ax=fig.add_subplot(gs[0,i]);im=ax.imshow(np.where(mask,values-mean,np.nan),cmap=cmap,origin="lower",extent=[0,3.5,0,3.5],vmin=-lim,vmax=lim,interpolation="nearest")
        ax.set_title(f"{title}\nmean {mean:.1f}",fontsize=8.2,pad=3);ax.set(xticks=[0,1,2,3],yticks=[0,1,2,3],xlabel="CCD x (deg)")
        if i==0:ax.set_ylabel("CCD y (deg)")
        else:ax.tick_params(labelleft=False)
    axc=fig.add_subplot(gs[0,3]);fig.colorbar(im,cax=axc,ticks=[-5,0,5],extend="both")
    axc.set_ylabel("Deviation (e⁻/pixel/s)",fontsize=8)
    ax=fig.add_subplot(gs[1,:3]);ax.scatter(x,ratio,facecolors="none",edgecolors=TEAL,s=20,lw=.9)
    ax.axhline(1,color=GREY,lw=.8);ax.axhline(median,color=ORANGE,lw=1.3,ls="--",label=f"Median {median:.3f}")
    ax.set(xlim=(-3,90),ylim=(.975,1.30),xticks=[0,20,40,60,80],yticks=[1.,1.1,1.2,1.3],xlabel="Absolute ecliptic latitude (deg)",ylabel="Observed / model")
    ax.set_title("(b) Total background across 57 sky positions and viewing geometries",loc="left",fontsize=8.6,weight="bold",pad=5)
    ax.legend(loc="upper right",fontsize=8);ax.grid(axis="y",alpha=.15)
    fig.text(.10,.98,"(a) Spatial variation in one TESS field",weight="bold",va="top")
    save(fig,"tess_checks")


def shares():
    tab=pd.read_csv(PUBLIC_DATA/"factors/factor_exposure_shares.csv")
    fig=plt.figure(figsize=(6.8,4.75))
    gs=fig.add_gridspec(2,2,height_ratios=[1.05,1],width_ratios=[1.1,1],left=.11,right=.975,top=.94,bottom=.11,wspace=.43,hspace=.74)
    a,b,c=fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])
    a.set_title("(a) Dominant factors",loc="left",weight="bold")
    for y,(key,label) in enumerate([("phase","Solar phase angle"),("speed","Image-plane speed"),("surface_spin","Surface and spin")]):
        lo,hi=tab[key].min(),tab[key].max();a.hlines(y,0,lo,color="#d4dcdf",lw=1.1)
        a.hlines(y,lo,hi,color=TEAL,lw=2.7);a.scatter([lo,hi],[y,y],s=12,color=TEAL,zorder=3)
        a.text(0,y+.16,f"{label}  {lo:.0f}–{hi:.0f}%",fontsize=8.3)
    a.set(yticks=[],xlim=(-1,71),ylim=(-.25,2.55),xticks=[0,20,40,60],xlabel="First-order variance share (%)")
    a.spines["left"].set_visible(False);a.grid(axis="x",alpha=.15)
    b.set_title("(b) Background and spectral response",loc="left",weight="bold",fontsize=8.7)
    for key,col in (("sky",TEAL),("response",ORANGE)):
        for shape,mark,ls in (("plate","o","-"),("cuboid","s","--")):
            vals=tab.loc[tab["shape"]==shape].sort_values("exposure_s")[key]
            b.plot([0,1],vals,ls=ls,marker=mark,color=col,mfc=col if shape=="plate" else "white",lw=1.3,ms=4.6)
    b.text(1.08,.355,"Sky",color=TEAL,va="center");b.text(1.08,.123,"Spectral\nresponse",color=ORANGE,va="center",fontsize=8)
    b.set(xlim=(-.13,1.72),ylim=(-.01,.39),xticks=[0,1],xticklabels=["0.5 s","4 s"],yticks=[0,.1,.2,.3],xlabel="Exposure time",ylabel="First-order share (%)")
    b.legend(handles=[Line2D([],[],color=GREY,marker=m,ls=l,label=word,mfc=f) for m,l,word,f in (("o","-","Plate",GREY),("s","--","Cube","white"))],fontsize=7.6,loc="upper left",handlelength=1.3)
    b.grid(axis="y",alpha=.15)
    effects=pd.read_csv(HERE/"factor_speed_phase.csv")
    c.set_title("(c) Speed and phase with interactions included",loc="left",weight="bold")
    for i,(shape,time) in enumerate((("plate",0.5),("plate",4),("cuboid",0.5),("cuboid",4))):
        grid=effects[(effects["shape"]==shape)&(effects["exposure_s"]==time)].set_index("factor")
        for key,dy,color,marker in (("speed",-.12,TEAL,"o"),("phase",.12,ORANGE,"s")):
            first,total=grid.loc[key,["first_pct","total_pct"]]
            c.plot([first,total],[i+dy,i+dy],color=color,lw=1.3)
            c.plot(first,i+dy,marker,ms=4,mec=color,mfc="white")
            c.plot(total,i+dy,marker,ms=4,mec=color,mfc=color)
    c.set(yticks=range(4),yticklabels=["Plate, 0.5 s","Plate, 4 s","Cube, 0.5 s","Cube, 4 s"],
          xlim=(0,25),ylim=(3.50,-.6),xticks=range(0,26,5),xlabel="Share of log SNR variance (%)")
    c.grid(axis="x",alpha=.15)
    c.legend(handles=[Line2D([],[],color=col,marker=m,ls="",label=word)
                      for col,m,word in ((TEAL,"o","Speed"),(ORANGE,"s","Phase"))],
             loc="lower right",bbox_to_anchor=(1,1),ncol=2,fontsize=8,columnspacing=1)
    save(fig,"factor_shares_r8")



def speed_ranges():
    """Figure 8: median threshold range for a 10 cm grey plate at four seconds."""
    grid = pd.read_csv(DATA / "controlled_10cm.csv", dtype={"spin": str})
    plate = grid[(grid["shape"] == "plate") & (grid.material == "grey")
                 & (grid["T"] == 4) & (grid.sky == "pilot")]
    states = [("fixed", "#333333", "o", "-", 7), ("1", "#624c99", "^", "-.", 4),
              ("10", TEAL, "s", ":", 4.5), ("100", "#be581f", "D", "--", 3.8)]
    ticks = np.sort(plate.omega.unique())
    fig, axes = plt.subplots(1,2,figsize=(6.8,3.6),sharey=True)
    fig.subplots_adjust(left=.10,right=.978,bottom=.29,top=.78,wspace=.27)
    for ax, band, panel in zip(axes, ("E5", "E12"), ("(a)", "(b)")):
        for state,colour,marker,line,size in states:
            rows=plate[(plate.band==band)&(plate.spin==state)].sort_values("omega")
            ax.plot(rows.omega,rows.p50,color=colour,marker=marker,ls=line,
                    ms=size,mfc="none" if state=="fixed" else colour,lw=1.2,
                    label="Fixed attitude" if state=="fixed" else f"{state} s")
        ax.set_xscale("asinh",linear_width=120)
        ax.set(xlim=(-28,5200),xlabel="Angular speed (arcsec/s)")
        ax.set_xticks(ticks,labels=[f"{v:,.0f}" for v in ticks])
        ax.tick_params(labelsize=7.6)
        ax.grid(axis="y",alpha=.21)
        ax.tick_params(axis="y", labelleft=True)
        ax.text(.97,.95,f"{panel} {band}",transform=ax.transAxes,ha="right",va="top",weight="bold")
        top=ax.twiny()
        top.set_xscale("asinh",linear_width=120)
        top.set_xlim(ax.get_xlim())
        top.set_xticks(ticks,labels=[f"{v/15.75:.1f}".rstrip("0").rstrip(".") for v in ticks])
        top.set_xlabel("Virtual camera speed (pixel/s)")
        top.tick_params(labelsize=7.5)
        top.spines["top"].set_visible(True)
        top.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())
    axes[0].set_ylabel("Median SNR 10 range (km)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=4,
               bbox_to_anchor=(.53,.02),fontsize=8)
    save(fig,"speed")
