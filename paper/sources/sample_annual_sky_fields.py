"""Evaluate the two detector footprints in supplementary Fig. S12 using the scientific providers.

The published NPZ supplies the two pointing declarations as well as the sampled outputs.
To reevaluate rather than redraw, supply the AI_SBSSA_v2 checkout and its external sky
assets at the recorded source revision. The optional --paper-skies reads the original
annual-selection product in place of the compact declarations embedded in the NPZ.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import os
import subprocess
import sys

import numpy as np


def bind_providers(repo):
    sys.path.insert(0, str(repo))
    from governance.regression.runner import _execution_environment

    env, _ = _execution_environment(repo, "HEAD", "WI-SYNTH-SCENE-FRAME-REFERENCE", repo)
    os.environ.update(env)
    unit = json.loads((repo/"wi/WI-SYNTH-SCENE-FRAME-REFERENCE/unit.yaml").read_text())
    for entry in unit["regression"]["import_roots"]:
        sys.path.insert(0, str(repo/"wi"/entry["unit"]/entry["path"]))
    from astropy.utils import iers
    from observation_bridge import sky_provider as providers
    from physics_observation_model import sky_background
    iers.conf.auto_download = False
    src = providers.load_sky_sources(
        repo, camera_id="cmv97_f2_e12", sky_declaration_path=providers.SKY_DECLARATION_E12_V3_PATH,
        zodiacal_source=providers.ZODIACAL_SOURCE_Z1)
    return sky_background, src


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def evaluate(row, focal_px, plane_coord, sky, src):
    z = normalize(np.array(row["ray_gcrs"]))
    r = normalize(np.array(row["observer_gcrs_km"]))
    # The fixed el45/az300 mount rolls camera +y toward the local outward radial.
    y = normalize(r - (r @ z)*z)
    x = np.cross(y, z)
    u, v = np.meshgrid(plane_coord, plane_coord)
    rays = normalize(z + u.ravel()[:, None]*x + v.ravel()[:, None]*y)
    omega = sky.pixel_solid_angle_sr(u.ravel()*focal_px, v.ravel()*focal_px, focal_px, 0, 0)
    epoch = datetime.fromisoformat(row["epoch_utc"])
    obs = np.asarray(row["observer_gcrs_km"])
    geom = sky.sky_geometry(rays, observer_gcrs_km=obs, epoch=epoch)
    maps = {k: np.empty(len(rays)) for k in ["zodiacal", "isotropic", "dgl", "unresolved", "dust"]}
    ones = np.ones(1, dtype=bool)
    iso_at_axis = sky.fold_radiance(
        src.wave_nm, src.zodiacal_provider.isotropic_residual_radiance_w_m2_sr_nm[None],
        response=src.response, area_m2=src.area_m2, omega_sr=np.array([1/focal_px**2]), rows=ones)[0]
    for begin in range(0,len(rays),192):
        sl = slice(begin, min(begin+192,len(rays)))
        # sky_geometry also stores 3-D solar/lunar vectors, which are shared by all rays.
        local = {k: v[sl] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(rays) else v
                 for k,v in geom.items()}
        brightness = sky.sky_radiance(
            local, wave_nm=src.wave_nm, solar_sed_w_m2_nm=src.solar_sed_w_m2_nm, m_sun_v=src.m_sun_v,
            zodiacal_table=src.zodiacal_table, zodiacal_colour_factor=src.zodiacal_colour_factor,
            dgl=src.dgl, unresolved=src.unresolved, zodiacal_provider=src.zodiacal_provider)
        comp = brightness["components"]
        for output, name in [("zodiacal","zodiacal"),("dgl","dgl"),("unresolved","unresolved_stars")]:
            c = comp[name]
            ok = np.isfinite(c["radiance"]).all(axis=1)
            if name=="dgl":
                ok &= sky.dgl_rows_supported(c["status"])
            if not ok.all():
                raise ValueError(f"Unsupported {name} in the selected footprint")
            maps[output][sl] = sky.fold_radiance(src.wave_nm,c["radiance"],response=src.response,
                                                area_m2=src.area_m2, omega_sr=omega[sl], rows=ok)
        maps["dust"][sl] = comp["dgl"]["i100_MJy_sr"]
        maps["isotropic"][sl] = iso_at_axis * omega[sl] * focal_px**2
    maps["zodiacal"] -= maps["isotropic"]
    maps["galactic_latitude"] = geom["gal_b_deg"]
    shape = u.shape
    maps = {name:values.reshape(shape) for name,values in maps.items()}
    mid = len(plane_coord)//2
    refs = row["components_scmos_e_px_s"]["E12"]
    for key,original in [("zodiacal","zodiacal_Z0"),("isotropic","isotropic"),("dgl","dgl"),
                         ("unresolved","unresolved_stars")]:
        np.testing.assert_allclose(maps[key][mid,mid],refs[original],rtol=1e-8)
    np.testing.assert_allclose(maps["dust"][mid,mid],row["I100_MJy_sr"],rtol=1e-8)
    return maps, np.stack([x,y,z])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--paper-skies",type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("annual_sky_fields.npz"))
    args=parser.parse_args()
    repo=args.repo.resolve()
    if args.paper_skies:
        raw=json.loads(args.paper_skies.read_text(encoding="utf-8"))
        names=["hold_index","epoch_utc","ray_gcrs","observer_gcrs_km","I100_MJy_sr","components_scmos_e_px_s"]
        binding={"pixel_model":raw["pixel_model"], "cases":{
            k:{n:raw["cases"][k][n] for n in names} for k in ["weak","strong"]}}
    else:
        with np.load(args.data,allow_pickle=False) as saved:
            binding=json.loads(saved["declaration"].item())
    sky,src=bind_providers(repo)
    focal=binding["pixel_model"]["virtual_sCMOS_focal_length_m"]/binding["pixel_model"]["virtual_sCMOS_pitch_m"]
    half_extent=3428/(2*focal)
    n=129
    q=(np.arange(n)-(n-1)/2)*(2*half_extent/n)
    result={"declaration":json.dumps(binding), "native_width":3428,"plane_coordinate":q,
            "model_revision":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()}
    for tag, label in [("weak","faint"),("strong","bright")]:
        case=binding["cases"][tag]
        maps,axes=evaluate(case,focal,q,sky,src)
        result.update({label+"_"+k:v.astype("f4") for k,v in maps.items()})
        result[label+"_rotation"]=axes
        total=sum(maps[k] for k in ["zodiacal","isotropic","dgl","unresolved"])
        print(label, "center/mean/range", total[n//2,n//2],total.mean(),total.min(),total.max(),
              "component means", [float(maps[k].mean()) for k in ["zodiacal","isotropic","dgl","unresolved"]],
              "DGL capped fraction",np.mean(maps["dust"]>25), flush=True)
    result["galactic_center_gcrs"]=sky.galactic_rotation()[0]
    np.savez_compressed(args.data,**result)


if __name__=="__main__":
    main()
