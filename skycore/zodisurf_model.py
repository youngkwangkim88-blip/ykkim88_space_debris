"""ZodiSURF (SKYSURF-11) alone: the declared NAMED ALTERNATIVE zodiacal provider for `sky_background.background_map`."""

from __future__ import annotations

import hashlib

import json

import math

from pathlib import Path

import numpy as np

from . import radiometry as SR

from . import sky_background as SKY

AU_KM = 149597870.7

MODEL_ID = "ZODISURF_SKYSURF11_KELSALL1998_GEOMETRY"

NAMED_ALTERNATIVE = "ZODISURF_ALONE"

COLOUR_LAW = "ZODISURF_ALBEDO_AND_PHASE_FUNCTION_PER_NODE_ABSOLUTE_NO_V_ANCHOR"

PARAMETERS_PATH = Path(__file__).resolve().parents[1] / "data" / "sky" / "zodisurf_model_v1.json"

KELSALL_COMPONENTS = ("cloud", "band1", "band2", "band3", "ring", "feature")

STATUS_ZL_OK = SKY.STATUS_ZL_OK

STATUS_ZL_ZODISURF_SUN_ANGLE_OUTSIDE_DOMAIN = "ZL_ZODISURF_SUN_ANGLE_OUTSIDE_DOMAIN"

REFUSAL_WAVELENGTH_OUTSIDE_DOMAIN = "ZL_ZODISURF_WAVELENGTH_OUTSIDE_DOMAIN"

APPROXIMATIONS = (
    "ZODISURF_ALONE_NAMED_ALTERNATIVE_NOT_THE_DEFAULT",
    "ZODISURF_PAPER_PRINTED_COEFFICIENTS_ARXIV_2510_18231V2_EQS_15_21",
    "KELSALL1998_GEOMETRY_DIRBE_CODE_FORMS",
    "NO_ISOTROPIC_COMPONENT_EXCLUDED_BY_CONSTRUCTION",
    "OBSERVER_HELIOCENTRIC_FROM_APPARENT_GEOCENTRIC_SUN_ABERRATION_20_ARCSEC",
    "SOLAR_SPECTRUM_CALLER_TSIS1_1NM",
    "SUN_ANGLE_AT_OR_BELOW_80_DEG_REFUSED_NO_SPLICE_NO_EXTRAPOLATION",
)

def load_parameters(path=None, *, sha256=None):
    path = Path(path) if path is not None else PARAMETERS_PATH
    raw = path.read_bytes()
    digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if sha256 is not None and digest != sha256:
        raise SKY.SkyBackgroundError(f"{path.name} has sha256 {digest}, the caller pins {sha256}", key="zodisurf_parameters")
    p = json.loads(raw.decode("utf-8"))
    p["_sha256"] = digest
    return p

def _wavelengths_in_domain(lam_um, p):
    lam = np.atleast_1d(np.asarray(lam_um, dtype=np.float64))
    lo, hi = p["domain"]["wavelength_um"]
    if np.any((lam < lo - 1e-12) | (lam > hi + 1e-12)):
        raise SKY.SkyBackgroundError(f"ZodiSURF is defined on {lo:g}-{hi:g} um; {lam.min():g}-{lam.max():g} um requested",
                                     key=REFUSAL_WAVELENGTH_OUTSIDE_DOMAIN)
    return lam

def albedo(lam_um, p):
    """Eq. 15: A = slope lambda + intercept (lambda in um), the paper's printed coefficients."""
    a = p["scattering"]["albedo"]
    return a["slope_per_um"] * _wavelengths_in_domain(lam_um, p) + a["intercept"]

def phase_parameters(lam_um, p):
    """(g, w), each (3, L): the forward, backward and gegenschein terms, linear in wavelength (Eqs. 16-21)."""
    s = p["scattering"]
    lam = _wavelengths_in_domain(lam_um, p)
    g = np.stack([t["slope_per_um"] * lam + t["intercept"] for t in s["g"]])
    w = np.stack([t["slope_per_um"] * lam + t["intercept"] for t in s["w"]])
    return g, w

def hg_phase(cos_theta, g, w):
    """Eq. 4: sum_i w_i (1 - g_i^2) / (1 + g_i^2 - 2 g_i cos Theta)^(3/2); `cos_theta` (...,), g and w (3, L) -> (..., L)."""
    c = np.asarray(cos_theta, dtype=np.float64)[..., None]
    out = np.zeros(c.shape[:-1] + (g.shape[1],))
    for gi, wi in zip(g, w):
        out += wi * (1.0 - gi ** 2) / (1.0 + gi ** 2 - 2.0 * gi * c) ** 1.5
    return out

def kelsall_phase(theta, c0, c1, c2):
    """Kelsall et al. (1998) phase function N (C0 + C1 Theta + exp(C2 Theta)) with its analytic 4 pi normalisation."""
    norm = 1.0 / (2.0 * math.pi * (2.0 * c0 + math.pi * c1 + (math.exp(c2 * math.pi) + 1.0) / (c2 ** 2 + 1.0)))
    return norm * (c0 + c1 * theta + np.exp(c2 * theta))

def _height(x, i_deg, omega_deg):
    i, om = math.radians(i_deg), math.radians(omega_deg)
    normal = np.array([math.sin(om) * math.sin(i), -math.cos(om) * math.sin(i), math.cos(i)])
    return x @ normal

def density(name, x, x_earth, p):
    """Number density of one Kelsall component at heliocentric positions `x` (..., 3); `x_earth` (..., 3) for the feature."""
    g = p["geometry"]
    if name == "cloud":
        c = g["cloud"]
        xc = x - np.array([c["x0"], c["y0"], c["z0"]])
        r = np.linalg.norm(xc, axis=-1)
        zeta = np.abs(_height(xc, c["i_deg"], c["Omega_deg"]) / r)
        gg = np.where(zeta < c["mu"], zeta ** 2 / (2.0 * c["mu"]), zeta - c["mu"] / 2.0)
        return c["n0_per_au"] * r ** -c["alpha"] * np.exp(-c["beta"] * gg ** c["gamma"])
    if name.startswith("band"):
        c = g[name]
        r = np.linalg.norm(x, axis=-1)
        zeta = np.abs(_height(x, c["i_deg"], c["Omega_deg"]) / r)
        q = zeta / math.radians(c["delta_zeta_deg"])
        return (3.0 * c["n0_per_au"] / r) * np.exp(-q ** 6) * (1.0 + q ** c["p"] / c["v"]) * (1.0 - np.exp(-(r / c["delta_r_au"]) ** 20))
    if name == "ring":
        c = g["ring"]
        r = np.linalg.norm(x, axis=-1)
        z = _height(x, c["i_deg"], c["Omega_deg"])
        return c["n0_per_au"] * np.exp(-((r - c["R_au"]) ** 2) / c["sigma_r_au"] ** 2 - np.abs(z) / c["sigma_z_au"])
    if name == "feature":
        c = g["feature"]
        r = np.linalg.norm(x, axis=-1)
        z = _height(x, c["i_deg"], c["Omega_deg"])
        xe = np.asarray(x_earth, dtype=np.float64)
        dtheta = np.arctan2(x[..., 1], x[..., 0]) - np.arctan2(xe[..., 1], xe[..., 0]) - math.radians(c["theta_deg"])
        dtheta = (dtheta + math.pi) % (2.0 * math.pi) - math.pi
        return c["n0_per_au"] * np.exp(-((r - c["R_au"]) ** 2) / c["sigma_r_au"] ** 2 - np.abs(z) / c["sigma_z_au"]
                                       - dtheta ** 2 / math.radians(c["sigma_theta_deg"]) ** 2)
    raise SKY.SkyBackgroundError(f"unknown zodiacal component {name!r}", key="zodisurf_component")

def sphere_distance(x_obs, u, radius):
    """Distance along unit `u` from `x_obs` (inside the sphere) to the heliocentric sphere of `radius` (outgoing root)."""
    xu = np.sum(x_obs * u, axis=-1)
    c = np.sum(x_obs * x_obs, axis=-1) - radius ** 2
    return -xu + np.sqrt(np.maximum(xu ** 2 - c, 0.0))

def line_of_sight_kernel(u_ecl, x_obs, x_earth, lam_um, p, *, n_quad=None, mode="zodisurf", components=KELSALL_COMPONENTS,
                         chunk=2048):
    """K per ray and wavelength [sr^-1], and per component, for heliocentric ecliptic J2000 inputs in AU.

    `u_ecl` (N, 3) unit directions; `x_obs`, `x_earth` (3,) or (N, 3). `mode="zodisurf"` uses Eqs. 4 and 15-21 at the
    wavelengths `lam_um` (L,), which must lie in the model's 0.3-1.6 um; `mode="kelsall_1p25um"` uses Kelsall's own 1.25 um
    albedo and phase function (the shared-geometry oracle; `lam_um` is ignored and L = 1). Returns (K (N, L), per_component
    (C, N, L)).
    """
    u = np.atleast_2d(np.asarray(u_ecl, dtype=np.float64))
    n = u.shape[0]
    xo = np.broadcast_to(np.asarray(x_obs, dtype=np.float64), (n, 3))
    xe = np.broadcast_to(np.asarray(x_earth, dtype=np.float64), (n, 3))
    nq = int(n_quad or p["line_of_sight"]["n_quad"])
    t, wq = np.polynomial.legendre.leggauss(nq)
    if mode == "zodisurf":
        lam = _wavelengths_in_domain(lam_um, p)
        g, w = phase_parameters(lam, p)
        a = albedo(lam, p)
    elif mode == "kelsall_1p25um":
        k = p["kelsall_1p25um_reference_mode"]
        lam = np.array([1.25])
        a = np.array([k["albedo"]])
    else:
        raise SKY.SkyBackgroundError(f"unknown mode {mode!r}", key="zodisurf_mode")
    shells = p["line_of_sight"]["shells_au"]
    r_obs = np.linalg.norm(xo, axis=-1)
    per = np.zeros((len(components), n, lam.size))
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        uu, oo, ee, ro = u[lo:hi], xo[lo:hi], xe[lo:hi], r_obs[lo:hi]
        for ci, name in enumerate(components):
            r_in, r_out = shells[name]
            if np.any(ro >= r_out):
                raise SKY.SkyBackgroundError(f"the observer lies outside the {name} shell", key="zodisurf_observer")
            start = np.where(ro >= r_in, 0.0, sphere_distance(oo, uu, r_in))
            stop = sphere_distance(oo, uu, r_out)
            half = 0.5 * (stop - start)
            s = half[:, None] * (t[None, :] + 1.0) + start[:, None]                    # (m, nq)
            x = oo[:, None, :] + s[..., None] * uu[:, None, :]                          # (m, nq, 3)
            r = np.linalg.norm(x, axis=-1)
            cos_theta = np.clip(-np.sum(x * uu[:, None, :], axis=-1) / r, -1.0, 1.0)
            geo = density(name, x, ee[:, None, :], p) / r ** 2 * (half[:, None] * wq[None, :])   # (m, nq)
            if mode == "zodisurf":
                per[ci, lo:hi, :] = a[None, :] * np.einsum("mq,mql->ml", geo, hg_phase(cos_theta, g, w))
            else:
                k = p["kelsall_1p25um_reference_mode"]
                phi = kelsall_phase(np.arccos(cos_theta), k["phase_C0"], k["phase_C1"], k["phase_C2"])
                per[ci, lo:hi, 0] = a[0] * np.sum(geo * phi, axis=-1)
    return per.sum(axis=0), per

def heliocentric_ecliptic_state(geometry):
    """Observer and Earth heliocentric positions (AU) and the rays, all in the mean ecliptic J2000 frame.

    The Sun's observer-relative vector (astropy's apparent GCRS Sun, the capsule's ephemeris) reversed is the observer's
    heliocentric position; subtracting the observer's GCRS offset gives the Earth's. The Sun's apparent direction
    carries annual aberration (about 20 arcsec, 1e-4 AU at 1 AU), declared as an approximation token.
    """
    rot = SKY.ecliptic_rotation_j2000()
    sun_vec = np.asarray(geometry["sun_direction_gcrs"], dtype=np.float64) * float(geometry["sun_distance_km"])
    obs_gcrs = np.asarray(geometry["observer_gcrs_km"], dtype=np.float64)
    x_obs = rot @ (-sun_vec) / AU_KM
    x_earth = rot @ (-sun_vec - obs_gcrs) / AU_KM
    u = np.asarray(geometry["rays_gcrs"], dtype=np.float64) @ rot.T
    return x_obs, x_earth, u

def wavelength_nodes_um(wave_nm, live, step_um):
    lo = math.floor(float(np.min(wave_nm[live])) / 1000.0 / step_um + 1e-9) * step_um
    hi = math.ceil(float(np.max(wave_nm[live])) / 1000.0 / step_um - 1e-9) * step_um
    return np.round(np.arange(lo, hi + 0.5 * step_um, step_um), 6)

def interpolation_matrix(nodes_um, wave_nm):
    """(L_nodes, W) weights of piecewise-linear interpolation from the nodes to `wave_nm` (held at the ends)."""
    x = np.clip(np.asarray(wave_nm, dtype=np.float64) / 1000.0, nodes_um[0], nodes_um[-1])
    j = np.clip(np.searchsorted(nodes_um, x, side="right") - 1, 0, nodes_um.size - 2)
    f = (x - nodes_um[j]) / (nodes_um[j + 1] - nodes_um[j])
    m = np.zeros((nodes_um.size, x.size))
    m[j, np.arange(x.size)] = 1.0 - f
    m[j + 1, np.arange(x.size)] += f
    return m

def _kernel_provider(wave_nm, solar_sed_w_m2_nm, p, *, sun_min, n_quad, lambda_node_step_um):
    """The ZodiSURF kernel as a provider refusing every node at or inside `sun_min` degrees of solar elongation: the body of
    `zodisurf_provider`, which passes the parameter file's checked 80-degree limit. The sum Z1 passes its own declared limit
    (`contract/sky/zodisurf_plus_isotropic_residual_v1.json` domain.sun_angle_min_deg), so that limit has one home."""
    wave = np.asarray(wave_nm, dtype=np.float64)
    sed = np.asarray(solar_sed_w_m2_nm, dtype=np.float64)
    live = sed > 0.0
    lo_um, hi_um = p["domain"]["wavelength_um"]
    if np.any(live & ((wave < lo_um * 1000.0 - 1e-9) | (wave > hi_um * 1000.0 + 1e-9))):
        raise SKY.SkyBackgroundError("the solar spectrum has weight outside ZodiSURF's 0.3-1.6 um domain",
                                     key=REFUSAL_WAVELENGTH_OUTSIDE_DOMAIN)
    nodes = wavelength_nodes_um(wave, live, lambda_node_step_um)
    nodes = nodes[(nodes >= lo_um - 1e-9) & (nodes <= hi_um + 1e-9)]
    interp = interpolation_matrix(nodes, wave)
    interp550 = interpolation_matrix(nodes, np.array([550.0]))
    tokens = list(APPROXIMATIONS) + [f"GAUSS_LEGENDRE_{int(n_quad or p['line_of_sight']['n_quad'])}_PER_COMPONENT",
                                     f"KERNEL_LINEAR_IN_WAVELENGTH_BETWEEN_{lambda_node_step_um:g}UM_NODES".replace(".", "P")]

    def provider(geometry):
        x_obs, x_earth, u = heliocentric_ecliptic_state(geometry)
        sun_dir = np.asarray(geometry["sun_direction_gcrs"], dtype=np.float64)
        elong = np.degrees(np.arccos(np.clip(np.asarray(geometry["rays_gcrs"]) @ sun_dir, -1.0, 1.0)))
        ok = elong > sun_min
        n = elong.size
        radiance = np.full((n, wave.size), math.nan)
        k550 = np.full(n, math.nan)
        if ok.any():
            k_nodes, _ = line_of_sight_kernel(u[ok], x_obs, x_earth, nodes, p, n_quad=n_quad)
            radiance[ok] = (k_nodes @ interp) * sed[None, :]
            k550[ok] = (k_nodes @ interp550)[:, 0]
        status = np.where(ok, STATUS_ZL_OK, STATUS_ZL_ZODISURF_SUN_ANGLE_OUTSIDE_DOMAIN).astype(object)
        return {"radiance": radiance, "status": status, "model": MODEL_ID, "named_alternative": NAMED_ALTERNATIVE,
                "colour_law": COLOUR_LAW, "elongation_deg": elong, "kernel_550_sr": k550,
                "observer_heliocentric_ecliptic_au": x_obs, "earth_heliocentric_ecliptic_au": x_earth,
                "wavelength_nodes_um": nodes, "sun_angle_min_deg": sun_min, "parameters_sha256": p.get("_sha256"),
                "approximation": list(tokens)}

    provider.wavelength_nodes_um = nodes
    provider.parameters = p
    return provider

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sky"

RESIDUAL_DECLARATION_PATH = DATA_DIR / "zodisurf_plus_isotropic_residual_v1.json"

NAMED_ALTERNATIVE_PLUS_ISOTROPIC_RESIDUAL = "ZODISURF_PLUS_ISOTROPIC_RESIDUAL"

MJY_SR_IN_W_M2_HZ_SR = 1e-20

SPEED_OF_LIGHT_M_S = 299792458.0

ALONE_ONLY_APPROXIMATIONS = ("ZODISURF_ALONE_NAMED_ALTERNATIVE_NOT_THE_DEFAULT", "NO_ISOTROPIC_COMPONENT_EXCLUDED_BY_CONSTRUCTION")

ALONE_SUN_ANGLE_TOKEN = "SUN_ANGLE_AT_OR_BELOW_80_DEG_REFUSED_NO_SPLICE_NO_EXTRAPOLATION"

SUN_ANGLE_TOKEN_FORMAT = "SUN_ANGLE_AT_OR_BELOW_{:g}_DEG_REFUSED_NO_SPLICE_NO_EXTRAPOLATION"

RESIDUAL_COLUMN = "diffuse_light_MJy_sr"

RESIDUAL_TABLE_COLUMNS = ("camera", "filter", "pivot_um", "sigma_stat_MJy_sr", "sigma_sys_MJy_sr", "diffuse_light_MJy_sr",
                          "sigma_DL_MJy_sr")

def _lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()

def load_isotropic_residual(path=None, *, declaration_sha256=None):
    """The Z1 declaration and the Table 4 residual it pins, each checked: the declaration's own digest
    (`radiometry.declared_digest`, and the caller's pin when given), its named alternative, and the table's sha256 and row
    count. Returns {declaration, rows, pivots_um (19,), i_nu_MJy_sr (19,), table_sha256, declaration_sha256}: the node
    values are the arithmetic means of column 6 over the rows that share a printed pivot, in the table's row order."""
    path = Path(path) if path is not None else RESIDUAL_DECLARATION_PATH
    decl = json.loads(path.read_text(encoding="utf-8"))
    digest = SR.declared_digest(decl)
    if decl.get("declaration_sha256") != digest:
        raise SKY.SkyBackgroundError(f"{path.name} records declaration_sha256 {decl.get('declaration_sha256')}, its declared "
                                     f"block hashes to {digest}", key="zodisurf_residual_declaration")
    if declaration_sha256 is not None and digest != declaration_sha256:
        raise SKY.SkyBackgroundError(f"{path.name} has declaration_sha256 {digest}, the caller pins {declaration_sha256}",
                                     key="zodisurf_residual_declaration")
    if decl.get("named_alternative") != NAMED_ALTERNATIVE_PLUS_ISOTROPIC_RESIDUAL:
        raise SKY.SkyBackgroundError(f"{path.name} declares {decl.get('named_alternative')!r}", key="zodisurf_residual_declaration")
    iso = decl["isotropic_residual"]
    table_path = path.with_name(Path(iso["table"]["path"]).name)
    table_sha = _lf_sha256(table_path)
    if table_sha != iso["table"]["sha256"]:
        raise SKY.SkyBackgroundError(f"{table_path.name} has sha256 {table_sha}, the declaration pins {iso['table']['sha256']}",
                                     key="zodisurf_residual_table")
    lines = [ln for ln in table_path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    header = tuple(c.strip() for c in lines[0].split(","))
    if header != RESIDUAL_TABLE_COLUMNS:
        raise SKY.SkyBackgroundError(f"{table_path.name} has columns {header}", key="zodisurf_residual_table")
    rows = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split(",")]
        rows.append({k: (v if k in ("camera", "filter") else float(v)) for k, v in zip(header, cells)})
    if len(rows) != int(iso["table"]["rows"]):
        raise SKY.SkyBackgroundError(f"{table_path.name} has {len(rows)} rows, the declaration says {iso['table']['rows']}",
                                     key="zodisurf_residual_table")
    pivots = sorted({r["pivot_um"] for r in rows})
    values = [float(np.mean([r[RESIDUAL_COLUMN] for r in rows if r["pivot_um"] == p])) for p in pivots]
    return {"declaration": decl, "rows": rows, "pivots_um": np.array(pivots), "i_nu_MJy_sr": np.array(values),
            "table_sha256": table_sha, "declaration_sha256": digest}

def isotropic_residual_sun_angle_min_deg(residual):
    """The sum's refusal limit in degrees of solar elongation: the declaration's domain.sun_angle_min_deg
    (`RESIDUAL_SUN_ANGLE_MIN_POINTER`), read from nowhere else. A node at or inside it is refused."""
    value = float(residual["declaration"]["domain"]["sun_angle_min_deg"])
    if not 0.0 < value < 180.0:
        raise SKY.SkyBackgroundError(f"the declared elongation limit {value:g} deg is not inside (0, 180)", key="zodisurf_residual_domain")
    return value

def _residual_wavelengths_in_domain(wave_nm, residual):
    w = np.atleast_1d(np.asarray(wave_nm, dtype=np.float64))
    lo, hi = residual["declaration"]["domain"]["wavelength_um"]
    if np.any((w < lo * 1000.0 - 1e-9) | (w > hi * 1000.0 + 1e-9)):
        raise SKY.SkyBackgroundError(f"the isotropic residual is defined on {lo:g}-{hi:g} um; {w.min():g}-{w.max():g} nm requested",
                                     key=REFUSAL_WAVELENGTH_OUTSIDE_DOMAIN)
    return w

def isotropic_residual_i_nu_mjy_sr(wave_nm, residual):
    """J(lambda) in MJy/sr: linear in wavelength between the pivot nodes, held at the end values outside them; a wavelength
    outside the declared 0.3-1.6 um is refused, never extrapolated."""
    w = _residual_wavelengths_in_domain(wave_nm, residual)
    return np.interp(w / 1000.0, residual["pivots_um"], residual["i_nu_MJy_sr"])      # np.interp holds the end values

def isotropic_residual_radiance(wave_nm, residual):
    """I_iso(lambda) = J(lambda) 1e-20 c / lambda_m^2 1e-9 in W m^-2 sr^-1 nm^-1, per wavelength (the conversion after J
    is interpolated)."""
    w = _residual_wavelengths_in_domain(wave_nm, residual)
    lam_m = w * 1e-9
    return isotropic_residual_i_nu_mjy_sr(w, residual) * MJY_SR_IN_W_M2_HZ_SR * SPEED_OF_LIGHT_M_S / lam_m ** 2 * 1e-9

def zodisurf_plus_isotropic_residual_provider(wave_nm, solar_sed_w_m2_nm, *, parameters=None, residual=None, n_quad=None,
                                              lambda_node_step_um=0.05):
    """The named alternative `ZODISURF_PLUS_ISOTROPIC_RESIDUAL` as a `background_map(zodiacal_provider=...)` callable.

    It builds the ZodiSURF kernel provider (`zodisurf_provider`'s body, unchanged) on the parameter file the declaration pins,
    refusing at or inside the declaration's own elongation limit (`isotropic_residual_sun_angle_min_deg`, the one home of
    that limit; 30 deg since the HUMAN's direction of 2026-10-15, below ZODISURF_ALONE's published 80), and adds the isotropic residual
    radiance once, at the nodes the kernel supports: refused nodes keep the refusal status and NaN, and never receive the
    residual alone. A base output that is not the kernel's ZODISURF_ALONE output is refused, so the residual is never added
    twice. Every grid wavelength must lie in 0.3-1.6 um (the residual adds light at every wavelength). There is no argument
    for a scale, an anchor, a colour, another residual table or another elongation limit.
    """
    res = residual if residual is not None else load_isotropic_residual()
    decl = res["declaration"]
    pin = decl["kernel"]["parameters"]
    p = parameters if parameters is not None else load_parameters(DATA_DIR / Path(pin["path"]).name, sha256=pin["sha256"])
    if p.get("_sha256") != pin["sha256"]:
        raise SKY.SkyBackgroundError(f"the isotropic residual is declared on the ZodiSURF parameter file {pin['sha256']}; "
                                     f"parameters {p.get('_sha256')} are refused", key="zodisurf_residual_kernel")
    if list(decl["domain"]["wavelength_um"]) != list(p["domain"]["wavelength_um"]):
        raise SKY.SkyBackgroundError("the residual declaration's wavelength domain differs from the kernel's", key="zodisurf_residual_kernel")
    sun_min = isotropic_residual_sun_angle_min_deg(res)
    wave = _residual_wavelengths_in_domain(wave_nm, res)
    sed = np.asarray(solar_sed_w_m2_nm, dtype=np.float64)
    alone = _kernel_provider(wave, sed, p, sun_min=sun_min, n_quad=n_quad, lambda_node_step_um=lambda_node_step_um)
    sun_token = SUN_ANGLE_TOKEN_FORMAT.format(sun_min)
    iso = isotropic_residual_radiance(wave, res)
    sed550 = float(np.interp(550.0, wave, sed))
    if not sed550 > 0.0:
        raise SKY.SkyBackgroundError("the solar spectrum has no weight at 550 nm, where the V-equivalent is read",
                                     key="zodisurf_residual_solar_550")
    k550_iso = float(np.interp(550.0, wave, iso)) / sed550
    identity = {"isotropic_residual_declaration_sha256": res["declaration_sha256"], "isotropic_residual_table_sha256": res["table_sha256"]}

    def provider(geometry):
        z = alone(geometry)
        if z.get("named_alternative") != NAMED_ALTERNATIVE or "isotropic_residual_radiance_w_m2_sr_nm" in z:
            raise SKY.SkyBackgroundError("the isotropic residual is added to a ZODISURF_ALONE radiance only, and once",
                                         key="zodisurf_residual_once")
        ok = np.asarray(z["status"], dtype=object) == STATUS_ZL_OK
        radiance = np.array(z["radiance"], dtype=np.float64, copy=True)
        radiance[ok] = radiance[ok] + iso[None, :]
        k550_alone = np.asarray(z["kernel_550_sr"], dtype=np.float64)
        tokens = ([sun_token if t == ALONE_SUN_ANGLE_TOKEN else t for t in z["approximation"] if t not in ALONE_ONLY_APPROXIMATIONS]
                  + list(decl["approximations"]))
        out = dict(z)
        out.update({"radiance": radiance, "model": decl["model_id"], "named_alternative": NAMED_ALTERNATIVE_PLUS_ISOTROPIC_RESIDUAL,
                    "colour_law": decl["colour_law"], "kernel_550_sr": np.where(ok, k550_alone + k550_iso, math.nan),
                    "zodisurf_alone_kernel_550_sr": k550_alone, "isotropic_residual_radiance_w_m2_sr_nm": iso,
                    "approximation": tokens, **identity})
        return out

    provider.wavelength_nodes_um = alone.wavelength_nodes_um
    provider.parameters = p
    provider.residual = res
    provider.isotropic_residual_radiance_w_m2_sr_nm = iso
    return provider
