"""The Moon's scattered light around its masked disk, for `synth-sky-background/v1`."""

from __future__ import annotations

import math

import numpy as np

OMEGA_MOON_REF_SR = 6.4177e-5

MOON_REF_DISTANCE_KM = 384400.0

MOON_RADIUS_KM = 1737.4

EARTH_RADIUS_KM = 6378.137

SUN_RADIUS_KM = 695700.0

AU_KM = 149597870.7

KS1991_PHASE_LINEAR_MAG_PER_DEG = 0.026

KS1991_PHASE_QUARTIC_MAG_PER_DEG4 = 4.0e-9

REFLECTANCE_PHASE_ANGLE_DEG = 7.0

DEFAULT_OPPOSITION_SURGE = 1.0

GRADE = "UNMEASURED_INSTRUMENT_SCENARIO"

SOURCE_ROUTE_ROLO311G = "ROLO311G_KIEFFER_STONE2005_EQ10"

SOURCE_ROUTE_A7_GREY = "A7_LIME2024_FIG1_TIMES_KS1991_GREY_PHASE"

SOURCE_ROUTES = (SOURCE_ROUTE_ROLO311G, SOURCE_ROUTE_A7_GREY)

APPROXIMATIONS_ROLO311G = (
    "SOURCE_ROLO311G_KIEFFER_STONE2005_EQ10_TABLE4_EQ11",
    "ROLO311G_SECTION42_APOLLO_SCALE_ADJUSTMENT_ABSENT_NOT_TABULATED",
    "ROLO311G_BANDS_LOG_LINEAR_IN_WAVELENGTH_NO_APOLLO_SHAPE_FACTOR",
    "LIBRATION_IAU2009_WGCCRE_MOON_ROTATION_MODEL_ARCHINAL2011",
    "SELENOGRAPHIC_LONGITUDE_EAST_POSITIVE_WAXING_PHI_POSITIVE",
    "OPPOSITION_EFFECT_FROM_ROLO311G_D1_D2_TERMS",
    "SOLAR_SPECTRUM_TSIS1_HSRS_NOT_THE_FITS_WEHRLI1986",
)

APPROXIMATIONS_A7_GREY = (
    "PHASE_LAW_GREY_KS1991_EQ9",
    "REFLECTANCE_ROLO_AT_G7_ONLY_LIME2024_FIG1_DIGITISED",
    "LIBRATION_IGNORED",
)

TOKEN_ROLO_OUTSIDE_DOMAIN_NOT_EVALUATED = "ROLO311G_OUTSIDE_FITTED_PHASE_DOMAIN_NOT_EVALUATED"

TOKEN_FALLBACK_NOT_A_ROLO_VALIDATION = "FALLBACK_A7_KS1991_GREY_IS_NOT_A_ROLO311G_VALIDATION"

A7_GREY_PHASE_SUPPORT_DEG = (0.0, 180.0)

SOURCE_STATUS_ROLO_INSIDE = "MOON_SOURCE_ROLO311G_INSIDE_FITTED_DOMAIN"

SOURCE_STATUS_FALLBACK_A7 = "MOON_SOURCE_FALLBACK_A7_GREY_ROLO311G_OUTSIDE_FITTED_DOMAIN"

SOURCE_STATUS_A7_ALTERNATIVE = "MOON_SOURCE_A7_GREY_DECLARED_ALTERNATIVE"

STATUS_MOON_SOURCE_REFUSED_ROLO_NO_FALLBACK = "MOON_SOURCE_REFUSED_ROLO311G_OUTSIDE_FITTED_PHASE_DOMAIN_NO_FALLBACK"

STATUS_MOON_SOURCE_REFUSED_FALLBACK_OUTSIDE_SUPPORT = "MOON_SOURCE_REFUSED_ROLO311G_OUTSIDE_FITTED_DOMAIN_AND_FALLBACK_OUTSIDE_ITS_SUPPORT"

STATUS_MOON_SOURCE_REFUSED_A7_OUTSIDE_SUPPORT = "MOON_SOURCE_REFUSED_A7_GREY_OUTSIDE_ITS_SUPPORT"

STATUS_MOON_OCCULTED_BY_EARTH = "MOON_OCCULTED_BY_EARTH_NO_DIRECT_LIGHT"

STATUS_MOON_IN_EARTH_UMBRA = "MOON_IN_EARTH_UMBRA_NO_DIRECT_LIGHT"

STATUS_MOON_IN_FIELD_MASKED = "MOON_IN_FIELD_MASKED"

PST_INSTRUMENT_STATUS = "PST_UNKNOWN_FOR_THIS_CAMERA"

class MoonHaloError(ValueError):
    pass

def _read_two_columns(path):
    """A committed two-column table: '#' provenance lines, one header line, then numeric rows (LF, ASCII)."""
    raw = open(path, "rb").read()
    if b"\r" in raw:
        raise MoonHaloError(f"{path}: table must be LF bytes")
    lines = [ln for ln in raw.decode("ascii").splitlines() if ln.strip() and not ln.startswith("#")]
    rows = [tuple(float(x) for x in ln.split(",")[:2]) for ln in lines[1:]]
    if not rows:
        raise MoonHaloError(f"{path}: no numeric rows")
    return np.asarray(rows, dtype=np.float64)

def phase_factor(phase_angle_deg, *, opposition_surge=DEFAULT_OPPOSITION_SURGE):
    """Phi(g): Krisciunas & Schaefer 1991 Eq. 9 brightness relative to the reflectance phase angle (7 deg).

    `10^(-0.4 [0.026 (|g| - 7) + 4e-9 (g^4 - 7^4)])`; the optional opposition surge multiplies only inside
    7 deg (K&S report the Moon about 35 percent brighter than Eq. 9 near opposition: 1.0 is the low case,
    `OPPOSITION_SURGE_HIGH` the full-Moon high case).
    """
    g = np.abs(np.asarray(phase_angle_deg, dtype=np.float64))
    g0 = REFLECTANCE_PHASE_ANGLE_DEG
    dm = KS1991_PHASE_LINEAR_MAG_PER_DEG * (g - g0) + KS1991_PHASE_QUARTIC_MAG_PER_DEG4 * (g ** 4 - g0 ** 4)
    f = 10.0 ** (-0.4 * dm)
    surge = np.where(g < g0, float(opposition_surge), 1.0)
    return f * surge

class ReflectanceTable:
    """ROLO disk-equivalent reflectance A(lambda) at phase angle 7 deg, interpolated log-linearly in wavelength."""

    def __init__(self, wave_nm, reflectance):
        w = np.asarray(wave_nm, dtype=np.float64)
        a = np.asarray(reflectance, dtype=np.float64)
        if w.ndim != 1 or w.size < 2 or w.shape != a.shape or np.any(np.diff(w) <= 0) or np.any(a <= 0):
            raise MoonHaloError("reflectance table needs strictly increasing wavelengths and positive values")
        self.wave_nm, self.reflectance = w, a

    @classmethod
    def load_csv(cls, path):
        rows = _read_two_columns(path)
        return cls(rows[:, 0], rows[:, 1])

    def __call__(self, wave_nm):
        """A(lambda) on the requested grid; outside the table's coverage the end values are held (the camera's response
        and the solar table are already zero or negligible there; declared as `REFLECTANCE_HELD_AT_TABLE_ENDS_OUTSIDE_COVERAGE`)."""
        wave = np.asarray(wave_nm, dtype=np.float64)
        return np.exp(np.interp(np.clip(wave, self.wave_nm[0], self.wave_nm[-1]), self.wave_nm, np.log(self.reflectance)))

    @property
    def coverage_nm(self):
        return float(self.wave_nm[0]), float(self.wave_nm[-1])

def phase_angle_deg(moon_direction, moon_distance_km, sun_direction, sun_distance_km):
    """Sun-Moon-observer angle g (degrees, at the Moon) and the Moon-Sun distance (AU) from observer-relative vectors."""
    moon = np.asarray(moon_direction, dtype=np.float64) * float(moon_distance_km)
    sun = np.asarray(sun_direction, dtype=np.float64) * float(sun_distance_km)
    to_sun = sun - moon
    to_observer = -moon
    cosg = float(to_sun @ to_observer) / (np.linalg.norm(to_sun) * np.linalg.norm(to_observer))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosg)))), float(np.linalg.norm(to_sun)) / AU_KM

def _overlap_class(centre_sep_rad, obstacle_radius_rad, disk_radius_rad):
    """How a disk of angular radius r whose centre is `sep` from an obstacle of angular radius R overlaps it:
    TOTAL when the whole disk is inside (sep + r <= R), CLEAR when none of it is (sep - r >= R), PARTIAL between."""
    if centre_sep_rad + disk_radius_rad <= obstacle_radius_rad:
        return "TOTAL"
    if centre_sep_rad - disk_radius_rad >= obstacle_radius_rad:
        return "CLEAR"
    return "PARTIAL"

def direct_light_geometry(observer_gcrs_km, moon_direction, moon_distance_km, sun_direction, sun_distance_km):
    """The Moon's overlap with the Earth's limb (as seen from the observer) and with the Earth's umbra (as seen from the Earth's
    centre), each TOTAL / PARTIAL / CLEAR, with the angles that decided it.

    Occultation: the Moon's disk against the Earth's disk from the observer (Earth angular radius from the observer's distance,
    lunar angular radius from the Moon's distance; the Moon must lie beyond the Earth's centre). Eclipse: the Moon's disk against
    the umbra (half-angle from the Earth's and Sun's radii at the Moon's geocentric distance); the penumbra is not modelled and no
    illuminated fraction is estimated for a partial event (declared approximation): a partial overlap leaves the halo UNAVAILABLE.
    """
    obs = np.asarray(observer_gcrs_km, dtype=np.float64)
    m_dir = np.asarray(moon_direction, dtype=np.float64)
    moon = obs + m_dir * float(moon_distance_km)
    sun = obs + np.asarray(sun_direction, dtype=np.float64) * float(sun_distance_km)
    lunar_radius_obs = math.asin(min(1.0, MOON_RADIUS_KM / float(moon_distance_km)))
    out = {"occultation": "CLEAR", "umbra": "CLEAR", "lunar_angular_radius_deg": math.degrees(lunar_radius_obs)}
    d_obs = float(np.linalg.norm(obs))
    if d_obs > EARTH_RADIUS_KM and float(moon_distance_km) > d_obs:
        earth_dir = -obs / d_obs
        earth_radius = math.asin(min(1.0, EARTH_RADIUS_KM / d_obs))
        sep = math.acos(max(-1.0, min(1.0, float(m_dir @ earth_dir))))
        out["occultation"] = _overlap_class(sep, earth_radius, lunar_radius_obs)
        out.update({"earth_angular_radius_deg": math.degrees(earth_radius), "moon_earth_centre_separation_deg": math.degrees(sep)})
    d_moon = float(np.linalg.norm(moon))
    d_sun = float(np.linalg.norm(sun))
    anti_sun = -sun / d_sun
    ang = math.acos(max(-1.0, min(1.0, float(moon @ anti_sun) / d_moon)))
    umbra_half_angle = math.asin(min(1.0, EARTH_RADIUS_KM / d_moon)) - math.asin(min(1.0, (SUN_RADIUS_KM - EARTH_RADIUS_KM) / d_sun))
    lunar_radius_geo = math.asin(min(1.0, MOON_RADIUS_KM / d_moon))
    out["umbra"] = _overlap_class(ang, umbra_half_angle, lunar_radius_geo)
    out.update({"umbra_half_angle_deg": math.degrees(umbra_half_angle), "moon_anti_sun_angle_deg": math.degrees(ang),
                "lunar_angular_radius_geocentric_deg": math.degrees(lunar_radius_geo)})
    return out

def lunar_irradiance(wave_nm, *, reflectance_g7, phase_deg, moon_distance_km, moon_sun_distance_au,
                     solar_sed_w_m2_nm, opposition_surge=DEFAULT_OPPOSITION_SURGE):
    """E_M(lambda) at the observer, W m^-2 nm^-1: `(Omega_0/pi) A_7 Phi(g) E_sun (d_0/d)^2 (1 AU/r)^2`."""
    a7 = np.asarray(reflectance_g7, dtype=np.float64)
    sun = np.asarray(solar_sed_w_m2_nm, dtype=np.float64)
    if a7.shape != sun.shape:
        raise MoonHaloError("reflectance and solar spectrum must share the wavelength grid")
    geometry = (MOON_REF_DISTANCE_KM / float(moon_distance_km)) ** 2 / float(moon_sun_distance_au) ** 2
    return (OMEGA_MOON_REF_SR / math.pi) * a7 * float(phase_factor(phase_deg, opposition_surge=opposition_surge)) * sun * geometry

def lunar_irradiance_from_reflectance(wave_nm, *, reflectance, moon_distance_km, moon_sun_distance_au, solar_sed_w_m2_nm):
    """E_M(lambda) at the observer, W m^-2 nm^-1, with NO phase law: `(Omega_0/pi) A E_sun (d_0/d)^2 (1 AU/r)^2`
    (the ROLO route, whose reflectance already carries the phase angle, the libration and the opposition effect)."""
    a = np.asarray(reflectance, dtype=np.float64)
    sun = np.asarray(solar_sed_w_m2_nm, dtype=np.float64)
    if a.shape != sun.shape:
        raise MoonHaloError("reflectance and solar spectrum must share the wavelength grid")
    geometry = (MOON_REF_DISTANCE_KM / float(moon_distance_km)) ** 2 / float(moon_sun_distance_au) ** 2
    return (OMEGA_MOON_REF_SR / math.pi) * a * sun * geometry

ROLO_COLUMNS = ("a0", "a1", "a2", "a3", "b1", "b2", "b3", "d1", "d2", "d3")

ROLO_CONSTANTS = ("c1", "c2", "c3", "c4", "p1", "p2", "p3", "p4")

ROLO_PHASE_DOMAIN_DEG = (1.55, 97.0)

ROLO_EXPECTED_BANDS = 32

ROLO_PHASE_INSIDE = "INSIDE_FITTED_1P55_97_DEG"

ROLO_PHASE_BELOW = "OUTSIDE_FITTED_AT_OR_BELOW_1P55_DEG"

ROLO_PHASE_ABOVE = "OUTSIDE_FITTED_AT_OR_ABOVE_97_DEG"

def _read_committed_csv(path):
    """A committed table: '#' provenance lines, one header line, then comma-separated rows (LF, ASCII); returns (header, rows)."""
    raw = open(path, "rb").read()
    if b"\r" in raw or not raw.isascii():
        raise MoonHaloError(f"{path}: table must be LF ASCII bytes")
    lines = [ln for ln in raw.decode("ascii").splitlines() if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2:
        raise MoonHaloError(f"{path}: no rows")
    header = lines[0].split(",")
    return header, [ln.split(",") for ln in lines[1:]]

class Rolo311gTable:
    """Table 4 (32 bands x a0..a3, b1..b3, d1..d3) and the Eq. 11 constants (c1..c4, p1..p4) of the ROLO 311g model."""

    def __init__(self, wave_nm, coefficients, constants):
        w = np.asarray(wave_nm, dtype=np.float64)
        c = np.asarray(coefficients, dtype=np.float64)
        if w.ndim != 1 or w.size < 2 or np.any(np.diff(w) <= 0) or c.shape != (w.size, len(ROLO_COLUMNS)) or not np.all(np.isfinite(c)):
            raise MoonHaloError("ROLO table needs strictly increasing band wavelengths and ten finite coefficients per band")
        k = {name: float(constants[name]) for name in ROLO_CONSTANTS}
        if not all(math.isfinite(v) for v in k.values()) or k["p1"] <= 0 or k["p2"] <= 0 or k["p4"] <= 0:
            raise MoonHaloError("ROLO constants must be finite with positive p1, p2, p4")
        self.wave_nm, self.coefficients, self.constants = w, c, k

    @classmethod
    def load_csv(cls, table_path, constants_path):
        header, rows = _read_committed_csv(table_path)
        if header != ["wavelength_nm", *ROLO_COLUMNS]:
            raise MoonHaloError(f"{table_path}: header {header}")
        if len(rows) != ROLO_EXPECTED_BANDS:
            raise MoonHaloError(f"{table_path}: {len(rows)} bands, the model has {ROLO_EXPECTED_BANDS}")
        wave = [float(r[0]) for r in rows]
        coef = [[float(x) for x in r[1:]] for r in rows]
        header, crow = _read_committed_csv(constants_path)
        if header[:2] != ["name", "value"]:
            raise MoonHaloError(f"{constants_path}: header {header}")
        constants = {r[0]: float(r[1]) for r in crow}
        missing = [n for n in ROLO_CONSTANTS if n not in constants]
        if missing:
            raise MoonHaloError(f"{constants_path}: missing {missing}")
        return cls(wave, coef, constants)

    @property
    def coverage_nm(self):
        return float(self.wave_nm[0]), float(self.wave_nm[-1])

    @staticmethod
    def phase_domain(phase_deg):
        """INSIDE only for 1.55 < |g| < 97 deg (the fitted data's open interval); both edges are outside."""
        g = abs(float(phase_deg))
        if g <= ROLO_PHASE_DOMAIN_DEG[0]:
            return ROLO_PHASE_BELOW
        if g >= ROLO_PHASE_DOMAIN_DEG[1]:
            return ROLO_PHASE_ABOVE
        return ROLO_PHASE_INSIDE

    def ln_band_reflectance(self, *, phase_deg, sun_selenographic_longitude_deg, observer_selenographic_latitude_deg,
                            observer_selenographic_longitude_deg):
        """Eq. 10 per band: ln A_k (32,). g (absolute) in radians for the polynomial and in degrees for the exponential and
        cosine terms; Phi in radians; theta, phi in degrees."""
        g_deg = abs(float(phase_deg))
        g = math.radians(g_deg)
        big_phi = math.radians(_wrap_deg(float(sun_selenographic_longitude_deg)))
        theta = float(observer_selenographic_latitude_deg)
        phi = float(observer_selenographic_longitude_deg)
        a = self.coefficients
        k = self.constants
        out = a[:, 0] + a[:, 1] * g + a[:, 2] * g ** 2 + a[:, 3] * g ** 3
        out = out + a[:, 4] * big_phi + a[:, 5] * big_phi ** 3 + a[:, 6] * big_phi ** 5
        out = out + k["c1"] * theta + k["c2"] * phi + k["c3"] * big_phi * theta + k["c4"] * big_phi * phi
        out = out + a[:, 7] * math.exp(-g_deg / k["p1"]) + a[:, 8] * math.exp(-g_deg / k["p2"]) + a[:, 9] * math.cos((g_deg - k["p3"]) / k["p4"])
        return out

def _wrap_deg(x):
    """An angle in degrees on (-180, 180]."""
    y = math.fmod(float(x) + 180.0, 360.0)
    if y <= 0.0:
        y += 360.0
    return y - 180.0

def rolo311g_reflectance(wave_nm, *, table, phase_deg, sun_selenographic_longitude_deg, observer_selenographic_latitude_deg=0.0,
                         observer_selenographic_longitude_deg=0.0):
    """ROLO 311g disk-equivalent reflectance A(lambda) on the requested grid (Kieffer & Stone 2005 Eq. 10, Table 4, Eq. 11).

    ln A is interpolated linearly in wavelength between the band centres and held at the end bands outside 350.0-2383.6 nm
    (`REFLECTANCE_HELD_AT_TABLE_ENDS_OUTSIDE_COVERAGE`); no Apollo shape factor and no Section 4.2 scale adjustment (neither is
    tabulated in the paper). Longitudes east-positive (waxing Phi > 0); the Sun's selenographic latitude does not enter.
    """
    ln_a = table.ln_band_reflectance(phase_deg=phase_deg, sun_selenographic_longitude_deg=sun_selenographic_longitude_deg,
                                     observer_selenographic_latitude_deg=observer_selenographic_latitude_deg,
                                     observer_selenographic_longitude_deg=observer_selenographic_longitude_deg)
    wave = np.asarray(wave_nm, dtype=np.float64)
    return np.exp(np.interp(np.clip(wave, table.wave_nm[0], table.wave_nm[-1]), table.wave_nm, ln_a))

JD_J2000_TDB = 2451545.0

SPEED_OF_LIGHT_KM_S = 299792.458

IAU2009_MOON_ANGLES_DEG = ((125.045, -0.0529921), (250.089, -0.1059842), (260.008, 13.0120009), (176.625, 13.3407154),
                           (357.529, 0.9856003), (311.589, 26.4057084), (134.963, 13.0649930), (276.617, 0.3287146),
                           (34.226, 1.7484877), (15.134, -0.1589763), (119.743, 0.0036096), (239.961, 0.1643573),
                           (25.053, 12.9590088))

IAU2009_MOON_POLE_RA_DEG = (269.9949, 0.0031)                 # deg, deg per Julian century

IAU2009_MOON_POLE_DEC_DEG = (66.5392, 0.0130)

IAU2009_MOON_PM_DEG = (38.3213, 13.17635815, -1.4e-12)        # deg, deg/day, deg/day^2

IAU2009_MOON_RA_SIN = (-3.8787, -0.1204, 0.0700, -0.0172, 0.0, 0.0072, 0.0, 0.0, 0.0, -0.0052, 0.0, 0.0, 0.0043)

IAU2009_MOON_DEC_COS = (1.5419, 0.0239, -0.0278, 0.0068, 0.0, -0.0029, 0.0009, 0.0, 0.0, 0.0008, 0.0, 0.0, -0.0009)

IAU2009_MOON_PM_SIN = (3.5610, 0.1208, -0.0642, 0.0158, 0.0252, -0.0066, -0.0047, -0.0046, 0.0028, 0.0052, 0.0040, 0.0019, -0.0044)

LIBRATION_MODEL = ("IAU/WGCCRE 2009 Moon rotation elements (Archinal et al. 2011, Celest. Mech. Dyn. Astron. 109, 101; the values NAIF "
                   "pck00010.tpc carries): pole RA/Dec and prime meridian W with the 13 periodic terms E1-E13, TDB, evaluated at the "
                   "epoch minus the Moon's light time; body-fixed = R3(W) R1(90 - dec0) R3(90 + ra0) x GCRS (GCRS axes taken as ICRF); "
                   "longitudes east-positive")

def moon_orientation_iau2009(jd_tdb):
    """(alpha0, delta0, W) of the Moon in degrees at a TDB Julian date (IAU/WGCCRE 2009)."""
    d = float(jd_tdb) - JD_J2000_TDB
    t = d / 36525.0
    e = [math.radians(b + r * d) for b, r in IAU2009_MOON_ANGLES_DEG]
    ra = IAU2009_MOON_POLE_RA_DEG[0] + IAU2009_MOON_POLE_RA_DEG[1] * t + sum(c * math.sin(x) for c, x in zip(IAU2009_MOON_RA_SIN, e))
    dec = IAU2009_MOON_POLE_DEC_DEG[0] + IAU2009_MOON_POLE_DEC_DEG[1] * t + sum(c * math.cos(x) for c, x in zip(IAU2009_MOON_DEC_COS, e))
    w = IAU2009_MOON_PM_DEG[0] + IAU2009_MOON_PM_DEG[1] * d + IAU2009_MOON_PM_DEG[2] * d * d + sum(c * math.sin(x) for c, x in zip(IAU2009_MOON_PM_SIN, e))
    return ra, dec, w

def _frame_rotation(axis, angle_deg):
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    if axis == 1:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

def icrf_to_moon_body_fixed(jd_tdb):
    """The 3x3 rotation taking ICRF (GCRS-axis) vectors into the Moon's body-fixed frame (x prime meridian, z north pole)."""
    ra, dec, w = moon_orientation_iau2009(jd_tdb)
    return _frame_rotation(3, w) @ _frame_rotation(1, 90.0 - dec) @ _frame_rotation(3, 90.0 + ra)

def _lat_lon_deg(v):
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    return math.degrees(math.asin(max(-1.0, min(1.0, float(v[2]))))), _wrap_deg(math.degrees(math.atan2(float(v[1]), float(v[0]))))

def selenographic_geometry(jd_tdb, moon_direction, moon_distance_km, sun_direction, sun_distance_km):
    """The observer's selenographic latitude/longitude and the Sun's selenographic latitude/longitude (degrees, longitudes
    east-positive on (-180, 180]) from observer-relative GCRS directions and distances, at the Moon's light-time-corrected epoch."""
    moon = np.asarray(moon_direction, dtype=np.float64) * float(moon_distance_km)
    sun = np.asarray(sun_direction, dtype=np.float64) * float(sun_distance_km)
    jd_moon = float(jd_tdb) - float(moon_distance_km) / SPEED_OF_LIGHT_KM_S / 86400.0
    rot = icrf_to_moon_body_fixed(jd_moon)
    obs_lat, obs_lon = _lat_lon_deg(rot @ (-moon))
    sun_lat, sun_lon = _lat_lon_deg(rot @ (sun - moon))
    return {"observer_selenographic_latitude_deg": obs_lat, "observer_selenographic_longitude_deg": obs_lon,
            "sun_selenographic_latitude_deg": sun_lat, "sun_selenographic_longitude_deg": sun_lon, "jd_tdb_at_moon": jd_moon}

DIRECT_LIGHT = "DIRECT"

def _source_route_support(source_route, rolo_out_of_domain_fallback, a7_phase_support_deg):
    """The source configuration's first checks, made before anything else (moon_halo_provider makes them before it reads its
    kernel): the route, the fallback and the A_7 route's declared phase support. Returns the support as (lo, hi) degrees."""
    if source_route not in SOURCE_ROUTES:
        raise MoonHaloError(f"unknown source route {source_route!r}")
    if rolo_out_of_domain_fallback not in (None, SOURCE_ROUTE_A7_GREY):
        raise MoonHaloError(f"the only configurable out-of-domain fallback is {SOURCE_ROUTE_A7_GREY}, not {rolo_out_of_domain_fallback!r}")
    if rolo_out_of_domain_fallback is not None and source_route != SOURCE_ROUTE_ROLO311G:
        raise MoonHaloError("an out-of-domain fallback belongs to the ROLO311G route only")
    s_lo, s_hi = (float(a7_phase_support_deg[0]), float(a7_phase_support_deg[1]))
    if not (0.0 <= s_lo < s_hi <= 180.0):
        raise MoonHaloError("the A7_GREY phase support must be an interval inside 0-180 deg")
    return s_lo, s_hi

def _lunar_source(wave_nm, *, solar_sed_w_m2_nm, rolo_table, reflectance_table, source_route, rolo_out_of_domain_fallback,
                  a7_support, opposition_surge, route_tokens):
    """The rest of the source's construction (moon_halo_provider makes it after reading its kernel): the grid, the route's
    tables and the surge case, then the resolution rule of the specification's section 1.3 as `source(geometry, g, r_au) ->
    (E_M on the grid or None when refused, the reflectance on the grid or None, the route metadata)`. ROLO is never evaluated
    outside its fitted open interval. `route_tokens(route, surge=None)` states a route's approximation tokens wherever the
    rule records them: moon_halo_provider passes `scenario_tokens` with its kernel's colour case (v1's list), and
    lunar_source_provider passes `source_tokens`. Returns (wave, i550, source)."""
    s_lo, s_hi = a7_support
    wave = np.asarray(wave_nm, dtype=np.float64)
    sun = np.asarray(solar_sed_w_m2_nm, dtype=np.float64)
    i550 = int(np.argmin(np.abs(wave - 550.0)))
    uses_a7 = source_route == SOURCE_ROUTE_A7_GREY or rolo_out_of_domain_fallback == SOURCE_ROUTE_A7_GREY
    if source_route == SOURCE_ROUTE_ROLO311G:
        if rolo_table is None:
            raise MoonHaloError("the default ROLO311G route needs rolo_table (pass source_route=SOURCE_ROUTE_A7_GREY for the retired route)")
        if opposition_surge is not None:
            raise MoonHaloError("the ROLO311G route carries its own opposition effect (Eq. 10 d1, d2 terms); an opposition_surge is refused")
    a7 = None
    surge = None
    if uses_a7:
        if reflectance_table is None:
            raise MoonHaloError("the A7_GREY route (alternative or configured fallback) needs reflectance_table")
        a7 = reflectance_table(wave)
        surge = DEFAULT_OPPOSITION_SURGE if opposition_surge is None else float(opposition_surge)

    def a7_value(g, d_moon, r_au):
        e = lunar_irradiance(wave, reflectance_g7=a7, phase_deg=g, moon_distance_km=d_moon, moon_sun_distance_au=r_au,
                             solar_sed_w_m2_nm=sun, opposition_surge=surge)
        return e, a7, route_tokens(SOURCE_ROUTE_A7_GREY, surge)

    def a7_supported(g):
        return s_lo <= abs(float(g)) <= s_hi

    def source(geometry, g, r_au):
        """The resolution rule (specification section 1.3): E_M on the grid (None when refused), the reflectance on the grid,
        and the route metadata. ROLO is never evaluated outside its fitted open interval."""
        d_moon = float(geometry["moon_distance_km"])
        meta = {"source_route_requested": source_route, "rolo_out_of_domain_fallback": rolo_out_of_domain_fallback,
                "a7_phase_support_deg": [s_lo, s_hi] if uses_a7 else None}
        if source_route == SOURCE_ROUTE_A7_GREY:
            if not a7_supported(g):
                meta.update({"source_route": None, "source_status": STATUS_MOON_SOURCE_REFUSED_A7_OUTSIDE_SUPPORT, "opposition_surge": surge,
                             "source_refusal_reason": f"phase {g:.4f} deg outside the A7_GREY route's declared support {s_lo:g}-{s_hi:g} deg",
                             "approximation": route_tokens(SOURCE_ROUTE_A7_GREY, surge)})
                return None, None, meta
            e, refl, tokens = a7_value(g, d_moon, r_au)
            meta.update({"source_route": SOURCE_ROUTE_A7_GREY, "source_status": SOURCE_STATUS_A7_ALTERNATIVE, "opposition_surge": surge,
                         "approximation": tokens})
            return e, refl, meta
        jd = geometry.get("epoch_tdb_jd")
        if jd is None or not math.isfinite(float(jd)):
            raise MoonHaloError("the ROLO311G route needs the exposure epoch as geometry['epoch_tdb_jd'] (sky_background.scene_geometry supplies it)")
        sel = selenographic_geometry(float(jd), geometry["moon_direction_gcrs"], d_moon, geometry["sun_direction_gcrs"], geometry["sun_distance_km"])
        domain = Rolo311gTable.phase_domain(g)
        meta.update({"rolo_phase_domain": domain, "selenographic": {key: float(v) for key, v in sel.items()}, "libration_model": LIBRATION_MODEL})
        if domain == ROLO_PHASE_INSIDE:
            refl = rolo311g_reflectance(wave, table=rolo_table, phase_deg=g, sun_selenographic_longitude_deg=sel["sun_selenographic_longitude_deg"],
                                        observer_selenographic_latitude_deg=sel["observer_selenographic_latitude_deg"],
                                        observer_selenographic_longitude_deg=sel["observer_selenographic_longitude_deg"])
            e = lunar_irradiance_from_reflectance(wave, reflectance=refl, moon_distance_km=d_moon, moon_sun_distance_au=r_au, solar_sed_w_m2_nm=sun)
            meta.update({"source_route": SOURCE_ROUTE_ROLO311G, "source_status": SOURCE_STATUS_ROLO_INSIDE, "opposition_surge": None,
                         "approximation": route_tokens(SOURCE_ROUTE_ROLO311G)})
            return e, refl, meta
        meta["rolo_out_of_domain_reason"] = (f"ROLO311G not evaluated: phase angle at the Moon {g:.4f} deg lies outside the model's fitted interval "
                                             f"{ROLO_PHASE_DOMAIN_DEG[0]:g} < g < {ROLO_PHASE_DOMAIN_DEG[1]:g} deg (Kieffer & Stone 2005, p. 2895; {domain})")
        if rolo_out_of_domain_fallback is None:
            meta.update({"source_route": None, "source_status": STATUS_MOON_SOURCE_REFUSED_ROLO_NO_FALLBACK, "opposition_surge": None,
                         "source_refusal_reason": meta["rolo_out_of_domain_reason"] + "; no fallback configured",
                         "approximation": route_tokens(SOURCE_ROUTE_ROLO311G)
                         + [TOKEN_ROLO_OUTSIDE_DOMAIN_NOT_EVALUATED]})
            return None, None, meta
        meta["fallback_route"] = rolo_out_of_domain_fallback
        if not a7_supported(g):
            meta.update({"source_route": None, "source_status": STATUS_MOON_SOURCE_REFUSED_FALLBACK_OUTSIDE_SUPPORT, "opposition_surge": surge,
                         "source_refusal_reason": (meta["rolo_out_of_domain_reason"] + f"; the configured fallback {rolo_out_of_domain_fallback} "
                                                   f"refuses too: phase outside its declared support {s_lo:g}-{s_hi:g} deg"),
                         "approximation": route_tokens(SOURCE_ROUTE_A7_GREY, surge)
                         + [TOKEN_ROLO_OUTSIDE_DOMAIN_NOT_EVALUATED]})
            return None, None, meta
        e, refl, tokens = a7_value(g, d_moon, r_au)
        meta.update({"source_route": SOURCE_ROUTE_A7_GREY, "source_status": SOURCE_STATUS_FALLBACK_A7, "opposition_surge": surge,
                     "approximation": tokens + [TOKEN_ROLO_OUTSIDE_DOMAIN_NOT_EVALUATED, TOKEN_FALLBACK_NOT_A_ROLO_VALIDATION]})
        return e, refl, meta

    return wave, i550, source

def _moon_seen_from(geometry):
    """The Moon of a geometry seen from its observer, read as moon_halo_provider has always read it: the Moon's direction,
    the observer's GCRS position (the geocentre when the geometry carries none), the phase angle at the Moon and the
    Moon-Sun distance. Returns (moon_dir, observer, g, r_au)."""
    moon_dir = np.asarray(geometry["moon_direction_gcrs"], dtype=np.float64)
    observer = np.asarray(geometry.get("observer_gcrs_km", np.zeros(3)), dtype=np.float64)
    g, r_au = phase_angle_deg(moon_dir, geometry["moon_distance_km"], geometry["sun_direction_gcrs"], geometry["sun_distance_km"])
    return moon_dir, observer, g, r_au

V2_SCHEMA = "synth-sky-background/v1:moon_stray_light_v2"

V2_MODE = "MOON_STRAY_LIGHT_V2_LIU2021_PST_ADDED_RATE"

V2_CENTRE_REFUSAL_DEG = 28.0

V2_PST_ZERO_FROM_DEG = 90.0

PST_FIT_SCHEMA = "physics-moon-pst-fit/v1"

PST_MODEL_LIU2021 = "LIU2021_DEPLOYABLE_BAFFLE_FIG15_EXPONENTIAL_FIT"

COMMON_DISK_INTEGRAL_N = 20000

STATUS_V2_ADDED = "MOON_OUT_OF_FIELD_STRAY_LIGHT_ADDED_LIU2021_PST"

STATUS_V2_PST_ZERO_AT_OR_BEYOND_90 = "MOON_OUT_OF_FIELD_PST_ZERO_AT_OR_BEYOND_90_DEG_DECLARED"

STATUS_V2_REFUSED_CENTRE_WITHIN_28 = "MOON_REFUSED_CENTRE_WITHIN_28_DEG_OF_BORESIGHT"

STATUS_V2_REFUSED_DISK_IN_OR_NEAR_FIELD = "MOON_REFUSED_DISK_IN_OR_NEAR_FIELD_OUTSIDE_THE_PST_DOMAIN"

STATUS_MOON_PARTIALLY_OCCULTED_AREA_WEIGHTED = "MOON_PARTIALLY_OCCULTED_BY_EARTH_AREA_WEIGHTED"

STATUS_MOON_PARTIALLY_IN_UMBRA_AREA_WEIGHTED = "MOON_PARTIALLY_IN_EARTH_UMBRA_AREA_WEIGHTED"

STATUS_MOON_PARTIALLY_OCCULTED_AND_IN_UMBRA_AREA_WEIGHTED = "MOON_PARTIALLY_OCCULTED_BY_EARTH_AND_PARTIALLY_IN_EARTH_UMBRA_AREA_WEIGHTED"

AREA_WEIGHT_WHOLE_DISK = "WHOLE_DISK_LIT"

AREA_WEIGHT_TOTAL_EVENT = "TOTAL_EVENT_ZERO"

AREA_WEIGHT_CLOSED_FORM = "CLOSED_FORM_TWO_CIRCLE_OVERLAP"

AREA_WEIGHT_COMMON_DISK = "COMMON_DISK_INTEGRAL"

V2_CONVERSION_RULE = "C_IRR_4N2_OVER_PI_T_OPTICS_LAMBDA"

V2_SPATIAL_CONVENTION = "UNIFORM_SKY_EQUIVALENT_RADIANCE_A"

V2_SCENARIO_TOKEN = "MOON_V2_PST_LIU2021_FIG15_EXPFIT_C_IRR_T_OPTICS_UNIFORM_SKY_EQUIVALENT_GREY_ZERO_BEYOND_90"

SOLAR_STRAY_LIGHT_V2 = "SOLAR_STRAY_LIGHT_OUTSIDE_THE_REFUSAL_UNMODELLED"

APPROXIMATIONS_V2_COMMON_SOURCE = (
    "EARTH_SHADOW_UMBRA_ONLY_NO_PENUMBRA",
    "EARTHSHINE_ON_THE_MOON_IGNORED",
    "REFLECTANCE_HELD_AT_TABLE_ENDS_OUTSIDE_COVERAGE",
)

APPROXIMATIONS_V2 = (
    "PST_LIU2021_DEPLOYABLE_BAFFLE_FIG15_SIMULATION_EXPONENTIAL_FIT_SYSTEMATIC_RESIDUALS",
    "PST_ZERO_AT_AND_BEYOND_90_DEG",
    "PST_GREY_SAME_AT_EVERY_WAVELENGTH",
    "PST_AT_THE_FRAME_MOON_CENTRE_SEPARATION",
    "CONVERSION_C_IRR_4N2_OVER_PI_T_OPTICS_LAMBDA_NOT_C_POW",
    "UNIFORM_SKY_EQUIVALENT_RADIANCE_DETECTOR_PATTERN_PIXEL_SOLID_ANGLE_TIMES_VIGNETTE",
    "ZERO_WHERE_THE_OPTICS_TRANSMISSION_IS_ZERO",
    "REFUSED_MOON_CENTRE_WITHIN_28_DEG_OF_BORESIGHT_CLOSED",
    "REFUSAL_28_DEG_IS_LIU2021_SUNLIGHT_SUPPRESSION_ANGLE_APPLIED_TO_THE_MOON",
    "PARTIAL_EVENT_UNIFORM_DISK_AREA_WEIGHT_FLAT_TWO_CIRCLE",
    "SIMULTANEOUS_PARTIAL_EVENTS_COMMON_DISK_INTEGRAL",
    "PARTIAL_EVENT_CENTROID_SHIFT_OMITTED_FIRST_ORDER",
    "SOLAR_STRAY_LIGHT_OUTSIDE_THE_REFUSAL_UNMODELLED",
)

def v2_tokens(*, route=SOURCE_ROUTE_ROLO311G, opposition_surge=None):
    """The v2 mode's token list: the route's source tokens, v1's source facts that stay true, the surge case on the retired
    route, and the v2 tokens. (The provider appends the ROLO out-of-domain and fallback tokens as v1's source rule does.)"""
    if route == SOURCE_ROUTE_ROLO311G:
        return list(APPROXIMATIONS_ROLO311G) + list(APPROXIMATIONS_V2_COMMON_SOURCE) + list(APPROXIMATIONS_V2)
    if route == SOURCE_ROUTE_A7_GREY:
        surge = DEFAULT_OPPOSITION_SURGE if opposition_surge is None else opposition_surge
        return (list(APPROXIMATIONS_A7_GREY) + list(APPROXIMATIONS_V2_COMMON_SOURCE)
                + [f"OPPOSITION_SURGE_{float(surge):g}".replace(".", "P")] + list(APPROXIMATIONS_V2))
    raise MoonHaloError(f"unknown source route {route!r}")

class Liu2021ExponentialPST:
    """The HUMAN's 2026-10-18 PST law fitted to Liu et al. 2021 Fig. 15 (the deployable baffle), degrees from the boresight:

        PST(t) = A [exp(-k (t - 28)) - exp(-k (90 - 28))]   for 28 < t < 90;   0 for t >= 90;   refused for t <= 28

    A and k come from the committed fit record (`load_json`). The law is a SIMULATED, grey, detector-plane illuminance ratio
    (Liu Eq. 39) with systematic residuals the record states; it is an assumption about this camera, never its measured PST.
    """
    name = PST_MODEL_LIU2021

    def __init__(self, a, k_per_deg, *, lower_deg=V2_CENTRE_REFUSAL_DEG, upper_deg=V2_PST_ZERO_FROM_DEG):
        a, k, lo, hi = float(a), float(k_per_deg), float(lower_deg), float(upper_deg)
        if not all(math.isfinite(v) for v in (a, k, lo, hi)) or a <= 0.0 or k <= 0.0:
            raise MoonHaloError("the PST law needs a finite positive amplitude A and rate k")
        if lo != V2_CENTRE_REFUSAL_DEG or hi != V2_PST_ZERO_FROM_DEG:
            raise MoonHaloError(f"the PST law's support is the declared {V2_CENTRE_REFUSAL_DEG:g}-{V2_PST_ZERO_FROM_DEG:g} deg, not {lo:g}-{hi:g}")
        self.a, self.k, self.lower_deg, self.upper_deg = a, k, lo, hi

    @classmethod
    def load_json(cls, path, *, sha256=None):
        """The law from the committed fit record (LF bytes, schema `physics-moon-pst-fit/v1`), optionally pinned by sha256."""
        import hashlib
        import json
        raw = open(path, "rb").read()
        if b"\r" in raw:
            raise MoonHaloError(f"{path}: the fit record must be LF bytes")
        if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
            raise MoonHaloError(f"{path}: sha256 differs from the declared {sha256}")
        record = json.loads(raw.decode("ascii"))
        if record.get("schema") != PST_FIT_SCHEMA:
            raise MoonHaloError(f"{path}: schema {record.get('schema')!r} is not {PST_FIT_SCHEMA}")
        fit = record["fit"]
        return cls(fit["a"], fit["k_per_deg"], lower_deg=fit["lower_deg"], upper_deg=fit["upper_deg"])

    def parameters(self):
        return {"a": self.a, "k_per_deg": self.k, "lower_deg": self.lower_deg, "upper_deg": self.upper_deg}

    def __call__(self, angle_deg):
        t = float(angle_deg)
        if not math.isfinite(t) or t < 0.0 or t > 180.0:
            raise MoonHaloError(f"PST asked at {angle_deg!r} deg: an angle from the boresight lies in 0-180 deg")
        if t <= self.lower_deg:
            raise MoonHaloError(f"PST is not evaluated at or inside {self.lower_deg:g} deg: such a scene is refused")
        if t >= self.upper_deg:
            return 0.0
        return self.a * (math.exp(-self.k * (t - self.lower_deg)) - math.exp(-self.k * (self.upper_deg - self.lower_deg)))

def angle_between_deg(a, b):
    """The angle between direction vectors, degrees, row by row (broadcast): atan2(|a x b|, a . b), exact near 0 and 180."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if a.shape[-1] != 3 or b.shape[-1] != 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise MoonHaloError("directions must be finite 3-vectors")
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if np.any(na <= 0.0) or np.any(nb <= 0.0):
        raise MoonHaloError("a direction of zero length has no angle")
    a, b = a / na[..., None], b / nb[..., None]
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))

def centre_separation_refusal(boresights_gcrs, centre_directions_gcrs, *, angle_deg=V2_CENTRE_REFUSAL_DEG):
    """The closed 28-deg rule over an exposure: the smallest boresight-to-centre angle over the samples, and whether it refuses.

    `boresights_gcrs` and `centre_directions_gcrs` are (K, 3) or (3,) (broadcast) samples of the exposure's geometry; the
    rule refuses when the minimum is <= `angle_deg` (equality refuses, as v1's `theta_b <= theta_cut` selects). It serves the
    Moon's centre and, for a consumer's census, the Sun's."""
    sep = angle_between_deg(boresights_gcrs, centre_directions_gcrs)
    m = float(np.min(sep))
    return {"min_separation_deg": m, "refused": bool(m <= float(angle_deg)), "n_samples": int(sep.size), "angle_deg": float(angle_deg)}

def _v2_exposure_separation(geometry, moon_dir):
    """The frame's Moon-boresight separation (v1's key) and the minimum over the exposure samples a consumer supplies
    (`exposure_boresights_gcrs`, with `exposure_moon_directions_gcrs` or the frame's Moon direction)."""
    frame = float(geometry["moon_boresight_separation_deg"])
    if not math.isfinite(frame):
        raise MoonHaloError("the frame's moon_boresight_separation_deg is not finite")
    least, n = frame, 1
    samples = geometry.get("exposure_boresights_gcrs")
    if samples is not None:
        centres = geometry.get("exposure_moon_directions_gcrs")
        rule = centre_separation_refusal(samples, moon_dir if centres is None else centres)
        least, n = min(least, rule["min_separation_deg"]), n + rule["n_samples"]
    return {"frame_separation_deg": frame, "min_separation_deg": least, "n_samples": n, "refused": bool(least <= V2_CENTRE_REFUSAL_DEG)}

def _segment_area(rho, height):
    """Area of the segment of a circle of radius `rho` that lies beyond a chord at `height` from its edge (0 <= h <= 2 rho),
    computed without the cancellation of `rho^2 acos(1 - h/rho) - ...` when the circle is much larger than the segment."""
    h = min(max(float(height), 0.0), 2.0 * rho)
    if h <= rho:
        half = 2.0 * math.asin(math.sqrt(h / (2.0 * rho)))
    else:
        half = math.pi - 2.0 * math.asin(math.sqrt((2.0 * rho - h) / (2.0 * rho)))
    u = 2.0 * half
    f = u - math.sin(u) if u > 1e-3 else u ** 3 / 6.0 - u ** 5 / 120.0 + u ** 7 / 5040.0
    return 0.5 * rho * rho * f

def disk_covered_fraction(separation, disk_radius, obstacle_radius):
    """The fraction of a uniform disk of radius r covered by an obstacle disk of radius R whose centre is `separation` away:
    the closed-form flat two-circle overlap (lens) area over pi r^2. 1 when the disk lies inside the obstacle, 0 when apart
    (or when R <= 0: no obstacle), (R / r)^2 when the obstacle lies inside the disk."""
    d, r, big = float(separation), float(disk_radius), float(obstacle_radius)
    if not all(math.isfinite(v) for v in (d, r, big)) or d < 0.0 or r <= 0.0:
        raise MoonHaloError("the overlap needs a finite separation >= 0 and a finite positive disk radius")
    if big <= 0.0 or d - r >= big:
        return 0.0
    if d + r <= big:
        return 1.0
    if d + big <= r:
        return (big / r) ** 2
    x_r = ((d - big) * (d + big) + r * r) / (2.0 * d)        # the radical line's distance from the disk's centre
    area = _segment_area(r, r - x_r) + _segment_area(big, x_r - (d - big))
    return min(1.0, max(0.0, area / (math.pi * r * r)))

def common_disk_unmasked_fraction(masks, *, n=COMMON_DISK_INTEGRAL_N):
    """w = (1/pi) x the area of the unit lunar disk outside EVERY mask, a deterministic chord integral.

    `masks` holds one or two (cx, cy, radius) disks in common lunar-disk coordinates (units of the lunar radius, origin at the
    disk's centre). At each abscissa x the vertical chord of the unit disk minus the union of the masks' chords is measured
    exactly (inclusion-exclusion); the chord length is integrated with `n` midpoints in t, x = sin t, which removes the square
    root at the disk's edge. The area surviving both masks is NOT the product of the two surviving fractions: two half disks
    cut from opposite sides leave 0, from the same side 0.5, and the product says 0.25 for both."""
    masks = [tuple(float(v) for v in m) for m in masks]
    if not 1 <= len(masks) <= 2 or any(len(m) != 3 or not all(math.isfinite(v) for v in m) for m in masks):
        raise MoonHaloError("one or two finite (cx, cy, radius) masks")
    t = (np.arange(int(n)) + 0.5) / int(n) * math.pi - 0.5 * math.pi
    x = np.sin(t)
    half = np.cos(t)
    chords = []
    for cx, cy, radius in masks:
        dx = np.abs(x - cx)
        inside = dx < radius
        b = np.where(inside, np.sqrt(np.clip((radius - dx) * (radius + dx), 0.0, None)), 0.0)
        chords.append((np.where(inside, cy - b, 0.0), np.where(inside, cy + b, 0.0), inside))

    def overlap(*intervals):
        return np.clip(np.minimum.reduce([iv[1] for iv in intervals]) - np.maximum.reduce([iv[0] for iv in intervals]), 0.0, None)

    disk = (-half, half)
    length = 2.0 * half
    for lo, hi, inside in chords:
        length = length - np.where(inside, overlap(disk, (lo, hi)), 0.0)
    if len(chords) == 2:
        (l1, h1, i1), (l2, h2, i2) = chords
        length = length + np.where(i1 & i2, overlap(disk, (l1, h1), (l2, h2)), 0.0)
    w = float(np.sum(length * half)) / int(n)               # (1/pi) sum(length cos t) (pi/n)
    return min(1.0, max(0.0, w))

def _disk_frame(direction):
    """An orthonormal pair spanning the plane perpendicular to `direction` (the lunar disk as the observer sees it)."""
    m = np.asarray(direction, dtype=np.float64)
    m = m / np.linalg.norm(m)
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(m[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(m, helper)
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(m, e1)

def _toward_in_disk(at, toward, e1, e2):
    """The unit 2-vector, in the disk frame (e1, e2), of the tangent direction at `at` toward `toward`."""
    at = np.asarray(at, dtype=np.float64) / np.linalg.norm(at)
    toward = np.asarray(toward, dtype=np.float64) / np.linalg.norm(toward)
    tangent = toward - float(toward @ at) * at
    v = np.array([float(tangent @ e1), float(tangent @ e2)])
    norm = float(np.linalg.norm(v))
    return np.array([1.0, 0.0]) if norm < 1e-15 else v / norm

def direct_light_area_weight(observer_gcrs_km, moon_direction, moon_distance_km, sun_direction, sun_distance_km, *,
                             n=COMMON_DISK_INTEGRAL_N):
    """v2's direct-light verdict: the fraction w of the uniform lunar disk that is neither hidden by the Earth nor in its umbra.

    v1's `direct_light_geometry` classifies each event TOTAL / PARTIAL / CLEAR. TOTAL comes before PARTIAL: a TOTAL event of
    either kind gives w = 0 with v1's no-direct-light status. One PARTIAL event gives w = 1 - the closed-form two-circle
    covered fraction (flat angles: the Moon's disk against the Earth's disk seen from the observer, or against the umbra seen
    from the Earth's centre). Both PARTIAL gives the common-disk integral: both masks placed on one lunar disk, in units of
    the lunar radius, oriented by the directions toward the Earth's centre (observer's sky) and toward the anti-Sun axis
    (geocentric sky), the latter mapped by the Moon's physical displacement (parallax foreshortening neglected). Invalid
    geometry raises; it never becomes zero light. Returns {status (None when the whole disk is lit), area_weight, method,
    geometry (v1's dict), covered_by_earth, covered_by_umbra}."""
    obs = np.asarray(observer_gcrs_km, dtype=np.float64)
    m_dir = np.asarray(moon_direction, dtype=np.float64)
    s_dir = np.asarray(sun_direction, dtype=np.float64)
    d_moon, d_sun = float(moon_distance_km), float(sun_distance_km)
    if obs.shape != (3,) or m_dir.shape != (3,) or s_dir.shape != (3,) or not (np.isfinite(obs).all() and np.isfinite(m_dir).all()
                                                                          and np.isfinite(s_dir).all()):
        raise MoonHaloError("the direct-light weight needs finite 3-vectors for the observer, the Moon and the Sun")
    if abs(float(np.linalg.norm(m_dir)) - 1.0) > 1e-9 or abs(float(np.linalg.norm(s_dir)) - 1.0) > 1e-9:
        raise MoonHaloError("the Moon and Sun directions must be unit vectors")
    d_obs = float(np.linalg.norm(obs))
    if not d_obs > EARTH_RADIUS_KM:
        raise MoonHaloError("the direct-light weight needs an observer above the Earth's surface (observer_gcrs_km)")
    if not (math.isfinite(d_moon) and d_moon > d_obs and math.isfinite(d_sun) and d_sun > d_moon):
        raise MoonHaloError("the Moon must lie beyond the observer's distance from the Earth, and the Sun beyond the Moon")
    g = direct_light_geometry(obs, m_dir, d_moon, s_dir, d_sun)
    out = {"geometry": g, "covered_by_earth": None, "covered_by_umbra": None}
    if g["occultation"] == "TOTAL":
        out.update(status=STATUS_MOON_OCCULTED_BY_EARTH, area_weight=0.0, method=AREA_WEIGHT_TOTAL_EVENT)
        return out
    if g["umbra"] == "TOTAL":
        out.update(status=STATUS_MOON_IN_EARTH_UMBRA, area_weight=0.0, method=AREA_WEIGHT_TOTAL_EVENT)
        return out
    occ, umb = g["occultation"] == "PARTIAL", g["umbra"] == "PARTIAL"
    r_obs = math.radians(g["lunar_angular_radius_deg"])
    r_geo = math.radians(g["lunar_angular_radius_geocentric_deg"])
    if occ:
        s, big_e = math.radians(g["moon_earth_centre_separation_deg"]), math.radians(g["earth_angular_radius_deg"])
        out["covered_by_earth"] = disk_covered_fraction(s, r_obs, big_e)
    if umb:
        a, big_u = math.radians(g["moon_anti_sun_angle_deg"]), math.radians(g["umbra_half_angle_deg"])
        out["covered_by_umbra"] = disk_covered_fraction(a, r_geo, big_u)
    if occ and umb:
        e1, e2 = _disk_frame(m_dir)
        t_earth = _toward_in_disk(m_dir, -obs / d_obs, e1, e2)
        moon_geo = obs + m_dir * d_moon
        anti_sun = -(obs + s_dir * d_sun)
        t_umbra = _toward_in_disk(moon_geo, anti_sun, e1, e2)
        masks = [(s / r_obs * t_earth[0], s / r_obs * t_earth[1], big_e / r_obs),
                 (a / r_geo * t_umbra[0], a / r_geo * t_umbra[1], big_u / r_geo)]
        out.update(status=STATUS_MOON_PARTIALLY_OCCULTED_AND_IN_UMBRA_AREA_WEIGHTED,
                   area_weight=common_disk_unmasked_fraction(masks, n=n), method=AREA_WEIGHT_COMMON_DISK, masks_in_lunar_radii=masks)
    elif occ:
        out.update(status=STATUS_MOON_PARTIALLY_OCCULTED_AREA_WEIGHTED, area_weight=1.0 - out["covered_by_earth"], method=AREA_WEIGHT_CLOSED_FORM)
    elif umb:
        out.update(status=STATUS_MOON_PARTIALLY_IN_UMBRA_AREA_WEIGHTED, area_weight=1.0 - out["covered_by_umbra"], method=AREA_WEIGHT_CLOSED_FORM)
    else:
        out.update(status=None, area_weight=1.0, method=AREA_WEIGHT_WHOLE_DISK)
    w = out["area_weight"]
    if not (math.isfinite(w) and 0.0 <= w <= 1.0):
        raise MoonHaloError(f"the area weight {w!r} left [0, 1]")
    return out

def _optics_transmission(optics_transmission, shape):
    t = np.asarray(optics_transmission, dtype=np.float64)
    if t.shape != shape or not np.isfinite(t).all() or np.any(t < 0.0) or np.any(t > 1.0) or not np.any(t > 0.0):
        raise MoonHaloError("optics_transmission must be T_optics(lambda) on the wave grid: finite, in [0, 1], not all zero "
                            "(t_lens x t_window x t_filter of the camera whose response the fold uses)")
    return t

def _f_number(f_number):
    n = float(f_number)
    if not (math.isfinite(n) and n > 0.0):
        raise MoonHaloError("the f-number must be finite and positive")
    return n

def irradiance_to_sky_equivalent_conversion(optics_transmission, *, f_number):
    """C_irr(lambda) = 4 N^2 / (pi T_optics(lambda)) sr^-1, and 0 where T_optics(lambda) = 0: the on-axis paraxial conversion of
    a detector-plane irradiance to the uniform sky radiance that produces it through the same optics. No solid angle enters:
    C_pow = 1 / (T Omega_FOV), and the maximum of the two, have no role for Liu's illuminance-ratio PST."""
    t = np.asarray(optics_transmission, dtype=np.float64)
    t = _optics_transmission(t, t.shape)
    n = _f_number(f_number)
    out = np.zeros_like(t)
    np.divide(4.0 * n * n, math.pi * t, out=out, where=t > 0.0)
    return out

def moon_stray_light_v2_provider(wave_nm, *, solar_sed_w_m2_nm, pst_model, optics_transmission, f_number, rolo_table=None,
                                 reflectance_table=None, source_route=SOURCE_ROUTE_ROLO311G, rolo_out_of_domain_fallback=None,
                                 a7_phase_support_deg=A7_GREY_PHASE_SUPPORT_DEG, opposition_surge=None):
    """Version 2 (contract/sky_background_v2.md): a callable `(geometry) -> dict` for `sky_background.background_map(moon=...)`.

    Per frame, on the geometry `scene_geometry` makes (with `observer_gcrs_km`, and on the ROLO route `epoch_tdb_jd`):

    1. REFUSED (NaN at every node) when the Moon's centre is at or within 28 deg of the boresight at the frame or at any
       exposure sample the consumer supplies (`exposure_boresights_gcrs`, optionally `exposure_moon_directions_gcrs`), or when
       the geometry marks the Moon's disk in or near the field (the PST law is an out-of-field law).
    2. Zero (v1's no-direct-light statuses) when the Earth hides the whole disk or the umbra covers it (TOTAL before PARTIAL).
    3. NaN with the source's refusal status when the lunar source refuses (v1's rule, unchanged).
    4. Otherwise the uniform sky-equivalent radiance, at every node,
           L(lambda) = w PST(theta_b) E_M(lambda) 4 N^2 / (pi T_optics(lambda))     (0 where T_optics = 0)
       with w the area weight of a partial event (1 when the disk is lit whole), theta_b the frame's Moon-boresight separation,
       PST the Liu et al. 2021 law (0 at and beyond 90 deg) and T_optics the caller's optics transmission on `wave_nm` -- the
       SAME component product (t_lens x t_window x t_filter) the fold's response carries, so background_map's fold
       A Omega_p integral L R lambda/(hc) leaves A_pix integral w PST E_M QE lambda/(hc) on the axis. No solid angle is an
       input. The rate is ADDED to the sky; nothing is compared with an envelope.

    The metadata states the term's presence and size: the separations, the area weight and its method, the PST used and the
    law's parameters, E_M at 550 nm and its route (the A_7 grey fallback flagged), the detector irradiance and the added
    radiance at 550 nm, their ratio to the field's zodiacal minimum when background_map supplies it, the conversion, the
    spatial convention, the scenario token and the approximation tokens.
    """
    a7_support = _source_route_support(source_route, rolo_out_of_domain_fallback, a7_phase_support_deg)
    if not isinstance(pst_model, Liu2021ExponentialPST):
        raise MoonHaloError("the v2 mode's PST is the declared Liu2021ExponentialPST law")
    wave_grid = np.asarray(wave_nm, dtype=np.float64)
    t_optics = _optics_transmission(optics_transmission, wave_grid.shape)
    n_f = _f_number(f_number)
    c_irr = irradiance_to_sky_equivalent_conversion(t_optics, f_number=n_f)

    def route_tokens(route, surge=None):
        return v2_tokens(route=route, opposition_surge=surge)

    wave, i550, source = _lunar_source(wave_nm, solar_sed_w_m2_nm=solar_sed_w_m2_nm, rolo_table=rolo_table,
                                       reflectance_table=reflectance_table, source_route=source_route,
                                       rolo_out_of_domain_fallback=rolo_out_of_domain_fallback, a7_support=a7_support,
                                       opposition_surge=opposition_surge, route_tokens=route_tokens)
    conversion = {"rule": V2_CONVERSION_RULE, "f_number": n_f, "t_optics_550": float(t_optics[i550]),
                  "c_irr_550_per_sr": float(c_irr[i550])}

    def provider(geometry):
        rays = np.asarray(geometry["rays_gcrs"], dtype=np.float64)
        if rays.ndim != 2 or rays.shape[1] != 3:
            raise MoonHaloError("rays_gcrs must be (N, 3)")
        if geometry.get("observer_gcrs_km") is None:
            raise MoonHaloError("the v2 mode needs the observer's GCRS position (observer_gcrs_km); a geocentre is not assumed")
        n = rays.shape[0]
        moon_dir, observer, g, r_au = _moon_seen_from(geometry)
        exposure = _v2_exposure_separation(geometry, moon_dir)
        light = direct_light_area_weight(observer, moon_dir, geometry["moon_distance_km"], geometry["sun_direction_gcrs"],
                                         geometry["sun_distance_km"])
        e_moon, refl, route_meta = source(geometry, g, r_au)
        zl_min = geometry.get("zodiacal_min_radiance_550_w_m2_sr_nm")
        meta = {"schema": V2_SCHEMA, "mode": V2_MODE, "grade": GRADE, **route_meta,
                "phase_angle_at_moon_deg": g, "moon_sun_distance_au": r_au,
                "moon_boresight_separation_deg": exposure["frame_separation_deg"],
                "moon_centre_separation_min_over_exposure_deg": exposure["min_separation_deg"],
                "exposure_separation_samples": exposure["n_samples"],
                "refusal_angle_deg": V2_CENTRE_REFUSAL_DEG, "refusal_rule": "MOON_CENTRE_SEPARATION_LE_28_DEG_REFUSED",
                "direct_light": light["status"] or DIRECT_LIGHT, "direct_light_geometry": light["geometry"],
                "area_weight": light["area_weight"], "area_weight_method": light["method"],
                "covered_by_earth": light["covered_by_earth"], "covered_by_umbra": light["covered_by_umbra"],
                "e_moon_550_w_m2_nm": None if e_moon is None else float(e_moon[i550]),
                "reflectance_550": None if refl is None else float(refl[i550]),
                "e_moon_source": {"route": route_meta["source_route"], "status": route_meta["source_status"],
                                  "a7_grey_fallback_used": route_meta["source_status"] == SOURCE_STATUS_FALLBACK_A7,
                                  "phase_angle_at_moon_deg": g, "rolo_fitted_phase_domain_deg": list(ROLO_PHASE_DOMAIN_DEG)},
                "pst_model": pst_model.name, "pst_law": pst_model.parameters(), "pst_used": None,
                "pst_instrument_status": PST_INSTRUMENT_STATUS,
                "conversion": dict(conversion), "spatial_convention": V2_SPATIAL_CONVENTION, "scenario_token": V2_SCENARIO_TOKEN,
                "solar_stray_light": SOLAR_STRAY_LIGHT_V2,
                "e_detector_550_w_m2_nm": None, "added_radiance_550_w_m2_sr_nm": None, "term_present": False,
                "zodiacal_min_radiance_550_w_m2_sr_nm": None if zl_min is None else float(zl_min),
                "added_over_zodiacal_min": None}

        def uniform(value, status):
            meta["field_status"] = status
            return {"radiance": np.full((n, wave.size), value), "status": np.full(n, status, dtype=object), **meta}

        if exposure["refused"]:
            meta["refusal_reason"] = (f"the Moon's centre is {exposure['min_separation_deg']:.4f} deg from the boresight, at or within "
                                      f"{V2_CENTRE_REFUSAL_DEG:g} deg (closed), over {exposure['n_samples']} sample(s) of the exposure")
            return uniform(math.nan, STATUS_V2_REFUSED_CENTRE_WITHIN_28)
        in_field = np.asarray(geometry.get("moon_in_field", np.zeros(n, dtype=bool)), dtype=bool)
        if in_field.any() or geometry.get("moon_status") == STATUS_MOON_IN_FIELD_MASKED:
            meta["refusal_reason"] = "the geometry marks the lunar disk in or near the field; the PST law is an out-of-field law"
            return uniform(math.nan, STATUS_V2_REFUSED_DISK_IN_OR_NEAR_FIELD)
        if light["method"] == AREA_WEIGHT_TOTAL_EVENT:
            # wholly hidden or wholly in the umbra: no direct light, a true zero whatever the source says
            meta.update(e_detector_550_w_m2_nm=0.0, added_radiance_550_w_m2_sr_nm=0.0)
            return uniform(0.0, light["status"])
        if e_moon is None:
            # the source refused: the refusal propagates to every node, nothing is clamped, extrapolated or zeroed
            return uniform(math.nan, meta["source_status"])
        pst = pst_model(exposure["frame_separation_deg"])
        e_detector = light["area_weight"] * pst * np.asarray(e_moon, dtype=np.float64)
        row = e_detector * c_irr
        meta.update(pst_used=pst, e_detector_550_w_m2_nm=float(e_detector[i550]), added_radiance_550_w_m2_sr_nm=float(row[i550]),
                    term_present=bool(np.any(row > 0.0)))
        if zl_min is not None and float(zl_min) > 0.0:
            meta["added_over_zodiacal_min"] = float(row[i550]) / float(zl_min)
        else:
            meta["not_compared_reason"] = "no zodiacal minimum was supplied to compare the added radiance against"
        status = STATUS_V2_PST_ZERO_AT_OR_BEYOND_90 if pst == 0.0 else (light["status"] or STATUS_V2_ADDED)
        meta["field_status"] = status
        return {"radiance": np.tile(row, (n, 1)), "status": np.full(n, status, dtype=object), **meta}

    return provider
