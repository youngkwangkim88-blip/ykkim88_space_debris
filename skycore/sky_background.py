"""`synth-sky-background/v1`: direction-dependent diffuse sky for the declared camera."""

from __future__ import annotations

import math

from datetime import datetime, timezone

from pathlib import Path

import numpy as np

from . import radiometry as SR

ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi

ARCSEC2_PER_SR = ARCSEC_PER_RAD ** 2

OBLIQUITY_J2000_ARCSEC = 84381.406

GALACTIC_POLE_RA_DEG = 192.85948

GALACTIC_POLE_DEC_DEG = 27.12825

GALACTIC_L_NCP_DEG = 122.93192

MOON_RADIUS_KM = 1737.4

STATUS_ZL_OK = "ZL_OK"

STATUS_ZL_UNSUPPORTED_NEAR_SUN = "ZL_UNSUPPORTED_NEAR_SUN"

STATUS_ZL_OUT_OF_DOMAIN = "ZL_OUT_OF_DOMAIN"

STATUS_MOON_DISK_OCCULTS_DIFFUSE_SKY = "MOON_DISK_OCCULTS_DIFFUSE_SKY_SURFACE_NOT_RENDERED"

STATUS_MOON_DISK_CLEAR = "MOON_DISK_CLEAR"

class SkyBackgroundError(ValueError):
    """A declaration or input the sky model refuses; never a silent default."""

    def __init__(self, message, *, key=None):
        super().__init__(message)
        self.key = key

def _rotation_x(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])

def ecliptic_rotation_j2000():
    """GCRS (ICRS-aligned) to mean ecliptic J2000: rotate about x by the obliquity."""
    return _rotation_x(math.radians(OBLIQUITY_J2000_ARCSEC / 3600.0))

def galactic_rotation():
    """ICRS to Galactic: the IAU 1958 system in the ICRS realisation, built from the pole and l_NCP.

    Rows are the Galactic x, y, z axes expressed in ICRS: z is the north Galactic pole; x points
    to the Galactic centre direction, which lies 90 degrees from the pole at position angle set by
    `l_NCP` (the Galactic longitude of the north celestial pole); y completes the right-handed set.
    The owned test compares every element with astropy's Galactic frame.
    """
    ra_p, dec_p = math.radians(GALACTIC_POLE_RA_DEG), math.radians(GALACTIC_POLE_DEC_DEG)
    z = np.array([math.cos(dec_p) * math.cos(ra_p), math.cos(dec_p) * math.sin(ra_p), math.sin(dec_p)])
    ncp = np.array([0.0, 0.0, 1.0])
    # the Galactic-plane direction toward the NCP's projection has longitude l_NCP; x (l = 0) is
    # that direction rotated about z by -l_NCP
    east = np.cross(z, ncp)                     # direction of increasing l at the NCP's foot point
    east /= np.linalg.norm(east)
    toward_ncp = np.cross(east, z)              # unit vector in the plane pointing at the NCP's foot point
    l_ncp = math.radians(GALACTIC_L_NCP_DEG)
    x = math.cos(l_ncp) * toward_ncp - math.sin(l_ncp) * east
    y = np.cross(z, x)
    return np.stack([x, y, z])

def lonlat_from_vectors(vectors):
    """Longitude in [0, 2pi) and latitude in [-pi/2, pi/2] of unit vectors (N, 3)."""
    v = np.asarray(vectors, dtype=np.float64)
    lon = np.arctan2(v[..., 1], v[..., 0]) % (2.0 * math.pi)
    lat = np.arcsin(np.clip(v[..., 2], -1.0, 1.0))
    return lon, lat

def galactic_from_gcrs(vectors):
    return lonlat_from_vectors(np.asarray(vectors) @ galactic_rotation().T)

def helioecliptic(rays_gcrs, sun_direction_gcrs):
    """Sun-relative ecliptic longitude difference in [0, pi] and ecliptic latitude, per ray.

    `delta_lambda` is measured between the ECLIPTIC-PLANE PROJECTIONS of the ray and the observer's
    Sun direction; it is not the elongation. At an ecliptic pole the projection vanishes and the
    longitude is undefined: those rays get `delta_lambda = pi/2` as a placeholder, and the table
    lookup uses the pole row value, which does not depend on longitude.
    """
    rot = ecliptic_rotation_j2000()
    r = np.asarray(rays_gcrs, dtype=np.float64) @ rot.T
    s = np.asarray(sun_direction_gcrs, dtype=np.float64) @ rot.T
    s = s / np.linalg.norm(s)
    beta = np.arcsin(np.clip(r[..., 2], -1.0, 1.0))
    r_proj = r[..., :2]
    s_proj = s[:2] / max(np.linalg.norm(s[:2]), 1e-300)
    norm = np.linalg.norm(r_proj, axis=-1)
    at_pole = norm < 1e-12
    safe = np.where(at_pole, 1.0, norm)
    cosd = np.clip((r_proj @ s_proj) / safe, -1.0, 1.0)
    delta = np.where(at_pole, math.pi / 2.0, np.arccos(cosd))
    return delta, beta, at_pole

def angular_separation(a, b):
    """Angle between unit vectors, numerically safe near 0 and pi."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.arctan2(cross, dot)

def utc_epoch(epoch):
    """Require a caller-supplied instant and normalize its explicit offset to UTC.

    No wall clock, machine-local timezone, date-only midnight or fixed epoch is
    inferred. This is the shared public input boundary for scalar and batched
    ephemerides and their scene consumers. Time-scale conversion remains Astropy's.
    """
    if epoch is None or (isinstance(epoch, str) and not epoch.strip()):
        raise SkyBackgroundError("the epoch is required; supply a timestamp with Z or an explicit offset", key="epoch")
    if isinstance(epoch, str):
        text = epoch[:-1] + "+00:00" if epoch.endswith("Z") else epoch
        try:
            epoch = datetime.fromisoformat(text)
        except ValueError as err:
            raise SkyBackgroundError("the epoch must be an ISO 8601 timestamp with an explicit timezone", key="epoch") from err
    if not isinstance(epoch, datetime):
        raise SkyBackgroundError(f"unsupported epoch {epoch!r}; supply a timezone-aware timestamp", key="epoch")
    if epoch.utcoffset() is None:
        raise SkyBackgroundError("the epoch must carry a timezone (UTC); a date-only or naive timestamp is refused", key="epoch")
    return epoch.astimezone(timezone.utc)

_utc = utc_epoch

def sun_gcrs_km(epoch):
    epoch = utc_epoch(epoch)
    from astropy.coordinates import get_sun
    from astropy.time import Time
    t = Time(epoch.strftime("%Y-%m-%dT%H:%M:%S.%f"), scale="utc")
    return np.asarray(get_sun(t).cartesian.xyz.to("km").value, dtype=np.float64)

def moon_gcrs_km(epoch):
    epoch = utc_epoch(epoch)
    from astropy.coordinates import get_body
    from astropy.time import Time
    t = Time(epoch.strftime("%Y-%m-%dT%H:%M:%S.%f"), scale="utc")
    return np.asarray(get_body("moon", t, ephemeris="builtin").cartesian.xyz.to("km").value, dtype=np.float64)

def epoch_tdb_jd(epoch):
    """The epoch as a TDB Julian date (the argument of the IAU lunar rotation model used by the Moon term)."""
    epoch = utc_epoch(epoch)
    from astropy.time import Time
    t = Time(epoch.strftime("%Y-%m-%dT%H:%M:%S.%f"), scale="utc").tdb
    return float(t.jd1) + float(t.jd2)

def observer_relative_direction(body_gcrs_km, observer_gcrs_km):
    d = np.asarray(body_gcrs_km, dtype=np.float64) - np.asarray(observer_gcrs_km, dtype=np.float64)
    return d / np.linalg.norm(d), float(np.linalg.norm(d))

def disk_mask(rays_gcrs, body_direction, body_distance_km, body_radius_km, *, margin_deg=0.0):
    """Rays inside a body's apparent disk (plus a margin), and its apparent angular radius."""
    angular_radius = math.asin(min(1.0, body_radius_km / body_distance_km))
    sep = angular_separation(np.asarray(rays_gcrs, dtype=np.float64), np.asarray(body_direction, dtype=np.float64))
    return sep < angular_radius + math.radians(margin_deg), angular_radius, sep

class ZodiacalTable:
    """A published `mu_V(delta_lambda, |beta|)` table, interpolated bilinearly in LINEAR radiance.

    CSV layout is the PUBLISHED orientation: comment lines start with `#`; the header row is
    `delta_lambda_deg,<beta values...>` (the column latitudes); each following row starts with its
    Sun-relative longitude and holds one cell per latitude: a magnitude per arcsec^2, or the
    literal `SE` for the source's near-Sun cells, which this model refuses rather than clamps.
    Internally the table is stored as `mu_v[beta_index, delta_lambda_index]`. Off-knot values are
    this project's linear-radiance interpolation, not the source's own magnitude-space
    interpolation, and are not claimed to match it.
    """

    def __init__(self, delta_lambda_deg, beta_deg, mu_v, *, source, sha256=None):
        self.delta_lambda_deg = np.asarray(delta_lambda_deg, dtype=np.float64)
        self.beta_deg = np.asarray(beta_deg, dtype=np.float64)
        self.mu_v = np.asarray(mu_v, dtype=np.float64)          # NaN marks SE / unsupported cells
        self.source = source
        self.sha256 = sha256
        if self.mu_v.shape != (self.beta_deg.size, self.delta_lambda_deg.size):
            raise SkyBackgroundError("zodiacal table shape does not match its axes", key="zodiacal_table")
        if np.any(np.diff(self.delta_lambda_deg) <= 0.0) or np.any(np.diff(self.beta_deg) <= 0.0):
            raise SkyBackgroundError("zodiacal table axes must be strictly increasing", key="zodiacal_table")

    @classmethod
    def load_csv(cls, path, *, sha256=None):
        path = Path(path)
        if not path.exists():
            raise SkyBackgroundError(f"zodiacal table {path} is absent", key="zodiacal_table")
        if sha256 is not None:
            digest = SR.sha256_of(path)
            if digest != sha256:
                raise SkyBackgroundError(f"zodiacal table {path.name} has sha256 {digest}, declaration pins {sha256}",
                                         key="zodiacal_table")
        rows = [line.strip().split(",") for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        header = rows[0]
        betas = np.array([float(x) for x in header[1:]])
        dls, values = [], []
        for row in rows[1:]:
            dls.append(float(row[0]))
            values.append([math.nan if cell.strip().upper() in ("SE", "SA", "NA", "", "NAN") else float(cell) for cell in row[1:]])
        # published orientation (rows = delta_lambda, columns = beta) -> internal (beta rows, delta_lambda columns)
        return cls(np.array(dls), betas, np.array(values, dtype=np.float64).T, source=str(path.name), sha256=sha256 or SR.sha256_of(path))

    def lookup(self, delta_lambda_deg, abs_beta_deg, at_pole=None):
        """`mu_V` per ray (NaN where unsupported) and a status per ray."""
        dl = np.asarray(delta_lambda_deg, dtype=np.float64)
        b = np.abs(np.asarray(abs_beta_deg, dtype=np.float64))
        n = dl.size
        out = np.full(n, math.nan)
        status = np.full(n, STATUS_ZL_OK, dtype=object)
        # domain
        lo_dl, hi_dl = self.delta_lambda_deg[0], self.delta_lambda_deg[-1]
        lo_b, hi_b = self.beta_deg[0], self.beta_deg[-1]
        outside = (dl < lo_dl - 1e-9) | (dl > hi_dl + 1e-9) | (b < lo_b - 1e-9) | (b > hi_b + 1e-9)
        status[outside] = STATUS_ZL_OUT_OF_DOMAIN
        dlc = np.clip(dl, lo_dl, hi_dl)
        bc = np.clip(b, lo_b, hi_b)
        i = np.clip(np.searchsorted(self.delta_lambda_deg, dlc, side="right") - 1, 0, self.delta_lambda_deg.size - 2)
        j = np.clip(np.searchsorted(self.beta_deg, bc, side="right") - 1, 0, self.beta_deg.size - 2)
        x = (dlc - self.delta_lambda_deg[i]) / (self.delta_lambda_deg[i + 1] - self.delta_lambda_deg[i])
        y = (bc - self.beta_deg[j]) / (self.beta_deg[j + 1] - self.beta_deg[j])
        # the four corner knots, in linear radiance (relative units 10^(-0.4 mu)); NaN marks SE
        corners = np.stack([self.mu_v[j, i], self.mu_v[j, i + 1], self.mu_v[j + 1, i], self.mu_v[j + 1, i + 1]], axis=-1)
        weights = np.stack([(1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y], axis=-1)
        finite = np.isfinite(corners)
        # an SE corner is refused only when it carries weight; a knot exactly on the cell edge away
        # from the SE corner keeps its value
        unsupported = ((~finite) & (weights > 1e-12)).any(axis=-1) & ~outside
        status[unsupported] = STATUS_ZL_UNSUPPORTED_NEAR_SUN
        with np.errstate(invalid="ignore"):
            lin = np.where(finite, 10.0 ** (-0.4 * np.where(finite, corners, 0.0)), 0.0)
            lin_interp = np.sum(lin * weights, axis=-1)
        ok = status == STATUS_ZL_OK
        out[ok] = -2.5 * np.log10(lin_interp[ok])
        if at_pole is not None:
            pole = np.asarray(at_pole, dtype=bool)
            if pole.any():
                pole_row = self.mu_v[-1, :]
                if not np.isfinite(pole_row).all() or np.ptp(pole_row) > 1e-9:
                    raise SkyBackgroundError("the table's last |beta| row must be the ecliptic pole with one longitude-independent value",
                                             key="zodiacal_table")
                out[pole] = float(pole_row[0])
                status[pole] = STATUS_ZL_OK
        return out, status

def radiance_from_solar_surface_brightness(mu_v_mag_arcsec2, wave_nm, solar_sed_w_m2_nm, m_sun_v, colour_factor=None):
    """Spectral radiance (W m^-2 sr^-1 nm^-1) of a solar-coloured surface of V brightness `mu_V`.

    Convention: a surface brightness `mu_V` (Vega/Johnson V magnitudes per arcsec^2) of a source with
    the Sun's spectrum is the solar irradiance spectrum scaled by `10^(-0.4 (mu_V - m_sun_V))` per
    arcsec^2, with `m_sun_V` the Sun's V magnitude in the same system (the capsule's anchor-implied
    -26.762575; the literature Vega-system value is -26.76). This is exact for a solar spectrum and
    needs no V passband; the optional colour factor (normalised to 1 at 550 nm) reddens the zodiacal
    light relative to the Sun as the source prescribes, and the small change it makes to the V-band
    integral is declared as an approximation rather than re-normalised.
    """
    scale = 10.0 ** (-0.4 * (np.asarray(mu_v_mag_arcsec2, dtype=np.float64) - m_sun_v)) * ARCSEC2_PER_SR
    sed = np.asarray(solar_sed_w_m2_nm, dtype=np.float64)[None, :]
    if colour_factor is not None:
        cf = np.asarray(colour_factor, dtype=np.float64)
        sed = sed * (cf[None, :] if cf.ndim == 1 else cf)          # one colour for all, or one colour per surface
    return np.asarray(scale)[..., None] * sed

def _sky_directions(rays, observer_gcrs_km, epoch):
    """The Sun and the Moon made observer-relative, and each ray's helioecliptic and Galactic coordinates (radians), as
    `scene_geometry` has always computed them: the SKY geometry's one implementation, which `sky_geometry` runs too.
    Returns (observer, sun_dir, sun_dist, moon_dir, moon_dist, delta, beta, at_pole, gal_l, gal_b)."""
    observer = np.asarray(observer_gcrs_km, dtype=np.float64)
    sun_dir, sun_dist = observer_relative_direction(sun_gcrs_km(epoch), observer)
    moon_dir, moon_dist = observer_relative_direction(moon_gcrs_km(epoch), observer)
    delta, beta, at_pole = helioecliptic(rays, sun_dir)
    gal_l, gal_b = galactic_from_gcrs(rays)
    return observer, sun_dir, sun_dist, moon_dir, moon_dist, delta, beta, at_pole, gal_l, gal_b

def _epoch_keys(epoch):
    """The epoch a geometry carries for the lunar orientation (ROLO 311g libration angles): its UTC text and TDB Julian date."""
    return {"epoch_utc": _utc(epoch).isoformat().replace("+00:00", "Z"), "epoch_tdb_jd": epoch_tdb_jd(epoch)}

def sky_geometry(rays_gcrs, *, observer_gcrs_km, epoch):
    """`physics-sky-radiance/v1`: the SKY geometry of GCRS unit rays seen from one observer at one epoch.

    The Sun and the Moon from the built-in ephemeris made observer-relative, each ray's helioecliptic coordinates
    (`delta_lambda`, `beta`, the pole flag) and Galactic coordinates, the epoch the lunar orientation needs, and the lunar disk
    per ray WITHOUT a margin: `disk_mask` at zero margin gives the separation from the Moon's centre and the apparent radius
    (radians), the inside flag and its status. No camera intrinsics, no margin and no field verdict: the Earth-limb,
    Sun-avoidance and Moon-mask rules are a consumer's (`scene_geometry` keeps the camera's). `scene_geometry` computes its
    sky keys through the same statements (`_sky_directions`, `_epoch_keys`), so the two cannot differ.
    """
    rays = np.asarray(rays_gcrs, dtype=np.float64)
    observer, sun_dir, sun_dist, moon_dir, moon_dist, delta, beta, at_pole, gal_l, gal_b = _sky_directions(rays, observer_gcrs_km, epoch)
    in_disk, moon_radius, moon_sep = disk_mask(rays, moon_dir, moon_dist, MOON_RADIUS_KM)
    return {
        "rays_gcrs": rays, "delta_lambda_deg": np.degrees(delta), "beta_deg": np.degrees(beta), "at_ecliptic_pole": at_pole,
        "gal_l_deg": np.degrees(gal_l), "gal_b_deg": np.degrees(gal_b),
        "sun_direction_gcrs": sun_dir, "sun_distance_km": sun_dist, "moon_direction_gcrs": moon_dir,
        "moon_distance_km": moon_dist, "moon_angular_radius_deg": math.degrees(moon_radius), "observer_gcrs_km": observer,
        **_epoch_keys(epoch),
        "moon_separation_rad": moon_sep, "moon_angular_radius_rad": moon_radius, "in_moon_disk": in_disk,
        "moon_disk_ray_status": np.where(in_disk, STATUS_MOON_DISK_OCCULTS_DIFFUSE_SKY, STATUS_MOON_DISK_CLEAR).astype(object),
    }
