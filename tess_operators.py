import math
import numpy as np

CLIP_SIGMA, CLIP_ITERS = 3.0, 5
ORBIT_GAP_DAYS = 1.0
BTJD_OFFSET_JD = 2457000.0
SATURATION_T, SATURATION_MASK_PX = 7.0, 1.5
PRF_GRID_REV2 = [
    (0.55, 0.05, 1.5),
    (0.55, 0.05, 2.5),
    (0.55, 0.05, 3.5),
    (0.55, 0.15, 1.5),
    (0.55, 0.15, 2.5),
    (0.55, 0.15, 3.5),
    (0.55, 0.25, 1.5),
    (0.55, 0.25, 2.5),
    (0.55, 0.25, 3.5),
    (0.55, 0.35, 1.5),
    (0.55, 0.35, 2.5),
    (0.55, 0.35, 3.5),
    (0.65, 0.05, 1.5),
    (0.65, 0.05, 2.5),
    (0.65, 0.05, 3.5),
    (0.65, 0.15, 1.5),
    (0.65, 0.15, 2.5),
    (0.65, 0.15, 3.5),
    (0.65, 0.25, 1.5),
    (0.65, 0.25, 2.5),
    (0.65, 0.25, 3.5),
    (0.65, 0.35, 1.5),
    (0.65, 0.35, 2.5),
    (0.65, 0.35, 3.5),
    (0.75, 0.05, 1.5),
    (0.75, 0.05, 2.5),
    (0.75, 0.05, 3.5),
    (0.75, 0.15, 1.5),
    (0.75, 0.15, 2.5),
    (0.75, 0.15, 3.5),
    (0.75, 0.25, 1.5),
    (0.75, 0.25, 2.5),
    (0.75, 0.25, 3.5),
    (0.75, 0.35, 1.5),
    (0.75, 0.35, 2.5),
    (0.75, 0.35, 3.5),
    (0.85, 0.05, 1.5),
    (0.85, 0.05, 2.5),
    (0.85, 0.05, 3.5),
    (0.85, 0.15, 1.5),
    (0.85, 0.15, 2.5),
    (0.85, 0.15, 3.5),
    (0.85, 0.25, 1.5),
    (0.85, 0.25, 2.5),
    (0.85, 0.25, 3.5),
    (0.85, 0.35, 1.5),
    (0.85, 0.35, 2.5),
    (0.85, 0.35, 3.5),
    (0.95, 0.05, 1.5),
    (0.95, 0.05, 2.5),
    (0.95, 0.05, 3.5),
    (0.95, 0.15, 1.5),
    (0.95, 0.15, 2.5),
    (0.95, 0.15, 3.5),
    (0.95, 0.25, 1.5),
    (0.95, 0.25, 2.5),
    (0.95, 0.25, 3.5),
    (0.95, 0.35, 1.5),
    (0.95, 0.35, 2.5),
    (0.95, 0.35, 3.5),
]

from scipy.ndimage import map_coordinates, gaussian_filter


def unit(ra, dec):
    a, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], axis=-1)


def omega_at(wcs, x, y):
    def v(a, b):
        return unit(*wcs.all_pix2world(a, b, 0))

    dx = v(x + 0.5, y) - v(x - 0.5, y)
    dy = v(x, y + 0.5) - v(x, y - 0.5)
    return np.linalg.norm(np.cross(dx, dy), axis=-1)


def omega_grid(wcs, xs, ys):
    """Pixel solid angle on an arbitrary (ny, nx) grid of source pixel coordinates, sampled every ~32 px and interpolated."""
    ny, nx = xs.shape
    sy = np.rint(np.linspace(0, ny - 1, max(2, (ny + 31) // 32))).astype(int)
    sx = np.rint(np.linspace(0, nx - 1, max(2, (nx + 31) // 32))).astype(int)
    a = omega_at(wcs, xs[np.ix_(sy, sx)], ys[np.ix_(sy, sx)])
    gy, gx = np.mgrid[:ny, :nx]
    return map_coordinates(
        a, [gy / (ny - 1) * (len(sy) - 1), gx / (nx - 1) * (len(sx) - 1)], order=1, mode="nearest"
    )


def smooth_valid(a, valid, sigma):
    weight = gaussian_filter(valid.astype(float), sigma, mode="constant", cval=0)
    num = gaussian_filter(np.where(valid, a, 0), sigma, mode="constant", cval=0)
    return np.divide(num, weight, out=np.full_like(num, np.nan), where=weight > 0.05), weight


def cutout_background(flux):
    """Per cadence: the sigma-clipped median and the 20th percentile of the finite pixels, and the finite-pixel count."""
    import warnings
    from astropy.stats import sigma_clipped_stats

    f = np.asarray(flux, dtype=np.float64)
    flat = f.reshape(f.shape[0], -1)
    finite = np.isfinite(flat).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, med, _ = sigma_clipped_stats(flat, sigma=CLIP_SIGMA, maxiters=CLIP_ITERS, axis=1)
        p20 = np.nanpercentile(np.where(finite[:, None] > 0, flat, 0.0), 20, axis=1)
    med = np.where(finite > 0, np.asarray(med, dtype=np.float64), math.nan)
    p20 = np.where(finite > 0, p20, math.nan)
    return med, p20, finite


def running_median(t, y, window_days):
    out = np.empty_like(y)
    half = window_days / 2.0
    lo = 0
    hi = 0
    n = t.size
    for i in range(n):
        while t[lo] < t[i] - half:
            lo += 1
        while hi < n and t[hi] <= t[i] + half:
            hi += 1
        out[i] = np.median(y[lo:hi])
    return out


def split_orbits(t, gap_days=ORBIT_GAP_DAYS):
    if t.size == 0:
        return []
    breaks = np.where(np.diff(t) > gap_days)[0]
    edges = np.concatenate([[0], breaks + 1, [t.size]])
    return [np.arange(edges[k], edges[k + 1]) for k in range(edges.size - 1)]


def sun_directions(time_btjd):
    from astropy.coordinates import get_sun
    from astropy.time import Time

    t = Time(np.asarray(time_btjd, dtype=np.float64) + BTJD_OFFSET_JD, format="jd", scale="tdb")
    xyz = np.asarray(get_sun(t).cartesian.xyz.value, dtype=np.float64).T
    return xyz / np.linalg.norm(xyz, axis=1)[:, None]


def t_mag(g, bp, rp):
    c = bp - rp
    poly = g - 0.00522555 * c**3 + 0.0891337 * c**2 - 0.633923 * c + 0.0324473
    ok = np.isfinite(c) & (c >= -0.2) & (c <= 3.0)
    return np.where(ok, poly, g - 0.430)


def pixel_gauss(centre, n, sigma):
    from scipy.special import erf

    edges = np.arange(n + 1) - 0.5
    z = (edges[None, :] - centre[:, None]) / (math.sqrt(2.0) * sigma)
    cdf = 0.5 * (1.0 + erf(z))
    return np.diff(cdf, axis=1)  # (stars, n)


def star_image_halo(x, y, flux, ny, nx, sigma_core, f_halo, sigma_halo):
    img = np.zeros((ny, nx))
    for sigma, w in ((sigma_core, 1.0 - f_halo), (sigma_halo, f_halo)):
        gx = pixel_gauss(x, nx, sigma)
        gy = pixel_gauss(y, ny, sigma)
        img += w * np.einsum("s,sy,sx->yx", flux, gy, gx)
    return img


def fit_background(img, x, y, flux, tmag):
    """SPEC_REV2.md section 3.3 revision 2: the 60-point PRF family, weights that also down-weight star-dominated pixels."""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    mask = np.isfinite(img)
    for xs, ys in zip(x[tmag < SATURATION_T], y[tmag < SATURATION_T]):
        mask &= (xx - xs) ** 2 + (yy - ys) ** 2 > SATURATION_MASK_PX**2
    b0 = max(float(np.percentile(img[mask], 10)), 1.0)
    out = []
    for sc, f, sh in PRF_GRID_REV2:
        s_img = star_image_halo(x, y, flux, ny, nx, sc, f, sh)
        wts = 1.0 / (np.abs(img) + 1.0) / (1.0 + (s_img / b0) ** 2)
        A = np.stack([np.ones(mask.sum()), s_img[mask]], axis=1)
        sw = np.sqrt(wts[mask])
        coef, *_ = np.linalg.lstsq(A * sw[:, None], img[mask] * sw, rcond=None)
        resid = img[mask] - A @ coef
        out.append(
            {
                "sigma_core": sc,
                "f_halo": f,
                "sigma_halo": sh,
                "B": float(coef[0]),
                "s": float(coef[1]),
                "chi2": float(np.sum(wts[mask] * resid**2)),
                "star_light_mean": float(np.mean(coef[1] * s_img[mask])),
            }
        )
    best = min(out, key=lambda r: r["chi2"])
    return best, out, int(mask.sum())
