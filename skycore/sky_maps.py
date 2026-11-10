"""Sky maps for the Galactic components of `synth-sky-background/v1`: SFD (Lambert ZEA) and HEALPix readers,"""

from __future__ import annotations

import math

import os

import re

from pathlib import Path

import numpy as np

from . import radiometry as SR

NU_I_NU_100UM_NW_M2_SR_PER_MJY_SR = 29.9792458      # nu = c / 100 um; 1 MJy/sr = 1e-20 W m^-2 Hz^-1 sr^-1

def healpix_pix2ang_ring(nside, pix):
    """Centre colatitude and longitude of RING pixels (the inverse of `healpix_ang2pix_ring`, for tests)."""
    nside = int(nside)
    pix = np.asarray(pix, dtype=np.int64)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    theta = np.empty(pix.shape, dtype=np.float64)
    phi = np.empty(pix.shape, dtype=np.float64)
    north = pix < ncap
    # north cap
    p = pix[north]
    # the ring from the quadratic: iring = floor((1 + sqrt(1 + 2 p)) / 2)
    iring = ((1 + np.sqrt(1.0 + 2.0 * p)) / 2.0).astype(np.int64)
    iphi = p - 2 * iring * (iring - 1)
    theta[north] = np.arccos(1.0 - iring ** 2 / (3.0 * nside ** 2))
    phi[north] = (iphi + 0.5) * math.pi / (2.0 * iring)
    # equatorial belt
    belt = (pix >= ncap) & (pix < npix - ncap)
    p = pix[belt] - ncap
    iring = p // (4 * nside) + nside
    iphi = p % (4 * nside)
    # rings whose (ring + nside) is even have their first centre half a spacing in; odd rings start at phi = 0
    # (checked against astropy-healpix 2.0.1 for every belt ring at nside 16 and at random nside 1024 pixels)
    fodd = np.where(((iring + nside) & 1) == 0, 0.5, 0.0)
    theta[belt] = np.arccos((2 * nside - iring) * 2.0 / (3.0 * nside))
    phi[belt] = (iphi + fodd) * math.pi / (2.0 * nside)
    # south cap
    south = pix >= npix - ncap
    p = npix - 1 - pix[south]
    iring = ((1 + np.sqrt(1.0 + 2.0 * p)) / 2.0).astype(np.int64)
    iphi = 4 * iring - 1 - (p - 2 * iring * (iring - 1))
    theta[south] = np.arccos(-1.0 + iring ** 2 / (3.0 * nside ** 2))
    phi[south] = (iphi + 0.5) * math.pi / (2.0 * iring)
    return theta, phi

def _spread_bits(v):
    """Interleave zeros between the bits of `v` (int64 array), the HEALPix xyf -> nested bit spreading."""
    v = np.asarray(v, dtype=np.int64)
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v

def healpix_ang2pix_nest(nside, theta_rad, phi_rad):
    """NESTED-scheme pixel index of colatitude `theta` and longitude `phi` (the HEALPix C++ `loc2pix` nested
    branch: face number from the (jp, jm) edge-line indices, then bit-interleaved (ix, iy) within the face).
    `nside` must be a power of two. Checked against astropy-healpix 2.0.1 at nside 32 in the owned tests."""
    nside = int(nside)
    order = nside.bit_length() - 1
    if 1 << order != nside:
        raise SR.RadiometryError("nested HEALPix needs a power-of-two nside", key="healpix")
    theta = np.asarray(theta_rad, dtype=np.float64)
    phi = np.asarray(phi_rad, dtype=np.float64) % (2.0 * math.pi)
    z = np.cos(theta)
    za = np.abs(z)
    tt = phi / (math.pi / 2.0)          # in [0, 4)
    ix = np.empty(theta.shape, dtype=np.int64)
    iy = np.empty(theta.shape, dtype=np.int64)
    face = np.empty(theta.shape, dtype=np.int64)
    eq = za <= 2.0 / 3.0
    temp1 = nside * (0.5 + tt[eq])
    temp2 = nside * z[eq] * 0.75
    jp = np.floor(temp1 - temp2).astype(np.int64)
    jm = np.floor(temp1 + temp2).astype(np.int64)
    ifp = jp >> order
    ifm = jm >> order
    face[eq] = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    ix[eq] = jm & (nside - 1)
    iy[eq] = nside - (jp & (nside - 1)) - 1
    pc = ~eq
    ntt = np.minimum(np.floor(tt[pc]).astype(np.int64), 3)
    tp = tt[pc] - ntt
    tmp = nside * np.sqrt(3.0 * (1.0 - za[pc]))
    jp = np.minimum(np.floor(tp * tmp).astype(np.int64), nside - 1)
    jm = np.minimum(np.floor((1.0 - tp) * tmp).astype(np.int64), nside - 1)
    north = z[pc] >= 0
    face[pc] = np.where(north, ntt, ntt + 8)
    ix[pc] = np.where(north, nside - jm - 1, jp)
    iy[pc] = np.where(north, nside - jp - 1, jm)
    return face * (nside * nside) + _spread_bits(ix) + (_spread_bits(iy) << 1)

def dgl_north_weight(b_deg, *, half_width_deg):
    """The weight of the north hemisphere's C alpha' at Galactic latitude `b_deg`: 1 at b >= half_width_deg, 0 at
    b <= -half_width_deg, and between them the quintic smoothstep w = t^3 (10 - 15 t + 6 t^2), t = (b + h) / (2 h), which
    rises monotonically, is 1/2 at the equator and has zero first and second derivatives at both ends (twice continuously
    differentiable). A half width of 0 is the sign rule: 1 for b >= 0, else 0. The ends are exact, never a rounded blend."""
    h = float(half_width_deg)
    if not (math.isfinite(h) and h >= 0.0):
        raise SR.RadiometryError(f"the hemisphere transition half width must be finite and >= 0, got {half_width_deg!r}",
                                 key="hemisphere_transition")
    b = np.asarray(b_deg, dtype=np.float64)
    if h == 0.0:
        return np.where(b >= 0.0, 1.0, 0.0)
    t = np.clip((b + h) / (2.0 * h), 0.0, 1.0)
    # the polynomial rounds a few units of 1e-16 above 1 just inside the north end; the clip keeps the weight, and so C and
    # C alpha', monotone to the last bit instead of overshooting the north value and stepping back at b = h
    w = np.clip(t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t), 0.0, 1.0)
    return np.where(b >= h, 1.0, np.where(b <= -h, 0.0, w))

CELL_SAMPLING_SMOOTH_RBF = "smooth_rbf"

CELL_SAMPLING_NEAREST = "nearest"

CELL_SAMPLING_SMOOTH_RBF_MEAN_PRESERVING = "smooth_rbf_mean_preserving"

CELL_SAMPLING_METHODS = (CELL_SAMPLING_SMOOTH_RBF, CELL_SAMPLING_NEAREST, CELL_SAMPLING_SMOOTH_RBF_MEAN_PRESERVING)

MEAN_PRESERVING_SUB_LEVELS = 3

MEAN_PRESERVING_NODE_SCHEMA = "physics-mean-preserving-nodes/v1"

RBF_SUPPORT_SPACINGS = 4.0

_CELL_REACH_SPACINGS = 1.2

_RBF_CHUNK = 1 << 15                                # queries per vectorised pass: bounds the temporaries near 100 MB

RBF_EVALUATOR_FAST = "fast"

RBF_EVALUATOR_REFERENCE = "reference"

RBF_EVALUATORS = (RBF_EVALUATOR_FAST, RBF_EVALUATOR_REFERENCE)

_FAST_KERNEL = None

UNRESOLVED_COLOUR_TABLE_SCHEMA = "physics-unresolved-colour-factors/v1"

UNRESOLVED_COLOUR_BANDS = ("E5", "E12", "TESS")

UNRESOLVED_COLOUR_COVERAGE = ("INTERIOR", "EDGE", "OUTSIDE")

def _fast_kernel():
    """The compiled evaluation loop of the smooth laws, built on first use. numba is a declared dependency; the import is
    deferred so that a consumer that never reads a smooth law does not pay for it."""
    global _FAST_KERNEL
    if _FAST_KERNEL is None:
        import numba

        @numba.njit(cache=False)
        def kernel(qx, qy, qz, cell, cand_xyz, cand_coef, mean, radius, out):
            width = cand_xyz.shape[1]
            for p in range(qx.size):
                c = cell[p]
                s = 0.0
                for k in range(width):
                    d = qx[p] * cand_xyz[c, k, 0] + qy[p] * cand_xyz[c, k, 1] + qz[p] * cand_xyz[c, k, 2]
                    t = 2.0 - 2.0 * d
                    if t < 0.0:
                        t = 0.0
                    r = math.sqrt(t) / radius
                    if r >= 1.0:
                        continue
                    u = 1.0 - r
                    u2 = u * u
                    s += u2 * u2 * u2 * (35.0 * r * r + 18.0 * r + 3.0) * cand_coef[c, k]
                out[p] = mean + s

        _FAST_KERNEL = kernel
    return _FAST_KERNEL

def wendland_c4(r):
    """Wendland's psi_{3,2}(r) = (1 - r)^6_+ (35 r^2 + 18 r + 3) (Wendland 1995, Adv. Comput. Math. 4, 389): zero for
    r >= 1, four times continuously differentiable, and positive definite in three dimensions, hence on the unit
    sphere under the chordal distance."""
    t = np.clip(1.0 - r, 0.0, None)
    return t ** 6 * (35.0 * r * r + 18.0 * r + 3.0)

class UnresolvedCellMap:
    """The committed full-sky Gaia complement per HEALPix NESTED cell in the ICRS frame (level 5 = nside 32 for
    the pilot's table): `mu_ge15p5` per cell, the mean G surface brightness of the cell.

    `sample` reads it by the law it is asked for. `CELL_SAMPLING_NEAREST` (the default, v1's law) takes the ray's own
    cell. `CELL_SAMPLING_SMOOTH_RBF` (contract/sky_cell_sampling_v1.json) is a radial-basis interpolant of the natural
    log of the cells' linear surface brightness:

        ln F(x) = m + sum_j c_j psi(|x - x_j| / R),    F = exp(ln F),

    over the cell centres x_j (unit vectors; |.| the chordal distance), with psi the Wendland C4 function
    (`wendland_c4`), R = `RBF_SUPPORT_SPACINGS` cell spacings, m the mean of the cells' ln values, and the c_j fixed so
    that every cell centre returns its own cell. The interpolant depends on the direction alone, through distances to
    cell centres: it has no rings, knots or grid lines along which a lattice could form, and the poles and the longitude
    seam are ordinary points. It is four times continuously differentiable, so nothing is stepped or creased, and it is
    positive because it is an exponential. A query reads only the cells within R: each cell keeps a table of the
    centres that a query inside it can reach. Owned tests measure each property on the committed map, including its
    residual against knot-free readings of the same cells in the Galactic plane.

    `CELL_SAMPLING_SMOOTH_RBF_MEAN_PRESERVING` (contract/sky_cell_sampling_v2.json) is the same interpolant, kernel,
    support and candidate tables, of node values z_j instead of the cells' own ln values (m is then the mean of the z_j).
    A cell value is a mean surface brightness, and an interpolant exact at the centres does not keep it: the z_j are
    derived once (work/physics_observation_model/derive_mean_preserving_nodes_v1.py) so that, for every cell, the mean of
    F over the centres of its 4^MEAN_PRESERVING_SUB_LEVELS NESTED sub-cells equals the cell's value. The law reads them
    from a committed table that `bind_mean_preserving_nodes` holds to the declaration's sha256 pin; it never iterates.

    `coloured` builds a camera band's map from the G map and the colour factor table (contract/sky_unresolved_colour_v1.json);
    such a map carries `colour` (its band and the table's digest), which its band node table must name."""

    def __init__(self, nside, mu_by_cell, *, column):
        self.nside = int(nside)
        self.mu = np.asarray(mu_by_cell, dtype=np.float64)
        self.column = column
        if self.mu.size != 12 * self.nside ** 2 or not np.all(np.isfinite(self.mu)):
            raise SR.RadiometryError("cell map must cover every cell with a finite value", key="cell_map")
        self.colour = None
        self._rbf = None
        self._rbf_geom = None
        self._mp_nodes = None
        self._rbf_mp = None

    def _rbf_geometry(self):
        """The data-free part of the smooth laws, built once: the cell centres (unit vectors, NESTED order), each cell's
        list of the centres a query inside it can reach, the interpolation matrix psi(|x_i - x_j| / R) over the support,
        and the support radius R."""
        if self._rbf_geom is None:
            import scipy.sparse as sparse
            n = 12 * self.nside ** 2
            th, ph = healpix_pix2ang_ring(self.nside, np.arange(n))
            centres = np.empty((n, 3))
            centres[healpix_ang2pix_nest(self.nside, th, ph)] = np.stack(
                [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)
            spacing = math.sqrt(4.0 * math.pi / n)
            radius = RBF_SUPPORT_SPACINGS * spacing
            reach = radius + _CELL_REACH_SPACINGS * spacing
            lists, rows, cols = [], [], []
            for lo in range(0, n, 2048):
                dots = centres[lo:lo + 2048] @ centres.T
                for i, row in enumerate(dots):
                    lists.append(np.flatnonzero(row >= 1.0 - 0.5 * reach * reach))
                    j = np.flatnonzero(row >= 1.0 - 0.5 * radius * radius)
                    rows.append(np.full(j.size, lo + i))
                    cols.append(j)
            rows, cols = np.concatenate(rows), np.concatenate(cols)
            chord = np.sqrt(np.maximum(2.0 - 2.0 * np.einsum("ij,ij->i", centres[rows], centres[cols]), 0.0))
            matrix = sparse.csr_matrix((wendland_c4(chord / radius), (rows, cols)), shape=(n, n))
            self._rbf_geom = (centres, lists, matrix, radius)
        return self._rbf_geom

    def _rbf_solve(self, data):
        """The radial-basis interpolant of `data` (ln F at the cell centres, NESTED order): per cell the candidate centres
        (unit vectors) a query inside it can reach and their coefficients, padded with zero coefficients; the mean m;
        and the support radius."""
        import scipy.sparse.linalg as sparse_linalg
        centres, lists, matrix, radius = self._rbf_geometry()
        n = centres.shape[0]
        mean = float(data.mean())
        coef, info = sparse_linalg.cg(matrix, data - mean, rtol=1e-14, maxiter=10 * n)
        if info != 0 or np.max(np.abs(matrix @ coef + mean - data)) > 1e-10:
            coef = sparse_linalg.spsolve(matrix.tocsc(), data - mean)
        if np.max(np.abs(matrix @ coef + mean - data)) > 1e-10:
            raise SR.RadiometryError("the smooth law's interpolation system did not solve", key="cell_map")
        width = max(x.size for x in lists)
        cand = np.zeros((n, width), dtype=np.int64)
        cand_coef = np.zeros((n, width))
        for k, x in enumerate(lists):
            cand[k, :x.size] = x
            cand_coef[k, :x.size] = coef[x]
        return (centres[cand], cand_coef, mean, radius)

    def _rbf_state(self):
        """The smooth law's interpolant of the cells' own ln values, built once (`_rbf_solve`)."""
        if self._rbf is None:
            self._rbf = self._rbf_solve(-0.4 * math.log(10.0) * self.mu)    # ln of 10^(-0.4 mu), NESTED order
        return self._rbf

    def _rbf_mean_preserving_state(self):
        """The mean-preserving law's interpolant of the bound node values, built once (`_rbf_solve`)."""
        if self._mp_nodes is None:
            raise SR.RadiometryError("the mean-preserving law reads its node table: bind it first with "
                                     "bind_mean_preserving_nodes(path, sha256=...) from contract/sky_cell_sampling_v2.json",
                                     key="cell_map")
        if self._rbf_mp is None:
            self._rbf_mp = self._rbf_solve(self._mp_nodes)
        return self._rbf_mp

    def _law_state(self, method):
        if method == CELL_SAMPLING_SMOOTH_RBF:
            return self._rbf_state()
        if method == CELL_SAMPLING_SMOOTH_RBF_MEAN_PRESERVING:
            return self._rbf_mean_preserving_state()
        raise SR.RadiometryError(f"cell sampling {method!r} has no interpolant", key="cell_map")

    def _set_mean_preserving_nodes(self, nodes):
        """Set the mean-preserving law's node values (ln F per cell, NESTED order) without a table; the derivation uses
        it, and `bind_mean_preserving_nodes` after checking the table. Resets the law's interpolant."""
        nodes = np.array(nodes, dtype=np.float64)
        if nodes.shape != self.mu.shape or not np.all(np.isfinite(nodes)):
            raise SR.RadiometryError("mean-preserving nodes must give every cell a finite value", key="cell_map")
        self._mp_nodes = nodes
        self._rbf_mp = None

    def bind_mean_preserving_nodes(self, path, *, sha256):
        """Bind the mean-preserving law's node table (the file contract/sky_cell_sampling_v2.json names, held to the sha256
        it pins). The table must be the one derived for this map's law: its header names the law, the sub-sampling
        level, the support and this map's column, and its column of cell values equals this map's, cell for cell. A band
        map's table must also name its band and colour table (`coloured`), and a G map refuses a band's table. A
        refusal names the table. Returns the map."""
        import csv
        path = Path(path)
        if not path.is_file():
            raise SR.RadiometryError(f"mean-preserving node table {path.name} is absent", key=path.name)
        digest = SR.sha256_of(path)
        if digest != sha256:
            raise SR.RadiometryError(f"mean-preserving node table {path.name} has sha256 {digest}, the declaration pins "
                                     f"{sha256}", key=path.name)
        lines = path.read_text(encoding="utf-8").splitlines()
        header = {}
        for line in lines:
            hit = re.fullmatch(r"# ([a-z0-9_]+): (.*)", line)
            if hit:
                header[hit.group(1)] = hit.group(2).strip()
        want = {"schema": MEAN_PRESERVING_NODE_SCHEMA, "law": CELL_SAMPLING_SMOOTH_RBF_MEAN_PRESERVING,
                "source_column": self.column, "sub_cell_levels": str(MEAN_PRESERVING_SUB_LEVELS),
                "support_cell_spacings": repr(float(RBF_SUPPORT_SPACINGS))}
        if self.colour is not None:
            want.update(band=self.colour["band"], colour_table_sha256=self.colour["table_sha256"])
        wrong = sorted(k for k, v in want.items() if header.get(k) != v)
        if self.colour is None:
            wrong += sorted(k for k in ("band", "colour_table_sha256") if k in header)
        if wrong:
            raise SR.RadiometryError(f"mean-preserving node table {path.name} does not declare this law: {wrong}",
                                     key=path.name)
        rows = list(csv.DictReader(l for l in lines if l and not l.startswith("#")))
        n = self.mu.size
        try:
            index = np.array([int(r["hpx5"]) for r in rows], dtype=np.int64)
            values = np.array([float(r[self.column]) for r in rows])
            nodes = np.array([float(r["ln_f_node"]) for r in rows])
        except (KeyError, TypeError, ValueError) as exc:
            raise SR.RadiometryError(f"mean-preserving node table {path.name} is malformed: {exc!r}", key=path.name) from None
        if index.size != n or not np.array_equal(np.sort(index), np.arange(n)):
            raise SR.RadiometryError(f"mean-preserving node table {path.name} does not hold every cell once", key=path.name)
        mu, z = np.empty(n), np.empty(n)
        mu[index], z[index] = values, nodes
        if not np.array_equal(mu, self.mu):
            raise SR.RadiometryError(f"mean-preserving node table {path.name} was derived from other cell values",
                                     key=path.name)
        self._set_mean_preserving_nodes(z)
        return self

    def _log_surface_brightness(self, theta, phi, state=None, evaluator=RBF_EVALUATOR_REFERENCE):
        """ln F at colatitude theta and longitude phi (radians, 1-d arrays), by the smooth law unless `state` (from
        `_law_state`) names another interpolant, evaluated by the named evaluator (RBF_EVALUATORS)."""
        cand_xyz, cand_coef, mean, radius = self._rbf_state() if state is None else state
        cell = healpix_ang2pix_nest(self.nside, theta, phi)
        if evaluator == RBF_EVALUATOR_FAST:
            st = np.sin(theta)
            out = np.empty(theta.size)
            _fast_kernel()(np.ascontiguousarray(st * np.cos(phi)), np.ascontiguousarray(st * np.sin(phi)),
                           np.ascontiguousarray(np.cos(theta)), np.ascontiguousarray(cell, dtype=np.int64), cand_xyz,
                           cand_coef, float(mean), float(radius), out)
            return out
        q = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
        chord = np.sqrt(np.maximum(2.0 - 2.0 * np.einsum("pk,pck->pc", q, cand_xyz[cell]), 0.0))
        return mean + np.einsum("pc,pc->p", wendland_c4(chord / radius), cand_coef[cell])

    def log_surface_brightness(self, ra_deg, dec_deg, *, method=CELL_SAMPLING_SMOOTH_RBF,
                               evaluator=RBF_EVALUATOR_REFERENCE):
        """A declared interpolating law (`smooth_rbf` unless `method` names the mean-preserving one) as ln F (F in
        10^(-0.4 mu) per arcsec^2) at ICRS directions, any array shape, evaluated by `evaluator` (RBF_EVALUATORS: the numpy
        reference by default, the compiled loop on request)."""
        if evaluator not in RBF_EVALUATORS:
            raise SR.RadiometryError(f"smooth-law evaluator {evaluator!r} is not one of {RBF_EVALUATORS}", key="cell_map")
        state = self._law_state(method)
        ra, dec = np.broadcast_arrays(np.asarray(ra_deg, dtype=np.float64), np.asarray(dec_deg, dtype=np.float64))
        theta = np.radians(90.0 - dec).ravel()
        phi = np.mod(np.radians(ra), 2.0 * math.pi).ravel()
        out = np.empty(theta.size)
        for lo in range(0, theta.size, _RBF_CHUNK):
            out[lo:lo + _RBF_CHUNK] = self._log_surface_brightness(theta[lo:lo + _RBF_CHUNK], phi[lo:lo + _RBF_CHUNK], state,
                                                                   evaluator)
        if not np.all(np.isfinite(out)):
            raise SR.RadiometryError("the cell map's interpolated surface brightness is not finite", key="cell_map")
        return out.reshape(ra.shape)

    def linear_surface_brightness(self, ra_deg, dec_deg, *, method=CELL_SAMPLING_SMOOTH_RBF,
                                  evaluator=RBF_EVALUATOR_REFERENCE):
        """A declared interpolating law in linear units (10^(-0.4 mu) per arcsec^2) at ICRS directions, any array shape."""
        return np.exp(self.log_surface_brightness(ra_deg, dec_deg, method=method, evaluator=evaluator))

    @classmethod
    def load_csv(cls, path, *, sha256=None, column="mu_ge15p5"):
        import csv
        path = Path(path)
        if sha256 is not None:
            digest = SR.sha256_of(path)
            if digest != sha256:
                raise SR.RadiometryError(f"cell map has sha256 {digest}, declaration pins {sha256}", key=path.name)
        rows = list(csv.DictReader(l for l in path.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")))
        n = len(rows)
        nside = int(round(math.sqrt(n / 12.0)))
        if 12 * nside * nside != n:
            raise SR.RadiometryError(f"cell map has {n} rows, not 12 nside^2", key=path.name)
        mu = np.full(n, math.nan)
        for r in rows:
            mu[int(r["hpx5"])] = float(r[column])
        return cls(nside, mu, column=column)

    @classmethod
    def coloured(cls, g_map, table_path, *, sha256, band):
        """The camera band `band`'s map of the G map `g_map`: mu_X,i = mu_G,i + dmu_X,i, so F_X,i = kbar_X,i F_G,i, from the
        colour factor table that contract/sky_unresolved_colour_v1.json pins. The table must be the one derived for this map:
        held to `sha256`, its header names its schema and this map's column, it holds every cell once, its copy of the G
        values equals this map's cell for cell, and its offset is zero in every cell it flags OUTSIDE the colour coverage, so
        that the band map is the G map there exactly. The offset is added as printed decimals, so the band map is the same
        on every IEEE-754 machine. A refusal names the table. The map keeps `colour` (its band and the table's digest),
        which its node table must name (`bind_mean_preserving_nodes`)."""
        import csv
        path = Path(table_path)
        if band not in UNRESOLVED_COLOUR_BANDS:
            raise SR.RadiometryError(f"colour band {band!r} is not one of {UNRESOLVED_COLOUR_BANDS}", key=path.name)
        if g_map.colour is not None:
            raise SR.RadiometryError("a band map is coloured from the G map, not from another band map", key=path.name)
        if not path.is_file():
            raise SR.RadiometryError(f"colour factor table {path.name} is absent", key=path.name)
        digest = SR.sha256_of(path)
        if digest != sha256:
            raise SR.RadiometryError(f"colour factor table {path.name} has sha256 {digest}, the declaration pins {sha256}",
                                     key=path.name)
        lines = path.read_text(encoding="utf-8").splitlines()
        header = {}
        for line in lines:
            hit = re.fullmatch(r"# ([a-z0-9_]+): (.*)", line)
            if hit:
                header[hit.group(1)] = hit.group(2).strip()
        if header.get("schema") != UNRESOLVED_COLOUR_TABLE_SCHEMA or header.get("source_column") != g_map.column:
            raise SR.RadiometryError(f"colour factor table {path.name} is not a {UNRESOLVED_COLOUR_TABLE_SCHEMA} table of "
                                     f"column {g_map.column}", key=path.name)
        rows = list(csv.DictReader(l for l in lines if l and not l.startswith("#")))
        try:
            index = np.array([int(r["hpx5"]) for r in rows], dtype=np.int64)
            values = np.array([float(r[g_map.column]) for r in rows])
            offsets = np.array([float(r[f"dmu_{band.lower()}"]) for r in rows])
            outside = np.array([r["coverage"] == "OUTSIDE" for r in rows])
            known = all(r["coverage"] in UNRESOLVED_COLOUR_COVERAGE for r in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise SR.RadiometryError(f"colour factor table {path.name} is malformed: {exc!r}", key=path.name) from None
        n = g_map.mu.size
        if index.size != n or not np.array_equal(np.sort(index), np.arange(n)) or not known:
            raise SR.RadiometryError(f"colour factor table {path.name} does not hold every cell once with a known coverage",
                                     key=path.name)
        mu, dmu = np.empty(n), np.empty(n)
        mu[index], dmu[index] = values, offsets
        if not np.array_equal(mu, g_map.mu):
            raise SR.RadiometryError(f"colour factor table {path.name} was derived for other cell values", key=path.name)
        if not np.all(np.isfinite(dmu)) or np.any(offsets[outside] != 0.0):
            raise SR.RadiometryError(f"colour factor table {path.name} colours a cell outside its coverage or is not finite",
                                     key=path.name)
        out = cls(g_map.nside, g_map.mu + dmu, column=f"{g_map.column}_{band.lower()}")
        out.colour = {"band": band, "table_sha256": digest}
        return out

    def sample(self, ra_deg, dec_deg, *, method=CELL_SAMPLING_NEAREST, evaluator=RBF_EVALUATOR_REFERENCE):
        """The G surface brightness (mag/arcsec^2) at ICRS directions by the named law (class docstring); a smooth law is
        evaluated by `evaluator` (RBF_EVALUATORS)."""
        if evaluator not in RBF_EVALUATORS:
            raise SR.RadiometryError(f"smooth-law evaluator {evaluator!r} is not one of {RBF_EVALUATORS}", key="cell_map")
        if method == CELL_SAMPLING_NEAREST:
            theta = np.radians(90.0 - np.asarray(dec_deg, dtype=np.float64))
            phi = np.radians(np.asarray(ra_deg, dtype=np.float64))
            return self.mu[healpix_ang2pix_nest(self.nside, theta, phi)]
        if method not in (CELL_SAMPLING_SMOOTH_RBF, CELL_SAMPLING_SMOOTH_RBF_MEAN_PRESERVING):
            raise SR.RadiometryError(f"cell sampling {method!r} is not one of {CELL_SAMPLING_METHODS}", key="cell_map")
        return -2.5 / math.log(10.0) * self.log_surface_brightness(ra_deg, dec_deg, method=method, evaluator=evaluator)

UNRESOLVED_COLUMN_LIMIT = re.compile(r"mu_(ge|gt)(\d+)p(\d+)")

def unresolved_limit_token(column):
    """The limit token of a cell map's status, from its column: `G_GE_15P5` for `mu_ge15p5` (and for its band maps
    `mu_ge15p5_e5`), `G_GE_19P0` for `mu_ge19p0`. A column that does not encode a limit gives its own name upper-cased,
    never a limit it does not state."""
    hit = UNRESOLVED_COLUMN_LIMIT.match(str(column))
    if hit is None:
        return str(column).upper()
    return f"G_{hit.group(1).upper()}_{hit.group(2)}P{hit.group(3)}"

def unresolved_cell_provider(wave_nm, *, cell_map, solar_sed_w_m2_nm, m_sun_g, method=CELL_SAMPLING_NEAREST):
    """A callable `(l_deg, b_deg) -> dict` (the provider protocol of `background_map`) giving the unresolved-star
    radiance from the full-sky ICRS cell map: the Galactic direction is rotated to GCRS (= ICRS to the cell scale)
    and the map is read there by the named law (`UnresolvedCellMap.sample`: by default v1's ray's own cell; the
    smooth radial-basis law between cell centres when the caller binds contract/sky_cell_sampling_v1.json and passes
    `cell_sampling_for(...)`; its mean-preserving form when the caller binds contract/sky_cell_sampling_v2.json, passes
    `cell_sampling_for(...)` and has bound the node table to `cell_map`), solar-coloured through the anchored solar G
    magnitude. A band map (`UnresolvedCellMap.coloured`) carries its stars' colour in its cell values, so its status names
    the band; the spectral shape inside the band stays solar."""
    from . import sky_background as SKY
    rot = SKY.galactic_rotation()       # galactic -> GCRS, rows are vectors
    if method not in CELL_SAMPLING_METHODS:
        raise SR.RadiometryError(f"cell sampling {method!r} is not one of {CELL_SAMPLING_METHODS}", key="cell_map")
    colour = getattr(cell_map, "colour", None)
    limit = unresolved_limit_token(cell_map.column)
    label = (f"UNRESOLVED_STARS_OK_GAIA_HPX5_CELL_{limit}_SOLAR_COLOUR" if colour is None else
             f"UNRESOLVED_STARS_OK_GAIA_HPX5_CELL_{limit}_{colour['band']}_STAR_COLOURS_SOLAR_SHAPE")

    def provider(l_deg, b_deg):
        l = np.radians(np.asarray(l_deg, dtype=np.float64))
        b = np.radians(np.asarray(b_deg, dtype=np.float64))
        v = np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], axis=-1) @ rot
        ra = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
        dec = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
        mu = cell_map.sample(ra, dec, method=method)
        rad = SKY.radiance_from_solar_surface_brightness(mu, wave_nm, solar_sed_w_m2_nm, m_sun_g)
        status = np.full(l.size, label, dtype=object)
        return {"radiance": rad, "status": status, "mu_g_mag_arcsec2": mu}

    return provider
