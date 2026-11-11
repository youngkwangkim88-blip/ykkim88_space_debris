"""Re-evaluate the two annual skies for 100 rotating states with either read-noise value.

Run with Python, NumPy and SciPy and a v30 checkout supplied as the argument.
"""
from pathlib import Path
import argparse
import sys
import csv
import numpy as np
from scipy.constants import astronomical_unit

def threshold_with_read_noise(powers, exposure, sky_rate, read_noise, n_pixels, dark):
    """Equation (5) for profile weights, including the finite shared calibration masters."""
    s1, s2, s3, s4 = powers
    b = sky_rate * exposure
    base = dark * exposure + read_noise**2 + (10 / 3)**2 / 12
    vd = base / 64
    flat = 5000.
    vu = (flat + base) / 32 + vd
    c0 = s2 * (b + base + vd) + (vu*b*b/flat**2 - 2*vd*b/flat)*(s2 - s1*s1/n_pixels)
    c1 = s3 + (2*vu*b/flat**2 - 2*vd/flat)*(s3 - s1*s2/n_pixels)
    c2 = vu/flat**2*(s4 - s2*s2/n_pixels)
    a = s2**2 - 100*c2
    assert np.all(a>0)
    k = (100*c1 + np.sqrt((100*c1)**2 + 400*a*c0))/(2*a)
    np.testing.assert_allclose(k*s2/np.sqrt(c0 + k*c1 + k*k*c2), 10.)
    return 100/np.sqrt(k), k

def main(root):
    sys.path.insert(0,str(root))
    import fragment as f
    bins = {}
    with (root/'data/ranges_reference.csv').open(encoding='utf8') as fp:
        for r in csv.DictReader(fp):
            bins[r['band'],r['shape'],r['material'],float(r['T']),r['case']] = r
    light = f.sky.sightlines()['weak']
    dark = 0.7754788379076724
    groups = {}
    for band in ['E12','E5']:
        camera = f.setup(band)
        for shape in ['plate','cuboid']:
            t, q = f.attitudes(camera,shape,10.)
            for surface in ['grey','alum','mli','solar_cell','spback']:
                flux = f.factor(camera, q, shape, surface, light['target_phase_far_sun_deg']) * (astronomical_unit/1e3/light['sun_distance_km'])**2
                for exposure in [4.,.5]:
                    selected=t<exposure
                    im=f.render_fft(camera,flux[:,selected],t[selected],exposure,f.PAPER_SPEED)
                    powers=[np.sum(im**i,axis=1) for i in range(1,5)]
                    assert im.shape[0]==100
                    peak=im.max(axis=1)
                    for sigma in [3.3,11.96]:
                        by_sky={}
                        noise_var={}
                        for case in ['weak_date','strong_fixed']:
                            ref=bins[band,shape,surface,exposure,case]
                            rate=float(ref['b'])
                            rng, k = threshold_with_read_noise(powers,exposure,rate,sigma,f.NATIVE_PIXELS,dark)
                            noise_var[case]=(rate+dark)*exposure+sigma**2+(10/3)**2/12
                            if sigma==3.3:
                                np.testing.assert_allclose(rng,f.threshold(powers,100.,exposure,rate,f.NATIVE_PIXELS,dark)[0],rtol=1.e-10)
                                np.testing.assert_allclose(np.quantile(rng,[.05,.5,.95]), [float(ref[key]) for key in ['p05','p50','p95']],rtol=0,atol=.005)
                            assert np.all(peak*k+(dark+rate)*exposure<7400)
                            by_sky[case]=np.median(rng)
                        groups[band,shape,surface,exposure,sigma]=(by_sky['weak_date'],by_sky['strong_fixed'])
                        if (band,shape,surface,exposure)==('E12','plate','grey',4.0):
                            loss=100*(1-by_sky['strong_fixed']/by_sky['weak_date'])
                            approx=100*(1-(noise_var['weak_date']/noise_var['strong_fixed'])**.25)
                            cross_b=(sigma**2+(10/3)**2/12)/(rate+dark)
                            print('gray_plate',sigma, by_sky, 'loss (%)', loss, 'fourth-root (%)',approx,'bright crossover (s)', cross_b,flush=True)
    for sigma in [3.3,11.96]:
        for exposure in [.5,4.]:
            fg=[]; bg=[]; sky=[]; n=0
            for shape in ['plate','cuboid']:
                for surface in ['grey','alum','mli','solar_cell','spback']:
                    f5,b5=groups['E5',shape,surface,exposure,sigma];f12,b12=groups['E12',shape,surface,exposure,sigma]
                    gf,gb=100*(f12/f5-1),100*(b12/b5-1);ls=100*(1-b12/f12)
                    fg.append(gf);bg.append(gb);sky.append(ls)
                    n += ls > max(gf,gb)
            print(sigma,exposure,'QE faint %',min(fg),max(fg),'QE bright %',min(bg),max(bg),'sky %',min(sky),max(sky),'sky loss larger than both QE gains in',n,'of 10',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('release',type=Path,nargs='?',default=Path(__file__).resolve().parents[2]);main(p.parse_args().release.resolve())
