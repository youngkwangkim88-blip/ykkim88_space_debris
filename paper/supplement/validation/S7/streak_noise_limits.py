"""Compare the printed footprint with explicit finite pixels at fixed charge and sky."""
from pathlib import Path
import csv
import math

import numpy as np

HERE = Path(__file__).resolve().parent
READ = 11.96
EXPOSURE = 0.1
SKY = 1e4 * EXPOSURE
DARK = 0.12 * EXPOSURE
BASE_VARIANCE = SKY + DARK + READ**2
DRAWS = 50_000


def fractions(length, start_fraction, end_fraction):
    """An aligned uniform 3 x 3 pixel square translates horizontally during a single read."""
    left = length*start_fraction
    right = length*end_fraction
    columns = np.arange(math.ceil(right)+3)
    def overlap(position):
        return np.maximum(0, np.minimum(columns+1,position+3)-np.maximum(columns,position))/9
    if left==right:
        horizontal = overlap(left)
    else:
        # Pixel overlap is piecewise linear with knots at integer translations.
        knots = np.unique(np.r_[left, np.arange(math.ceil(left),math.ceil(right)), right])
        horizontal = sum((b-a)*overlap((a+b)/2) for a,b in zip(knots,knots[1:]))/(right-left)
    profile = np.tile(horizontal,3)
    assert np.isclose(profile.sum(),1)
    return profile[profile>0]


def simulate(profile, charge, rng):
    target = charge*profile
    w = np.stack([np.ones_like(profile), profile])
    sums = np.zeros(2)
    sums2 = np.zeros(2)
    for _ in range(DRAWS//1000):
        # Uniform sky and dark means are known here. The centered stochastic signal
        # contains target, sky and dark shot noise plus one independent pixel read.
        mu = target + SKY + DARK
        noise = rng.poisson(mu, size=(1000,len(mu)))-mu
        noise += rng.normal(0,READ,size=noise.shape)
        x = noise @ w.T
        sums += x.sum(axis=0)
        sums2 += (x*x).sum(axis=0)
    return (sums2 - sums*sums/DRAWS)/(DRAWS-1)


def main(output_path=HERE/'streak_noise_limits.csv'):
    with (HERE/'filho_sensor_comparison.csv').open(encoding='utf-8') as stream:
        row=next(r for r in csv.DictReader(stream) if r['case']=='solar_400_1000nm__9pixel_zodi_upper_read_11.96')
    charge=float(row['target_e_at_180km'])
    speed400_length=400/(206265*.00375/16)*EXPOSURE
    cases=[(0.,'steady'),(speed400_length,'steady'),(3.,'steady'),(10.,'steady'),(10.,'central_tenth')]
    rng=np.random.default_rng(20261104)
    result=[]
    for length,light in cases:
        start,end=(.45,.55) if light=='central_tenth' else (0.,1.)
        p=fractions(length,start,end)
        weights=np.stack([np.ones_like(p),p])
        target=charge*p
        predicted=(weights**2) @ (target+BASE_VARIANCE)
        signal=weights @ target
        snrs=signal/np.sqrt(predicted)
        sample=simulate(p,charge,rng)
        variance_ratios=sample/predicted
        n=3*length
        multiplier=n+np.sqrt(n)*length
        scalar=charge/math.sqrt(charge+multiplier*BASE_VARIANCE)
        literal=charge/math.sqrt(charge+multiplier*(SKY+DARK+READ))
        result.append(dict(light=light,trail_px=length,source_e=charge,sky_e_px=SKY,
            occupied_pixels=len(p),template_noise_pixels=1/(p@p),printed_multiplier=multiplier,
            filho_snr_read_as_variance=scalar,filho_snr_insert_table_rms=literal,
            eq9_aperture_snr=snrs[0],eq9_template_snr=snrs[1],
            empirical_to_predicted_variance_aperture=variance_ratios[0],
            empirical_to_predicted_variance_template=variance_ratios[1],draws=DRAWS))
    assert np.isclose(result[0]['eq9_aperture_snr'],float(row['snr_at_180km']))
    assert np.isclose(result[0]['eq9_aperture_snr'],result[0]['eq9_template_snr'])
    assert result[1]['filho_snr_read_as_variance']>result[1]['eq9_template_snr']
    assert result[3]['filho_snr_read_as_variance']<result[3]['eq9_template_snr']
    assert result[4]['eq9_template_snr']>result[3]['eq9_template_snr']
    with Path(output_path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=result[0])
        writer.writeheader()
        writer.writerows(result)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('streak_noise_limits.csv'))
    main(parser.parse_args().output)
