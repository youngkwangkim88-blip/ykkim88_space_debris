"""Weighted variance for a fixed template with fresh shot and read noise."""

import numpy as np


def weak_streak(inputs, settings):
    pars = settings["weak_streak"]
    image = inputs["weak_streak_e"]
    weights = image / image.sum()
    sky = pars["sky_rate"] * pars["exposure_s"]
    dark = pars["dark_rate"] * pars["exposure_s"]
    read = pars["read_e"]
    rng = np.random.Generator(np.random.PCG64(pars["seed"]))
    estimates = []
    for _ in range(pars["draws"]):
        observed = rng.poisson(image + sky + dark) + rng.normal(0, read, image.shape)
        estimates.append(np.sum(weights * (observed - sky - dark)))
    prediction = np.sum(weights**2 * (image + sky + dark + read**2))
    return "weak_streak_pre_ADC", pars["draws"], np.var(estimates, ddof=1), prediction


def native_cameras(inputs, settings):
    pars = settings['native']
    flux, exposure = pars['flux_e'], pars['exposure_s']
    for index, name in enumerate(('CMV4000','virtual_sCMOS')):
        detector = pars['detectors'][name]
        p = inputs[name+'_P4']
        s2 = np.sum(p*p)
        dark = detector['dark']
        read = detector['read']
        for variant in ('C0','CW'):
            sky = np.zeros_like(p) if variant=='C0' else inputs[name+'_sky_rate']
            mean = flux*p + (sky+dark)*exposure
            assert mean.max()<.1*detector['fullwell']
            seed = 91827+13*index+int(variant=='CW')
            rng = np.random.Generator(np.random.PCG64(seed))
            estimates = []
            for _ in range(16):
                shape = (64,*p.shape)
                image = rng.poisson(mean,size=shape) + rng.normal(0,read,size=shape)
                image += rng.uniform(-5/3,5/3,size=shape)
                image -= (sky+dark)*exposure
                estimates.extend(np.sum(image*p,axis=(1,2))/s2)
            pixel_variance = flux*p + dark*exposure + read**2 + detector['quant_var'] + sky*exposure
            prediction = np.sqrt(np.sum(p*p*pixel_variance)/s2**2)
            yield name+'_'+variant, len(estimates), np.std(estimates,ddof=1), prediction
