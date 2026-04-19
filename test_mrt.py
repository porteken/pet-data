"""Test script for Mean Radiant Temperature (MRT) calculations using thermofeel."""

import numpy as np
import thermofeel as tf

cossza = np.array([1.0, 0.5, 0.0])
fdir = np.array([200.0, 100.0, 0.0])
dsrp = tf.approximate_dsrp(fdir, cossza)

ssrd = np.array([300.0, 200.0, 100.0])
ssr = np.array([250.0, 150.0, 50.0])
strd = np.array([350.0, 300.0, 250.0])
strr = np.array([100.0, 80.0, 60.0])

mrt = tf.calculate_mean_radiant_temperature(
    ssrd=ssrd,
    ssr=ssr,
    dsrp=dsrp,
    strd=strd,
    fdir=fdir,
    strr=strr,
    cossza=cossza,
)
