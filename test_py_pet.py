"""Test script for the Python implementation of PET corrected calculations."""

import sys

sys.path.append("/home/kenneth-porter/pet-data")
from pet_corrected import pet_corrected

res = pet_corrected(tair=25, t_mrt=25, v_air=0.1, rh=50, icl=0.9)

res2 = pet_corrected(tair=25, t_mrt=25, v_air=0.1, rh=50, icl=0.5)
