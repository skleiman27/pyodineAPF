from astropy.io import fits as fits
import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
import tqdm

raw_obs = 'raw_placeholder.fits' #Insert raw observation file path here
pypeit_out = 'pyped_placeholder.fits' #Insert pyepit output file path here

#Opens files and headers, allows the output to be updated

out_file = fits.open(pypeit_out, mode='update')
out_header = out_file[0].header

raw_file = fits.open(raw_obs)
raw_header = raw_file[0].header

out_header['THEMIDPT'] = raw_header['THEMIDPT']
out_header.comments['THEMIDPT'] = 'Final photon-weighted midpoint'

out_header['ICELNAM'] = raw_header['ICELNAM']
out_header.comments['ICELNAM'] = 'Iodine cell position'


