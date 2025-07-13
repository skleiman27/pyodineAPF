import os
import sys
import glob

from astropy.io import fits as fits
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pyodine import template as temp
from astroquery.simbad import Simbad
Simbad = Simbad()

sci_dir = "/path/to/sci/dir"
raw_dir = "/path/to/raw/dir"

def or_none(header, key, fallback_value=None):
    """A convenience function to prevent non-existent Fits-header cards from
    throwing up errors
    
    :param header: The Fits-header.
    :type header: :class:`fits.Header`
    :param key: The keyword of the header card of interest.
    :type key: str
    :param fallback_value: What to return if the header card does not exist
        (default: None).
    :type fallback_value: str, int, float, or None
    
    :return: The header card or the 'fallback_value'.
    :rtype: str, int, float, or None
    """
    try:
        return header[key]
    except KeyError:
        # TODO: Log this event
        return fallback_value

def blaze_add(out_file):
    """
    Calculates blaze function and creates new fits file with blaze function added

    Inputs: 
    out_file (str): Path to PypeIt output file
    """

    for i in np.arange(1,56):
        d = out_file[i].data
        header = out_file[i].header
        order = header["HIERARCH ECH_ORDER"]
        wl = d["BOX_WAVE"]
        flux = d["BOX_COUNTS"]
        blaze = temp.normalize.top(flux,deg=5,max_iter=100)
        if i ==1:
            blaze_funcs = blaze
        if i > 1:
            blaze_funcs = np.vstack((blaze_funcs, blaze))


    blaze_hdu = fits.ImageHDU(data = blaze_funcs)
    with fits.open(out_file) as hdul:
        name = hdul[0].header["FILENAME"]
        dir_path = os.path.dirname(out_file)
        out_path = os.path.join(dir_path,"PyPyPied_"+name)
        hdul.append(blaze_hdu)
        hdul.writeto(out_path, overwrite=True)

def from_raw(out_header,raw_header):
    """
    Attaches information from raw telescope fits to PypeIt output file

    Inputs:
    out_header (header): header from PypeIt output
    raw_header (header): header from raw telescope output
    """

    out_header['THEMIDPT'] = raw_header['THEMIDPT']
    out_header.comments['THEMIDPT'] = 'Final photon-weighted midpoint'

    out_header['ICELNAM'] = raw_header['ICELNAM']
    out_header.comments['ICELNAM'] = 'Iodine cell position'

def from_simbad(header,use_simbad=True,catalog_file=None):
    """
    Uses target name to pull data from Simbad (or from input file) using astroquery

    Inputs:
    header (header): header from PypeIt output
    use_simbad (bool, def=True): decides if astroquery is to be used to pull data
    catlog_file(str, def = None): file path to catalog if not using Simbad
    """
    name = or_none(header, 'TARGET')
    Simbad.add_votable_fields('pmra', 'pmdec')
    info = Simbad.query_object(name)
    proper_motion = (info['pmra'].data[0], info['pmdec'].data[0])

    header["PMRA"] = info['pmra'].data[0]
    header.comments['PMRA'] = 'RA proper motion (mas)'

    header["PMDEC"] = info['pmdec'].data[0]
    header.comments['PMDEC'] = 'Dec proper motion (mas)'

    header["RA"] = info['ra'].data[0]
    header["DEC"] = info['dec'].data[0]

sci_ims = glob.glob(sci_dir+"/spec1d*fits")
for sci_im in sci_ims:

    out_file = fits.open(sci_im)
    out_header = out_file[0].header

    raw_name = out_header["FILENAME"]
    raw_im = os.path.join(raw_dir,raw_name)
    raw_file = fits.open(raw_im)
    raw_header = raw_file[0].header

    from_raw(out_header, raw_header)

    blaze_add(out_file)


raw_obs = 'raw_placeholder.fits' #Insert raw observation file path here
pypeit_out = 'pyped_placeholder.fits' #Insert pyepit output file path here

#Opens files and headers, allows the output to be updated

out_file = fits.open(pypeit_out)
out_header = out_file[0].header

raw_file = fits.open(raw_obs)
raw_header = raw_file[0].header

out_header['THEMIDPT'] = raw_header['THEMIDPT']
out_header.comments['THEMIDPT'] = 'Final photon-weighted midpoint'

out_header['ICELNAM'] = raw_header['ICELNAM']
out_header.comments['ICELNAM'] = 'Iodine cell position'

#add barycorrpy calculation here, add BVC estimate to header

#adds hdu with blaze function 
h = fits.open(out_file)
for i in np.arange(1,56):
    d = h[i].data
    header = h[i].header
    order = header["HIERARCH ECH_ORDER"]
    wl = d["BOX_WAVE"]
    flux = d["BOX_COUNTS"]
    blaze = temp.normalize.top(flux,deg=5,max_iter=100)
    if i ==1:
        blaze_funcs = blaze
    if i > 1:
        blaze_funcs = np.vstack((blaze_funcs, blaze))


blaze_hdu = fits.ImageHDU(data = blaze_funcs)
with fits.open(out_file) as hdul:
    name = hdul[0].header["FILENAME"]
    dir_path = os.path.dirname(out_file)
    out_path = os.path.join(dir_path,"PyPyPied_"+name)
    hdul.append(blaze_hdu)
    hdul.writeto(out_path, overwrite=True)
