from __future__ import print_function
from __future__ import division
import os
import sys
import glob
import argparse

from astropy.io import fits as fits
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from jdcal import gcal2jd
from pyodine import template as temp
from pyodine import timeseries as ts
from astroquery.simbad import Simbad
from barycorrpy import get_BC_vel , exposure_meter_BC_vel
#import utc_tdb

parser = argparse.ArgumentParser(description="Set default options")
parser.add_argument('-r', '--rawdir', \
                        default=None, help='File path to directory of raw images')
parser.add_argument('-s', '--scidir', default=None, \
                        help='Filepath to directory of science images')
parser.add_argument('-o', '--outdir', default=None, \
                        help='Output directory')
parser.add_argument('-c', '--catalog', default=None, \
                        help='Flag to use star catalog (defaults to simbad otherwise)')

#parser.add_argument('inpath')
                        
opt = parser.parse_args()


Simbad = Simbad()

sci_dir = opt.scidir
raw_dir = opt.rawdir
if sci_dir == None:
    raise ValueError("MISSING SCIENCE DIRECTORY")
if raw_dir == None:
    raise ValueError("MISSING RAW IMAGE DIRECTORY")

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

def blaze_add(out_file,out_path):
    """
    Calculates blaze function and creates new fits file with blaze function added

    Inputs: 
    out_file (hdul): fits file
    out_path (str): file path to out_file
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
    with fits.open(out_path) as hdul:
        name = hdul[0].header["FILENAME"]
        if opt.outdir == None:

            dir_path = os.path.dirname(out_file)
        else:
            dir_path = opt.outdir
        final_path = os.path.join(dir_path,"PyPyPied_"+name)
        hdul.append(blaze_hdu)
        hdul.writeto(final_path, overwrite=True)

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
    Simbad.add_votable_fields('pmra', 'pmdec','rvz_radvel','rvz_redshift','plx_value')
    info = Simbad.query_object(name)

    header["PMRA"] = info['pmra'].data[0]
    header.comments['PMRA'] = 'RA proper motion (mas/yr)'

    header["PMDEC"] = info['pmdec'].data[0]
    header.comments['PMDEC'] = 'Dec proper motion (mas)'

    header["radvel"]                          = info['rvz_radvel'].data[0]
    header.comments['radvel'] = 'Radial Velocity'

    header["redshift"] = info['rvz_redshift'].data[0]
    header.comments['redshift'] = 'Redshift'

    header["parallax"] = info['plx_value'].data[0]
    header.comments['parallax'] = 'Parallax'

    header["RA"] = info['ra'].data[0]
    header["DEC"] = info['dec'].data[0]

    

sci_ims = glob.glob(sci_dir+"/spec1d*fits")
os.makedirs(opt.outdir, exist_ok=True)

for sci_im in sci_ims:
    with fits.open(sci_im, mode='update') as out_file:
        
        out_file = fits.open(sci_im)
        out_header = out_file[0].header

        raw_name = out_header["FILENAME"]
        raw_im = os.path.join(raw_dir,raw_name)
        raw_file = fits.open(raw_im)
        raw_header = raw_file[0].header

        from_raw(out_header, raw_header)

        if opt.catalog == None:
            from_simbad(out_header)
        else:
            open()

        blaze_add(out_file,sci_im)

        date = out_header["THEMIDPT"]
        pri_date = date[0:10].split('-')
        sec_date = date[11: ].split(':')
        for i in np.arange(0,3):
            pri_date[i] = float(pri_date[i])
            sec_date[i] = float(sec_date[i])
        JD = sum(gcal2jd(pri_date[0],pri_date[1],pri_date[2]))+(sec_date[0]+sec_date[1]/60+sec_date[2]/3600)/24

        bc_vel = get_BC_vel(JDUTC=JD, ra=out_header["RA"], dec=out_header["DEC"], lat=out_header["LAT-OBS"], longi=out_header["LON-OBS"], alt=out_header["ALT-OBS"], pmra=out_header["PMRA"],
                    pmdec=out_header["PMDEC"], px=out_header["Parallax"], rv=out_header["radvel"], zmeas=out_header['redshift'],epoch=2451545.0)
        print(bc_vel)
        out_header["BVC"] = bc_vel[0][0]
        out_header.comments['BVC'] = 'Barycentric Velocity'

        name = out_file[0].header["FILENAME"]
        if opt.outdir == None:
            dir_path = os.path.dirname(out_file)
        else:
            dir_path = opt.outdir
        out_path = os.path.join(dir_path,"PyPyPied_"+name)
        out_file.writeto(out_path, overwrite=True)

#/data/APF_reductions/HD203030/20241001/raw/

#/data/APF_reductions/HD203030/20241001/Science/

#/data/APF_reductions/HD203030/20241001/PyPyPiOUT/