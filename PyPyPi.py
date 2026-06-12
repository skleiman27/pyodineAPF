from __future__ import print_function
from __future__ import division
import os
import sys
import glob
import argparse
import warnings

from astropy.io import fits as fits
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from jdcal import gcal2jd
from pyodine import template as temp
from pyodine import timeseries as ts
from astroquery.simbad import Simbad
from barycorrpy import get_BC_vel , exposure_meter_BC_vel
import utc_tdb
from progressbar import ProgressBar

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

def vactoair(wavelength):
    sigma_sq = (1.e4/wavelength)**2. #wavenumber squared
    factor = 1 + (5.792105e-2/(238.0185-sigma_sq)) + (1.67918e-3/(57.362-sigma_sq))
    factor = factor*(wavelength>=2000.) + 1.*(wavelength<2000.) #only modify above 2000A
    # Convert
    new_wave = wavelength/factor
    return new_wave

def waveconvadd(out_file):

    for i in np.arange(1,56):
        d = out_file[i].data
        wl_box = d["BOX_WAVE"]
        wl_opt = d["OPT_WAVE"]
        wl_boxair = vactoair(wl_box)
        wl_optair = vactoair(wl_opt)
        if i == 1:
            wl_boxairlist = wl_boxair
            wl_optairlist = wl_optair
        if i > 1:
            wl_boxairlist = np.vstack((wl_boxairlist, wl_boxair))
            wl_optairlist = np.vstack((wl_optairlist, wl_optair))
    wl_boxairHDU = fits.ImageHDU(data = wl_boxairlist)
    wl_optairHDU = fits.ImageHDU(data = wl_optairlist)
    return wl_boxairHDU, wl_optairHDU


def blaze_add_top(out_file,out_path):
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
        blaze = temp.normalize.top(flux,deg=5,max_iter=1000)
        if i ==1:
            blaze_funcs = blaze
        if i > 1:
            blaze_funcs = np.vstack((blaze_funcs, blaze))


    blaze_hdu = fits.ImageHDU(data = blaze_funcs)
    #print(np.shape(blaze_funcs))
    return blaze_hdu
    #with fits.open(out_path) as hdul:
    #    name = hdul[0].header["FILENAME"]
    #    if opt.outdir == None:#
    #
    #        dir_path = os.path.dirname(out_file)
    #    else:
    #        dir_path = opt.outdir
    #    final_path = os.path.join(dir_path,"PyPyPied_"+name)
    #    hdul.append(blaze_hdu)
    #    hdul.writeto(final_path, overwrite=True)

def blaze_add_alphashape(out_file,out_path):
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
        sigma = d["BOX_COUNTS_SIG"]
        wv_vet, spec_vet, sig_vet = remove_cosmic_rays_1order(wl, flux, sigma)
        spec_norm, sig_norm, cfit = AFS_continuum_norm_1order(wv_vet, spec_vet, sig_vet, wl, flux, sigma, plot=False)

        if i ==1:
            blaze_funcs = spec_norm
        if i > 1:
            blaze_funcs = np.vstack((blaze_funcs, spec_norm))


    blaze_hdu = fits.ImageHDU(data = blaze_funcs)
    #print(np.shape(blaze_funcs))
    return blaze_hdu



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

    header["radvel"] = info['rvz_radvel'].data[0]
    header.comments['radvel'] = 'Radial Velocity'

    header["redshift"] = info['rvz_redshift'].data[0]
    header.comments['redshift'] = 'Redshift'

    header["parallax"] = info['plx_value'].data[0]
    header.comments['parallax'] = 'Parallax'

    header["RA"] = info['ra'].data[0]
    header["DEC"] = info['dec'].data[0]

def run():
    sci_dir = opt.scidir
    raw_dir = opt.rawdir
    if sci_dir == None:
        raise ValueError("MISSING SCIENCE DIRECTORY")
    if raw_dir == None:
        raise ValueError("MISSING RAW IMAGE DIRECTORY")    

    sci_ims = glob.glob(sci_dir+"/spec1d*fits")
    os.makedirs(opt.outdir, exist_ok=True)
    bar = tqdm(total = len(sci_ims), desc = 'Images Processed')

    for i, sci_im in enumerate(sci_ims):
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


            wl_boxair_HDU, wl_optair_HDU = waveconvadd(out_file)

            blaze_hdu = blaze_add_top(out_file,sci_im)
            #blaze_hdu = blaze_add_alphashape(out_file,sci_im)

            
    # astropy.time
            date = out_header["THEMIDPT"]
            pri_date = date[0:10].split('-')
            sec_date = date[11: ].split(':')
            for i in np.arange(0,3):
                pri_date[i] = float(pri_date[i])
                sec_date[i] = float(sec_date[i])
            JD = sum(gcal2jd(pri_date[0],pri_date[1],pri_date[2]))+(sec_date[0]+sec_date[1]/60+sec_date[2]/3600)/24
            out_header["JD-UTC"] = JD

            bc_vel = get_BC_vel(JDUTC=JD, ra=out_header["RA"], dec=out_header["DEC"], lat=out_header["LAT-OBS"], longi=out_header["LON-OBS"], alt=out_header["ALT-OBS"], pmra=out_header["PMRA"],
                        pmdec=out_header["PMDEC"], px=out_header["Parallax"], rv=out_header["radvel"], zmeas=out_header['redshift'],epoch=2451545.0) #This is J2000 for epoch
            #print(bc_vel)
            out_header["BVC"] = bc_vel[0][0]
            out_header.comments['BVC'] = 'Barycentric Velocity'

            name = out_file[0].header["FILENAME"]
            if opt.outdir == None:
                dir_path = os.path.dirname(out_file)
                warnings.warn("Warning: No ouput directory detected. Defaulting to science directory" )
            else:
                dir_path = opt.outdir
            out_file.append(blaze_hdu)
            out_file.append(wl_boxair_HDU)
            out_file.append(wl_optair_HDU)
            out_path = os.path.join(dir_path,"PyPyPied_"+out_header["ICELNAM"]+"_"+out_header["TARGET"]+"_"+name)
            #out_file.info()
            out_file.writeto(out_path, overwrite=True)
            bar.update(i)

#/data/APF_reductions/HD203030/20241001/raw/

#/data/APF_reductions/HD203030/20241001/Science/

#/data/APF_reductions/HD203030/20241001/PyPyPiOUT_AS/

import numpy as np
import alphashape
from descartes import PolygonPatch
from astropy.io import fits
import matplotlib.pyplot as plt
from localreg import *
from astropy.io import fits


# AFS implementation from Xu et al. 2019

def AFS_continuum_norm_1order(wv, spec, sigma, wv_to, spec_to, sigma_to, q=0.95, plot=False):

    # Step zero: set some initial values
    alpha = (1./6.) * (np.max(wv) - np.min(wv))

    # Step 1: rescale intensity vector by u ----------------------
    u = (np.max(wv) - np.min(wv))/(10.*np.max(spec))
    spec *= u
    spec_to *= u
    sigma_to *= u
    #alpha = 2. * u

    # Step 2: calculate alpha shape ------------------------------
    spec_stack = np.transpose(np.vstack((wv, spec)))
    spec_stack_tuple = tuple(map(tuple, spec_stack))
    alpha_shape = alphashape.alphashape(spec_stack_tuple, alpha)

    # Step 3: Extract vertices that correspond to ----------------
    #         y-maxima in the alpha shape.
    #         Carry out local polynomial regression.

    try:
        x, y = alpha_shape.exterior.coords.xy
    except:
        # If we have a multipolygon
        polygon_points = [point for polygon in alpha_shape.geoms for point in polygon.exterior.coords[:-1]]
        x, y = list(zip(*polygon_points))[0], list(zip(*polygon_points))[1]


    AS_tilde = get_AS_tilde(wv, x, y)
    x_AS_tilde, y_AS_tilde = np.transpose(AS_tilde)[0], np.transpose(AS_tilde)[1]

    # Normalize x-values for weighting
    x_AS_tilde_norm = (x_AS_tilde[1] - np.min(x_AS_tilde))/(np.max(x_AS_tilde) - np.min(x_AS_tilde))
    B1 = local_polynomial_regr(x_AS_tilde, y_AS_tilde, x_AS_tilde)

    # Make sure there are no NaNs; set equal to continuum if they are found
    #B1 = np.nan_to_num(B1, nan=1.0)

    # Divide out continuum to get initial guess
    y1 = spec/B1

    # Step 4: Identify non-absorption points
    overlap_spec_indices = np.in1d(np.transpose(AS_tilde)[1], spec).nonzero()
    W_alpha = np.transpose(np.array([wv[overlap_spec_indices], spec[overlap_spec_indices]]))

    if plot == True:
        plot_alpha_shape_fit(wv, spec, alpha_shape, AS_tilde, W_alpha)

    # Step 5: Carry out second local polynomial regression -------
    S_alpha = get_S_alpha(y1, W_alpha, wv, spec, q) # q is the quantile
    x_S_alpha, y_S_alpha = np.transpose(S_alpha)[0], np.transpose(S_alpha)[1]

    # Normalize x-values for weighting
    x_S_alpha_norm = (x_S_alpha - np.min(x_S_alpha))/(np.max(x_S_alpha) - np.min(x_S_alpha))
    wv_all_norm = (wv - np.min(x_S_alpha))/(np.max(x_S_alpha) - np.min(x_S_alpha))
    B2 = local_polynomial_regr(x_S_alpha, y_S_alpha, wv_to)

    # Make sure there are no NaNs; set equal to continuum if they are found
    #B2 = np.nan_to_num(B2, nan=1.0)

    # Step 6: divide by B2 to get final blaze-removed spectrum ---
    y2 = spec_to/B2

    # divide sigma too
    y2_sig = sigma_to/B2

    # rename so everything is clearer
    spec_norm, sigma_norm, cfit = y2, y2_sig, B2

    return spec_norm, sigma_norm, cfit


def AFS_continuum_norm_1star(wv_1star, spec_1star, sigma_1star):

    num_pix_1order = len(wv_1star[0])

    spec_norm_1star = np.zeros(num_pix_1order)
    sigma_norm_1star = np.zeros(num_pix_1order)
    cfit_1star = np.zeros(num_pix_1order)

    for order in range(0, len(wv_1star)):
        wv_to, spec_to, sigma_to = wv_1star[order], spec_1star[order], sigma_1star[order]

        # Copy to avoid overwriting original values
        wv1, spec1, sigma1 = wv_1star[order], spec_1star[order], sigma_1star[order]

        wv_vetted, spec_vetted, sigma_vetted = remove_cosmic_rays_1order(wv1, spec1, sigma1)
        spec_norm, sigma_norm, cfit = AFS_continuum_norm_1order(wv_vetted, spec_vetted, \
                                                                sigma_vetted, wv_to, spec_to, sigma_to)

        spec_norm_1star = np.vstack((spec_norm_1star, spec_norm))
        sigma_norm_1star = np.vstack((sigma_norm_1star, sigma_norm))
        cfit_1star = np.vstack((cfit_1star, cfit))

    spec_norm_1star = spec_norm_1star[1:]
    sigma_norm_1star = sigma_norm_1star[1:]
    cfit_1star = cfit_1star[1:]

    return spec_norm_1star, sigma_norm_1star, cfit_1star


def remove_cosmic_rays_1order(wv1, spec1, sigma1, qs=0.98, qlim=0.99):

    Delta_L = np.diff(spec1)
    Q_qs = np.quantile(Delta_L, qs)
    Q_j_minus_1 = np.quantile(Delta_L, qlim)
    while Q_qs < Q_j_minus_1:

        keep_pix = np.where(abs(Delta_L) <= Q_j_minus_1)[0] + 1
        wv1 = wv1[keep_pix]
        spec1 = spec1[keep_pix]
        sigma1 = sigma1[keep_pix]

        Delta_L = np.diff(spec1)
        Q_j = np.quantile(Delta_L, qlim)
        Q_j_minus_1 = Q_j

    print('%i pixels kept after cosmic ray masking' %(len(wv1)))
    return wv1, spec1, sigma1


def get_AS_tilde(wv, x, y):

    # extract only y vertices that correspond to the maximum y in the alpha shape

    AS_tilde = np.zeros(2)
    for x_at_point in wv:
        ymax = 0

        # check to find highest point in the alpha shape at that wavelength
        for j in range(0, len(x)-1):
            if x[j] == x_at_point:
                if y[j] > ymax:
                    ymax = y[j]
            elif x[j] < x_at_point:
                if x[j+1] > x_at_point:
                    # get new y value by fitting that line
                    m, b = np.polyfit([x[j], x[j+1]], [y[j], y[j+1]], 1)
                    y_fit = (m*x_at_point) + b
                    if y_fit > ymax:
                        ymax = y_fit
                    else:
                        pass
        AS_tilde = np.vstack((AS_tilde, np.array([x_at_point, ymax])))

    # Get rid of initialization zeroes
    AS_tilde = AS_tilde[1:]

    return AS_tilde


def local_polynomial_regr(wv_inp, flux_inp, wv_to, poly_deg=2, m0=0.25):

    # Number of nearby pixels to include
    num_pix = int(len(wv_inp) * m0)

    flux_regr = np.array([])
    for wv_val in wv_to:

        # Subtract off current wavelength before completing fit
        wv_norm = wv_inp - wv_val

        # Pull out the num_pix nearby pixels for use in the fit
        nearby_wv_inds = np.argsort(abs(wv_norm))[:num_pix]
        wv_nearby, flux_nearby = wv_norm[nearby_wv_inds], flux_inp[nearby_wv_inds]

        # Get weights for polynomial fit
        weight_inp = abs(wv_nearby)/np.max(abs(wv_nearby))
        w = K_weight(weight_inp)

        # Complete the fit
        p = np.poly1d(np.polyfit(wv_nearby, flux_nearby, deg=poly_deg, w=w))

        # The intercept is an approximation to the estimate at the given lambda
        flux_regr = np.append(flux_regr, p(0))

    return flux_regr



def K_weight(x):

    return (1 - (x*x*x))**3.


def get_S_alpha(y1, W_alpha, wv, spec, q):

    S_alpha = np.zeros(2)
    for j in range(0, len(W_alpha)-1):

        # find y1 values that fall in the window
        min_wv_window, max_wv_window = W_alpha[j][0], W_alpha[j+1][0]
        wv_inwindow = wv[(wv >= min_wv_window) & (wv <= max_wv_window)]
        spec_inwindow = spec[(wv >= min_wv_window) & (wv <= max_wv_window)]
        y1_inwindow = y1[(wv >= min_wv_window) & (wv <= max_wv_window)]

        # find values above the q quantile in that window
        quant = np.quantile(y1_inwindow, q)
        spec_cont = spec_inwindow[(y1_inwindow > quant)]
        wv_cont = wv_inwindow[(y1_inwindow > quant)]

        points_cont = np.transpose(np.vstack((wv_cont, spec_cont)))
        S_alpha = np.vstack((S_alpha, points_cont))

    S_alpha = np.unique(S_alpha[1:], axis=0) # remove initialization zeroes and duplicates

    return S_alpha


def plot_alpha_shape_fit(wv, spec, alpha_shape, AS_tilde, W_alpha):

    # Handle different potential types of polygons
    lines = []
    boundary = alpha_shape.boundary
    if boundary.type == 'MultiLineString':
        for line in boundary:
            lines.append(line)
    else:
        lines.append(boundary)

    # Plot alpha shape and vertices
    fig, ax = plt.subplots()
    ax.plot(wv, spec)
    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
    for point in lines[0].coords:
      x, y = point
      ax.scatter(x, y, color='k')

    # Plot AS_tilde
    ax.plot(np.transpose(AS_tilde)[0], np.transpose(AS_tilde)[1], color='gray', label=r'$\tilde{AS}$')
    ax.scatter(np.transpose(W_alpha)[0], np.transpose(W_alpha)[1], marker='o', color='red', label=r'$W_{\alpha}$')
    ax.set_xlabel('wavelength (\AA)')
    ax.set_ylabel('normalized flux')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    run()
