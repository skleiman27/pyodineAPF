# %%
#%load_ext autoreload
#%autoreload 2

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
# Put in the pyodine path on your machine here!
pyodine_path = '/home/skleiman/pyodineAPF'

sys.path.append(pyodine_path)

import pyodine
import pyodine_create_templates     # <- the template creation routines
import utilities_apf as utilities


# %%
Pars = utilities.pyodine_parameters.Template_Parameters()
Pars.deconvolution_pars


# %%


# %%
#10 01 for O star, HR files

# O-star observations to use for the mod?elling
ostar_dir   = "/data/APF_reductions/HD203030/20241001/PyPyPiOUT/"
ostar_files = glob.glob(ostar_dir+"PyPyPied_In_HR7906*")
ostar_files.sort()
#ostar_files = ostar_files[0:2]
print(ostar_files)

# HD 203030 files
# Stellar observations to use for the deconvolution
temp_dir   = "/data/APF_reductions/HD203030/20241001/PyPyPiOUT/"
temp_files =  glob.glob(ostar_dir+"PyPyPied_Out_HD*")
temp_files.sort()
print(temp_files)

# Output pathname for the summed, normalized template observations
obs_sum_outname = os.path.join("/data/APF_recuctions/HD203030/20241001/summed_HD203030.fits")

# Output pathname for the template
temp_outname = os.path.join("/data/APF_recuctions/HD203030/20241001/template_HD203030.fits")

# Output directory for plots and pathnames for modelling results
os.makedirs("/data/APF_recuctions/HD203030/20241001/pyodine_outs/", exist_ok=True)
plot_dir  = "/data/APF_recuctions/HD203030/20241001/pyodine_outs/"
res_files = [os.path.join(plot_dir, 'HD203030_2024_10_01_res0.h5'),
             os.path.join(plot_dir, 'HD203030_2024_10_01_res1.pkl')]

# Log files
error_file = os.path.join(plot_dir, 'error.log')
info_file  = os.path.join(plot_dir, 'info.log')

# %%
pyodine_create_templates.create_template(utilities, Pars, ostar_files, temp_files, 
                                         temp_outname, plot_dir=plot_dir, res_files=res_files, 
                                         obs_sum_outname=obs_sum_outname, error_log=error_file, 
                                         info_log=info_file)

# %%
fit_results_1 = pyodine.fitters.results_io.load_results(res_files[1], filetype='dill')

# Also construct the original chunks array of the O/B-star observations
chunks = pyodine.components.ChunkArray()
for r in fit_results_1:
    chunks.append(r.chunk)

# %%
print('Number of chunks/fit results:', len(fit_results_1))
print('\nObject structure of each fit result:')
print(fit_results_1[0].__dict__)

# %%
chunks, fit_results_0 = pyodine.fitters.results_io.restore_results_object(
    utilities, res_files[0])

# %%
chunk_ind = 26

pyodine.plot_lib.plot_chunkmodel(fit_results_1, chunks, chunk_ind, template=False, 
                                 show_plot=True)


# %%
residuals = pyodine.plot_lib.plot_residual_hist(fit_results_1, title='Residuals histogram', 
                                                show_plot=True)

# %%
pyodine.plot_lib.plot_chunk_scatter(scatter=residuals, ylabel='Chunk residuals [%]', 
                                    title='Chunk residuals', show_plot=True)

# %%
# The LSF model, oversampling and convolution width used
lsf_model      = fit_results_1[0].model.lsf_model
osample_factor = fit_results_1[0].model.osample_factor
conv_width     = fit_results_1[0].model.conv_width

# Generate the pixel vector to evaluate the LSF over
lsf_x = lsf_model.generate_x(osample_factor=osample_factor, conv_width=conv_width)

# Now loop over all chunks, evaluate the LSFs and append them to a list
lsfs = []
for i in range(len(fit_results_1)):
    lsf_pars = fit_results_1[i].params.filter('lsf')
    lsfs.append(lsf_model.eval(lsf_x, lsf_pars))

# Finally plot a grid of 3x3 LSFs
pyodine.plot_lib.plot_lsfs_grid(lsfs, chunks, x_lsf=lsf_x, x_nr=3, y_nr=3, 
                                alpha=0.7, xlim=(-4,4), show_plot=True)

# %%
# The best-fit wavelength slopes for all chunks
wave_slopes_model = [r.params['wave_slope'] for r in fit_results_1]
# The estimated dispersion for all chunks (from the original spectrum)
wave_slopes_data = [(ch.wave[-1]-ch.wave[0])/len(ch) for ch in chunks]

pyodine.plot_lib.plot_chunk_scatter(scatter=[wave_slopes_model,wave_slopes_data], 
                                    scatter_fmt='.', scatter_label=['model', 'data'], 
                                    ylabel=r'wave_slope [$\AA$/pix]', show_plot=True)

# %%
# Chunk index
chunk_ind = 26

pyodine.plot_lib.plot_chunkmodel(fit_results_0, chunks, chunk_ind, template=False, 
                                 show_plot=True)

# %%
template = pyodine.template.base.StellarTemplate_Chunked(temp_outname)

print('The deconvolved template object:\n', template)
print('\nThe first chunk of the template:\n', template[0])

# %%
from astropy.io import fits

with fits.open(obs_sum_outname) as hdu:
    temp_obs_flux = hdu[0].data[0]      # <- ind 0: normalized flux
    temp_obs_wave = hdu[0].data[2]      # <- ind 2: wavelengths

# %%
# Chunk index
chunk_ind = 26

# The wavelength range covered by this chunk
wave_range = [template[chunk_ind].wave[0], template[chunk_ind].wave[-1]]

# Get the (order,pixel)-indices of the 
# summed observation covering this range
ind = np.where(np.logical_and(temp_obs_wave > wave_range[0], 
                              temp_obs_wave < wave_range[1]))

# And plot
fig = plt.figure(figsize=(10,6))
plt.plot(template[chunk_ind].wave, template[chunk_ind].flux, 
         alpha=0.7, label='Deconvolved template')
plt.plot(temp_obs_wave[ind], temp_obs_flux[ind], 
         alpha=0.7, label='Summed template obs.')
plt.legend()
plt.xlabel(r'Wavelength [$\AA$]')
plt.title('{}, chunk {}'.format(template.starname, chunk_ind))
plt.show()


