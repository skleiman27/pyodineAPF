"""
    Set here all the important parameters for the I2 reduction pipeline, both
    modeling of individual observations as well as the template creation.
    
    Paul Heeren, 3/02/2021
"""

from pyodine import models

import logging
import os
import sys


utilities_dir_path = os.path.dirname(os.path.realpath(__file__))

###############################################################################
## Here we define the instrument-specific setup of the LSFs. These values can #
## then be used to change the class variables of the respective LSF models.   #
###############################################################################

# For the MultiGaussian model: 11 positions & sigmas for the central Gauss and
# the satellites
_multigauss_setup_dict = {
        'positions': [-2.9, -2.5, -1.9, -1.4, -1.0, 0.0, 1.0, 1.4, 1.9, 2.5, 2.9],
        'sigmas':    [ 0.9,  0.9,  0.9,  0.9,  0.9, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9]
        }


class Parameters:
    """The control commands for the main routine
    
    The exact details of the algorithm are defined entirely by the parameters
    in this class: Parameters for chunk creation, general model parameters,
    and details about how many runs are used in the modelling and which LSF
    models are employed (and more).
    
    Furthermore, in the class method :method:'constrain_parameters' you can
    specify and alter input parameter descriptions for the model, e.g. set
    bounds or fix parameters.
    """
    
    def __init__(self):
        
        # Setup the logging if not existent yet
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                                format='%(message)s')
        
        # General parameters:
        self.osample_obs = 6                    # Oversample factor for the observation modeling
        self.lsf_conv_width = 6.                # LSF is evaluated over this many pixels (times 2)
        self.number_cores = 12                  # Number of processor cores for multiprocessing
        
        self.log_config_file = os.path.join(utilities_dir_path, 'logging.json')   # The logging config file
        self.log_level = logging.INFO           # The logging level used for console and info file
        
        self.use_progressbar = False            # Use a progressbar during chunk modelling?
        
        # Tellurics:
        self.telluric_mask = None               # Telluric mask to use (carmenes, uves or hitran); 
                                                # (None: tellurics are not taken care of)
        self.tell_wave_range = (None,6500)      # Load tellurics only within this wavelength range
        self.tell_dispersion = 0.002            # Dispersion (i.e. wavelength grid) of telluric mask
        
        # Chunking:
        # The wave_defined chunking algorithm is used by default, where the chunks are shifted with
        # respect to the template chunks to account for the change in barycentric velocity. Supply a
        # different value to delta_v in order to define the shift yourself (e.g. 0 for solar observations).
        self.order_range = (None,None)          # Order range (min,max) to use in observation modeling;
                                                # (None,None) uses automatically the same as in the template
        self.chunk_width = 91                   # Width of chunks in pixels in observation modeling
        self.chunk_padding = 6                  # Padding (left and right) of the chunks in pixels
        self.chunks_per_order = None            # Maximum number of chunks per order (optional)
        self.chunk_delta_v = 10.                # Velocity shift between template and observation 
                                                # (None: relative barycentric velocity)
        
        # Reference spectrum to use in normalizer and for the first velocity guess
        self.ref_spectrum = 'arcturus'          # Reference spectrum ('arcturus' or 'sun')
        self.velgues_order_range = (4,17)       # Orders used for velocity guess (should be outside I2 region)
        self.delta_v = 1000.                    # The velocity step size for the cross-correlation (in m/s)
        self.maxlag  = 500                      # The number of steps to each side in the cross-correlation
        
        # Normalize chunks in the beginning?
        self.normalize_chunks = False
        
        # Weighting of pixels:
        self.bad_pixel_mask = False             # Whether to run the bad pixel mask
        self.bad_pixel_cutoff = 0.22            # Cutoff parameter for the bad pixel mask
        self.correct_obs = False                # Whether to correct the observation in regions of weight = 0
        self.weight_type = 'flat'               # Type of weights (flat or lick, as implemented in pyodine.components.Spectrum)
        
        # I2 atlas:
        self.i2_to_use = 1                      # Index of I2 FTS to use (see archive/conf.py)
        self.wavelength_scale = 'air'           # Which wavelength scale to use ('air' or 'vacuum' - should always be the first)
        
        # Now to the run info: For each modelling run, define a new entry in the following dictionary
        # with all the neccessary information needed
        # (except fitting parameters, those are defined further below in constrain_parameters())
        self.model_runs = {
                0:
                {# First define the LSF
                 'lsf_model': models.lsf.SingleGaussian,    # LSF model to use (this is absolutely neccessary)
                 
                 # Before the chunks are modeled, you can smooth the wavelength guesses for the chunks
                 # over the orders with polynomials (in order to use the smoothed values as input for the run)
                 # (probably only makes sense before first run, later use smoothed results from previous runs):
                 'pre_wave_slope_deg': 3,                   # Polynomial degree of dispersion fitting (None or 0: no fitting)
                 'pre_wave_intercept_deg': 3,               # Same as above, for wavelength intercept (None or 0: no fitting)
                 # Fitting keywords
                 'use_chauvenet_pixels': True,              # Chauvenet criterion for pixel outliers? (None: False)
                 
                 # Save the fit results from this run?
                 # You can also define the filetype:
                 #    - 'h5py': Saves the most important results to hdf5 (small filesize, harder to recover)
                 #    - 'dill': Saves the whole object structure to pickle (large filesize, easy to recover)
                 'save_result': False,                       # Save the result of this run (None: True)
                 'save_filetype': 'h5py',                   # Filetype to save in (None: 'h5py')
                 # After the chunks have been modeled, you can model the wavelength results for the chunks
                 # over the orders with polynomials (in order to use the smoothed values as input for next run):
                 'wave_slope_deg': 3,                       # Polynomial degree of dispersion fitting (None or 0: no fitting)
                 'wave_intercept_deg': 3,                   # Same as above, for wavelength intercept (None or 0: no fitting)
                 # Plotting keywords
                 'plot_success': True,                      # Create plot of fitting success (None: False)
                 'plot_analysis': True,                     # Create analysis plots (residuals etc.) (None: False)
                 'plot_chunks': [150, 250, 400],            # A list with indices of chunks that will be plotted and saved
                 'plot_lsf_pars': True,                     # Plot lsf parameter results (None: False)
                 
                 # Median parameter results
                 'save_median_pars': True,                  # Save median results to text file (None: False)
                 },
                
                1:
                {# First define the LSF
                 'lsf_model': models.lsf.MultiGaussian,     # LSF model to use (this is absolutely neccessary)
                 'lsf_setup_dict': _multigauss_setup_dict,  # The instrument-specific LSF setup parameters
                 
                 # Before the chunks are modeled, you can smooth the wavelength guesses for the chunks
                 # over the orders with polynomials (in order to use the smoothed values as input for the run)
                 # (probably only makes sense before first run, later use smoothed results from previous runs):
                 'pre_wave_slope_deg': 0,                   # Polynomial degree of dispersion fitting (None or 0: no fitting)
                 'pre_wave_intercept_deg': 0,               # Same as above, for wavelength intercept (None or 0: no fitting)
                 # Fitting keywords
                 'use_chauvenet_pixels': True,              # Chauvenet criterion for pixel outliers? (None: False)
                 
                 # Save the fit results from this run?
                 # You can also define the filetype:
                 #    - 'h5py': Saves the most important results to hdf5 (small filesize, harder to recover)
                 #    - 'dill': Saves the whole object structure to pickle (large filesize, easy to recover)
                 'save_result': True,                       # Save the result of this run (None: True)
                 'save_filetype': 'h5py',                   # Filetype to save in (None: 'h5py')
                 # After the chunks have been modeled, you can model the wavelength results for the chunks
                 # over the orders with polynomials (in order to use the smoothed values as input for next run):
                 'wave_slope_deg': 3,                       # Polynomial degree of dispersion fitting (None or 0: no fitting)
                 'wave_intercept_deg': 3,                   # Same as above, for wavelength intercept (None or 0: no fitting)
                 # Plotting keywords
                 'plot_success': True,                      # Create plot of fitting success (None: False)
                 'plot_analysis': True,                     # Create analysis plots (residuals etc.) (None: False)
                 'plot_chunks': [150, 250, 400],            # A list with indices of chunks that will be plotted and saved
                 'plot_lsf_pars': True,                     # Plot lsf parameter results (None: False)
                 
                 # Median parameter results
                 'save_median_pars': True,                  # Save median results to text file (None: False)
                 }}
        """,
                
                2:
                {# First define the LSF
                 'lsf_model': models.lsf.FixedLSF,          # LSF model to use (this is absolutely neccessary)
                 # For fixed lsf consisting of smoothed lsf results from previous runs,
                 # define the smoothing parameters here:
                 'smooth_lsf_run': 1,                       # Smooth lsfs from this run (None: last run)
                 'smooth_pixels': 160,                      # Pixels (in dispersion direction) to smooth over
                 'smooth_orders': 3,                        # Orders (in cross-disp direction) to smooth over
                 'order_separation': 15,                    # Avg. pixels between orders in raw spectrum
                 'smooth_manual_redchi': False,             # If true, calculate smooth weights from manual redchi2
                                                            # (otherwise: the lmfit redchi2)
                 'smooth_osample': 0,                       # Oversampling to use in smoothing 
                                                            # (None or 0: use the oversampling from the model)
                 
                 # Before the chunks are modeled, you can smooth the wavelength guesses for the chunks
                 # over the orders with polynomials (in order to use the smoothed values as input for the run)
                 # (probably only makes sense before first run, later use smoothed results from previous runs):
                 'pre_wave_slope_deg': 0,                   # Polynomial degree of dispersion fitting (None: 0, no fitting)
                 'pre_wave_intercept_deg': 0,               # Same as above, for wavelength intercept (None: 0, no fitting)
                 # Fitting keywords
                 'use_chauvenet_pixels': True,              # Chauvenet criterion for pixel outliers? (None: False)
                 
                 # Save the fit results from this run?
                 # You can also define the filetype:
                 #    - 'h5py': Saves the most important results to hdf5 (small filesize, harder to recover)
                 #    - 'dill': Saves the whole object structure to pickle (large filesize, easy to recover)
                 'save_result': True,                       # Save the result of this run (None: True)
                 'save_filetype': 'dill',                   # Filetype to save in (None: 'h5py')
                 # After the chunks have been modeled, you can model the wavelength results for the chunks
                 # over the orders with polynomials (in order to use the smoothed values as input for next run):
                 'wave_slope_deg': 3,                       # Polynomial degree of dispersion fitting (None or 0: no fitting)
                 'wave_intercept_deg': 3,                   # Same as above, for wavelength intercept (None or 0: no fitting)
                 # Plotting keywords
                 'plot_success': True,                      # Create plot of fitting success (None: False)
                 'plot_analysis': True,                     # Create analysis plots (residuals etc.) (None: False)
                 'plot_chunks': [150, 250, 400],            # A list with indices of chunks that will be plotted and saved
                 'plot_lsf_pars': True,                     # Plot lsf parameter results (None: False)
                 
                 # Median parameter results
                 'save_median_pars': True,                  # Save median results to text file (None: False)
                 }
                
        }"""
        

    def constrain_parameters(self, lmfit_params, run_id, run_results, fitter):
        """Constrain the lmfit_parameters for the models, however you wish!
        
        :params lmfit_params: A list of :class:'lmfit.Parameters' objects
            for each chunk.
        :type lmfit_params: list[:class:`lmfit.Parameters`]
        :params run_id: The current run_id, to allow different definitions
            from run to run.
        :type run_id: int
        :params run_results: Dictionary with important observation info and
            results from previous modelling runs.
        :type run_results: dict
        :params fitter: The fitter used in the modelling.
        :type fitter: :class:`LmfitWrapper`
        
        :return: The updated list of :class:`lmfit.Parameters` objects.
        :rtype: list[:class:`lmfit.Parameters`]
        """
        
        logging.info('')
        logging.info('Constraining parameters for RUN {}'.format(run_id))
        
        ###########################################################################
        # RUN 0
        # Mainly there for first wavelength solution to feed into the next runs.
        # Using a Single-Gaussian model.
        ###########################################################################
        if run_id == 0:
            # Loop over the chunks
            for i in range(len(lmfit_params)):
                # Set velocity to velocity guess (from the reference spectrum)
                lmfit_params[i]['velocity'].set(
                        value=run_results[run_id]['velocity_guess'])
                
                # SingleGaussian model - just constrain the lsf_fwhm
                lmfit_params[i]['lsf_fwhm'].set(
                        value=2.0, min=0.5, max=4.0)
                
                # Constrain the iodine to not become negative (just in case)
                lmfit_params[i]['iod_depth'].set(min=0.1)
                
                # If the chunks were normalized beforehand:
                # Fix the continuum slope to 0
                #lmfit_params[i]['cont_slope'].set(
                #        value=0., vary=False)
        
        ###########################################################################
        # RUN 1
        # A full model now, with the full LSF description (constrained?!).
        # Fit the median LSF result from the first run to find good starting
        # values for this run's LSF (and possibly define bounds).
        # Use (smoothed) wavelength results and continuum results from run 0 
        # as starting values.
        ###########################################################################
        elif run_id == 1:
            # Dictionary of median lsf parameters from previous run
            median_lsf_pars = run_results[0]['median_pars'].filter('lsf')  #{p[4:]: run_results[0]['median_pars'][p] for p in run_results[0]['median_pars'] if 'lsf' in p}
            # Fit the lsf from last run to get good starting parameters
            lsf_fit_pars = fitter.fit_lsfs(self.model_runs[0]['lsf_model'], median_lsf_pars)
            
            logging.info('')
            logging.info('Fitted LSF parameters:')
            logging.info(lsf_fit_pars)
            
            # Loop over the chunks
            for i in range(len(lmfit_params)):
                # Set velocity to median velocity from last run
                lmfit_params[i]['velocity'].set(
                        value=run_results[0]['median_pars']['velocity']) #run_results[run_id]['velocity_guess'])
                # Set iodine and template depth to median results from last run
                lmfit_params[i]['iod_depth'].set(
                        value=run_results[0]['median_pars']['iod_depth'])
                lmfit_params[i]['tem_depth'].set(
                        value=run_results[0]['median_pars']['tem_depth'])
                
                # Wavelength dispersion: use results from before
                lmfit_params[i]['wave_slope'].set(
                        value=run_results[0]['results'][i].params['wave_slope']) #run_results[0]['wave_slope_fit'][i])#), #,
                        #min=run_results[0]['wave_slope_fit'][i]*0.99,
                        #max=run_results[0]['wave_slope_fit'][i]*1.01)
                
                # Wavelength intercept: use results from before
                lmfit_params[i]['wave_intercept'].set(
                        value=run_results[0]['results'][i].params['wave_intercept']) #run_results[0]['wave_intercept_fit'][i])#,
                        #min=run_results[0]['wave_intercept_fit'][i]*0.99,
                        #max=run_results[0]['wave_intercept_fit'][i]*1.01)
                
                # Continuum parameters: use results from before
                lmfit_params[i]['cont_intercept'].set(
                        value=run_results[0]['results'][i].params['cont_intercept'])
                lmfit_params[i]['cont_slope'].set(
                        value=run_results[0]['results'][i].params['cont_slope'])
                # If the chunks were normalized beforehand:
                # Fix the continuum slope to 0
                #lmfit_params[i]['cont_slope'].set(
                #        value=0., vary=False)
                
                # Set the LSF parameters to the fitted values,
                # and constrain them (otherwise crazy things can happen)
                for p in lsf_fit_pars.keys():
                    lmfit_params[i]['lsf_'+p].set(
                        value=lsf_fit_pars[p],
                        min=lsf_fit_pars[p]-abs(lsf_fit_pars[p])*0.4,
                        max=lsf_fit_pars[p]+abs(lsf_fit_pars[p])*0.4)
        """
        ###########################################################################
        # RUN 2
        # Final run. Use smoothed LSF results from run 1 and keep them fixed.
        # Only vary other parameters (use results from run 1 as starting values).
        ###########################################################################
        elif run_id == 2:
            # Loop over the chunks
            for i in range(len(lmfit_params)):
                # Set velocity to median velocity from last run
                lmfit_params[i]['velocity'].set(
                        value=run_results[1]['median_pars']['velocity']) #run_results[run_id]['velocity_guess'])
                # Set iodine and template depth to median results from last run
                lmfit_params[i]['iod_depth'].set(
                        value=run_results[1]['median_pars']['iod_depth'])
                lmfit_params[i]['tem_depth'].set(
                        value=run_results[1]['median_pars']['tem_depth'])
                
                # Wavelength dispersion: use results from before
                lmfit_params[i]['wave_slope'].set(
                        value=run_results[1]['wave_slope_fit'][i]),
                        #min=run_results[1]['wave_slope_fit'][i]*0.99,
                        #max=run_results[1]['wave_slope_fit'][i]*1.01)
                
                # Wavelength intercept: use results from before
                lmfit_params[i]['wave_intercept'].set(
                        value=run_results[1]['wave_intercept_fit'][i])
                        #min=run_results[1]['wave_intercept_fit'][i]*0.96,
                        #max=run_results[1]['wave_intercept_fit'][i]*1.04)
                        
                # Continuum parameters: use results from before
                lmfit_params[i]['cont_intercept'].set(
                        value=run_results[1]['results'][i].params['cont_intercept'])
                lmfit_params[i]['cont_slope'].set(
                        value=run_results[1]['results'][i].params['cont_slope'])
                # If the chunks were normalized beforehand:
                # Fix the continuum slope to 0
                #lmfit_params[i]['cont_slope'].set(
                #        value=0., vary=False)
                
                # For fixed, smoothed LSF: Don't vary order and pixel0 values!
                # (and better also not the amplitude)
                lmfit_params[i]['lsf_order'].vary = False
                lmfit_params[i]['lsf_pixel0'].vary = False
                lmfit_params[i]['lsf_amplitude'].vary = False
        """
        return lmfit_params
    
