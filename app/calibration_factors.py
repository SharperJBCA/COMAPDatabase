import numpy as np
import h5py
import matplotlib.pyplot as plt
import os 
from .create_database import update_entry, get_column_values, create_calibration_database
from .CaliModels import TauAFluxModel, CasAFluxModel, JupiterFluxModel, CalibratorList
from tqdm import tqdm 
from astropy.time import Time
import pandas as pd 

calibrator_models = {'TauA':TauAFluxModel(), 
                     'CasA':CasAFluxModel(), 
                     'jupiter':JupiterFluxModel(),
                     'CygA':lambda *args, **kwargs: 1}

def get_calibration_factor(filename, overwrite=False): 
                
    data = h5py.File(filename,'r')
    if not 'comap' in data:
        return None 
    if not data['comap'].attrs['source'].split(',')[0] in CalibratorList.keys():
        return None
    source = data['comap'].attrs['source'].split(',')[0]
    obsid = data['comap'].attrs['obsid']

    if not f'{source}_source_fit' in data:
        return None

    feeds = data['spectrometer/feeds'][...]-1
    src = np.zeros((20, 4, 7)) 
    src[feeds] = data[f'{source}_source_fit']['fits'][...]
    err = np.zeros((20, 4, 7)) 
    err[feeds] = data[f'{source}_source_fit']['errors'][...]
    el  = np.zeros((20)) 
    el[feeds] = np.nanmedian(data['spectrometer/pixel_pointing/pixel_el'][...],axis=-1)
    mjd = data['spectrometer/MJD'][0]
    data.close()

    return {'fits':src, 'errors':err, 'elevation':el, 'mjd':mjd, 'source':source,'obsid':obsid}


def flux_error(fits, error, frequency): 
    
    kb = 1.38e-23 
    c  = 2.99792458e8
    conv = 2*kb * (frequency/c)**2 * 1e26 * (np.pi/180.)**2    
    beam = 2*np.pi * fits[1]*fits[2]*conv
    flux = fits[0] * beam 
    sigma_out =  flux**2 * ((error[0]/fits[0])**2 +(error[1]/fits[1])**2 + (error[2]/fits[2])**2)

    return sigma_out**0.5

def get_source_flux(frequency : float, 
                    fits : np.ndarray[float,float], 
                    errors : np.ndarray[float,float]):
    """Calculates the flux density given gaussian fit parameters""" 
    
    kb = 1.38e-23 
    nu = frequency * 1e9
    c  = 2.99792458e8
    conv = 2*kb * (nu/c)**2 * 1e26 * (np.pi/180.)**2 
    if fits.ndim == 1:
        flux = fits[0] * 2*np.pi*fits[4]*fits[5] * conv
        flux_errs = flux_error(fits[[0,4,5]],errors[[0,4,5]], nu)
    else:
        flux = 2*np.pi*fits[:,0]*fits[:,4]*fits[:,5] * conv 
        #print('AMPLITUDE', nu*1e-9, fits.shape, fits[:,0])
        #print('FLUX',np.nanmedian(flux))
        #print('BEAM AREA', np.nanmedian(2*np.pi*fits[:,4]*fits[:,5]) * (np.pi/180.)**2)
        #print('WIDTHS', np.nanmedian(fits[:,4])*60*2.355, np.nanmedian(fits[:,5])*60*2.355)
        flux_errs = np.array([flux_error(fits[i,[0,4,5]],errors[i,[0,4,5]], nu) if fits[i,4] !=0 else 0 for i in range(flux.size)]) 
                
    return flux, flux_errs 

def get_source_geometric_radius(fits : np.ndarray[float,float], 
                                errors : np.ndarray[float,float]):
    """Calculates the geometric radius given gaussian fit parameters""" 
    
    if fits.ndim==1:
        radius = np.sqrt(fits[4]**2 + fits[5]**2)
        radius_err = (fits[4]/radius)**2 * errors[4]**2 +\
            (fits[5]/radius)**2 * errors[5]**2
    else:
        radius = np.sqrt(fits[:,4]**2 + fits[:,5]**2)
        radius_err = (fits[:,4]/radius)**2 * errors[:,4]**2 +\
            (fits[:,5]/radius)**2 * errors[:,5]**2
    radius_err = radius_err**0.5 
    
    return radius, radius_err

def create_source_mask(flux, flux_err, radius, radius_err, cali_factor, 
                        max_cali_factor=1,
                        min_cali_factor=0.5,
                        min_flux = 10,
                        max_flux=1000,
                        max_flux_err = 10, 
                        min_flux_err = 0.5,
                        max_geo_radius_diff=1e-3):
    """Calculate the mask for the bad SOURCE fits"""
    mask = (flux_err > max_flux_err) | ~np.isfinite(flux) |\
            ~np.isfinite(flux_err) |\
            (cali_factor > max_cali_factor) |\
            (cali_factor < min_cali_factor)

    return mask 


def get_source_mask(ifeed, h, mjd, frequency=27e9,iband=0):
    calibrator_source = h['comap'].attrs['source'].split(',')[0] #.decode('utf-8')
    data = h[f'{calibrator_source}_source_fit']
    flux_model = calibrator_models[calibrator_source]

    flux_feed1, flux_err_feed1 = get_source_flux(frequency, 
                                        data['fits'][ifeed,iband,:],
                                        data['errors'][ifeed,iband,:])
    radius_feed1, radius_err_feed1 = get_source_geometric_radius(data['fits'][ifeed,iband,:],
                                                        data['errors'][ifeed,iband,:])
    
    mask_feed1 = create_source_mask(flux_feed1, flux_err_feed1, radius_feed1, radius_err_feed1, flux_feed1/flux_model(frequency, mjd))

    return mask_feed1


def create_calibration_factors(database_name):

    filelist = get_column_values(database_name, 'level2_file_path')

    all_data = {'mjd':[], 'elevation':[], 'fits':[], 'errors':[], 'source':[],'obsid':[]}
    for filename in tqdm(filelist, desc='Reading Calibrator Data'): 
        data = get_calibration_factor(filename)
        if data is None:
            continue

        # Append data to all data 
        all_data['mjd'].append(data['mjd'])
        all_data['elevation'].append(data['elevation'])
        all_data['fits'].append(data['fits'])
        all_data['errors'].append(data['errors'])
        all_data['source'].append(data['source'])
        all_data['obsid'].append(data['obsid'])
    for key in all_data.keys():
        all_data[key] = np.array(all_data[key])

    # LOOP THROUGH EACH SOURCE
    for source in tqdm(['TauA','CasA','jupiter'],desc='Updating Database with Cal Factors'):
        idx = np.where(all_data['source'] == source)[0]
        if len(idx) == 0:
            continue
        flux_model = calibrator_models[source]
        fit = all_data['fits'][idx]
        error= all_data['errors'][idx]
        mjd = all_data['mjd'][idx]
        elevation = all_data['elevation'][idx]
        obsids = all_data['obsid'][idx]

        full_mask = np.zeros((fit.shape[0],20,4),dtype=bool)
        cal_factors= np.zeros((fit.shape[0],20,4))
        cal_errors = np.zeros((fit.shape[0],20,4))

        # CALCULATE THE GOOD CAL FACTORS 
        for iband,frequency in zip([0,1,2,3],[27,29,31,33]):
            flux_feed1, flux_err_feed1 = get_source_flux(frequency, 
                                        fit[:,0,iband,:],
                                        error[:,0,iband,:])
            radius_feed1, radius_err_feed1 = get_source_geometric_radius(fit[:,0,iband,:],
                                                                        error[:,0,iband,:])
            mask_feed1 = create_source_mask(flux_feed1, 
                                            flux_err_feed1, 
                                            radius_feed1, 
                                            radius_err_feed1, 
                                            flux_feed1/flux_model(frequency, mjd))
            for ifeed in range(20):

                flux, flux_err = get_source_flux(frequency, 
                                                 fit[:,ifeed,iband,:],
                                                 error[:,ifeed,iband,:])
                radius, radius_err = get_source_geometric_radius(fit[:,ifeed,iband,:],
                                                                 error[:,ifeed,iband,:])
                mask = create_source_mask(flux, 
                                        flux_err, 
                                        radius, 
                                        radius_err,
                                        flux/flux_model(frequency, mjd),
                                        max_flux_err = 2)

                full_mask[:,ifeed,iband] = mask | mask_feed1 
                cal_factors[:,ifeed,iband] = flux/flux_model(frequency, data['mjd'])
                cal_errors[:,ifeed,iband] = flux_err/flux_model(frequency, data['mjd'])
                
        # UPDATE THE SQL TABLE 
        create_calibration_database(f'{source}CalibrationFactors', database_name)
        for i, obsid in enumerate(tqdm(obsids)):
            update_entry(database_name, f'{source}CalibrationFactors',
                         obsid,
                         'calibration_factor_band0',
                            cal_factors[i,:,0])
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'calibration_factor_error_band0',
                                cal_errors[i,:,0])
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'calibration_factor_band1',
                                cal_factors[i,:,1])
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'calibration_factor_error_band1',
                                cal_errors[i,:,1])
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'calibration_factor_band2',
                                cal_factors[i,:,2])
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'calibration_factor_error_band2',
                                cal_errors[i,:,2])
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'calibration_factor_band3',
                                cal_factors[i,:,3])
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'calibration_factor_error_band3',
                                cal_errors[i,:,3])
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'good_flag',
                                np.max(full_mask[i,:,:].astype(float),axis=-1))
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'elevation',
                                elevation[i])
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'mjd',
                                mjd[i])
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'source_name',
                                source)
            update_entry(database_name, f'{source}CalibrationFactors',
                            obsid,
                            'obs_id',
                                obsid)
            
def plot_calibration_factors(database_name, figure_path='outputs', source='CasA'):
    os.makedirs(figure_path, exist_ok=True)
    cal_factor_band0 = get_column_values(database_name, 'calibration_factor_band0', table=f'{source}CalibrationFactors')
    cal_factor_band1 = get_column_values(database_name, 'calibration_factor_band1', table=f'{source}CalibrationFactors')
    cal_factor_band2 = get_column_values(database_name, 'calibration_factor_band2', table=f'{source}CalibrationFactors')
    cal_factor_band3 = get_column_values(database_name, 'calibration_factor_band3', table=f'{source}CalibrationFactors')
    good_flag = get_column_values(database_name, 'good_flag', table=f'{source}CalibrationFactors') == 0 

    mjd = np.array(get_column_values(database_name, 'mjd', table=f'{source}CalibrationFactors'))
    time = Time(mjd, format='mjd')

    print(time.shape, cal_factor_band0.shape)
    print(good_flag.shape)
    fig, ax = plt.subplots(4,1,figsize=(10,10),sharex=True)
    for ifeed in range(20):
        df = pd.DataFrame({'time':time.datetime[good_flag[:,ifeed]],
                            'band0':cal_factor_band0[good_flag[:,ifeed],ifeed],
                            'band1':cal_factor_band1[good_flag[:,ifeed],ifeed],
                            'band2':cal_factor_band2[good_flag[:,ifeed],ifeed],
                            'band3':cal_factor_band3[good_flag[:,ifeed],ifeed]})
        df = df.set_index('time')
        df = df.sort_index()
        # apply a running mean in time to each band 
        df['band0'] = df['band0'].rolling(10).median()
        df['band1'] = df['band1'].rolling(10).median()
        df['band2'] = df['band2'].rolling(10).median()
        df['band3'] = df['band3'].rolling(10).median()
        # plot the calibration factors
        df['band0'].plot(ax=ax[0],label='band0',marker='.',lw=0)
        df['band1'].plot(ax=ax[1],label='band1',marker='.',lw=0)
        df['band2'].plot(ax=ax[2],label='band2',marker='.',lw=0)
        df['band3'].plot(ax=ax[3],label='band3',marker='.',lw=0)
    ax[0].set_ylabel('Calibration Factor')
    ax[1].set_ylabel('Calibration Factor')
    ax[2].set_ylabel('Calibration Factor')
    ax[3].set_ylabel('Calibration Factor')
    ax[3].set_xlabel('Time')
    fig.savefig(f'{figure_path}/{source}_calibration_factors.png')
    plt.close(fig)

def apply_calibration_factors(database_name, source='CasA'):
    """Loop over all the level 2 files and apply the nearest calibration factor"""
    filelist = get_column_values(database_name, 'level2_file_path')

    cal_factor_band0 = get_column_values(database_name, 'calibration_factor_band0', table=f'{source}CalibrationFactors')
    cal_factor_band1 = get_column_values(database_name, 'calibration_factor_band1', table=f'{source}CalibrationFactors')
    cal_factor_band2 = get_column_values(database_name, 'calibration_factor_band2', table=f'{source}CalibrationFactors')
    cal_factor_band3 = get_column_values(database_name, 'calibration_factor_band3', table=f'{source}CalibrationFactors')
    mjd = np.array(get_column_values(database_name, 'mjd', table=f'{source}CalibrationFactors'))
    time = Time(mjd, format='mjd')
    good_flag = get_column_values(database_name, 'good_flag', table=f'{source}CalibrationFactors') == 0 

    feeds = [] 
    for ifeed in range(20):
        df = pd.DataFrame({'time':time.datetime[good_flag[:,ifeed]],
                            'band0':cal_factor_band0[good_flag[:,ifeed],ifeed],
                            'band1':cal_factor_band1[good_flag[:,ifeed],ifeed],
                            'band2':cal_factor_band2[good_flag[:,ifeed],ifeed],
                            'band3':cal_factor_band3[good_flag[:,ifeed],ifeed]})
        df = df.set_index('time')
        df = df.sort_index()
        # apply a running mean in time to each band 
        df['band0'] = df['band0'].rolling(10).median()
        df['band1'] = df['band1'].rolling(10).median()
        df['band2'] = df['band2'].rolling(10).median()
        df['band3'] = df['band3'].rolling(10).median()
        feeds.append(df)

    nfeeds = 19
    for filename in tqdm(filelist, desc=f'Adding {source} Calibration Factors to Level 2'):
        if not os.path.exists(filename):
            continue    
        h = h5py.File(filename,'r+')
        if not 'comap' in h:
            h.close()
            continue
    
        this_mjd = h['spectrometer/MJD'][0]

        
        band0 = [df['band0'].values for df in feeds]
        band1 = [df['band1'].values for df in feeds]
        band2 = [df['band2'].values for df in feeds]
        band3 = [df['band3'].values for df in feeds]
        mjd   = [Time(df.index.values).mjd for df in feeds]
        idx = [np.argmin(np.abs(mjd[ifeed]-this_mjd)) for ifeed in range(nfeeds)]
        h['comap'].attrs[f'{source}_calibration_factor_band0'] = [band0[ifeed][idx[ifeed]] for ifeed in range(nfeeds)] + [0]
        h['comap'].attrs[f'{source}_calibration_factor_band1'] = [band1[ifeed][idx[ifeed]] for ifeed in range(nfeeds)] + [0]
        h['comap'].attrs[f'{source}_calibration_factor_band2'] = [band2[ifeed][idx[ifeed]] for ifeed in range(nfeeds)] + [0]
        h['comap'].attrs[f'{source}_calibration_factor_band3'] = [band3[ifeed][idx[ifeed]] for ifeed in range(nfeeds)] + [0]
        h.close()
