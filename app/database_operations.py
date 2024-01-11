import glob
import os
import re
import sqlite3
import h5py 
import pandas as pd 
import numpy as np
from astropy.time import Time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from .create_database import COMAPStats,insert_observation_data

def get_hdf5_files(directory):
    return glob.glob(os.path.join(directory, "**/*.hd5"), recursive=True)

def extract_obsid(filename):
    pattern = r'Level2_comap-(\d+)-\d{4}-\d{2}-\d{2}-\d{6}.hd5'
    match = re.search(pattern, filename)
    return int(match.group(1)) if match else None

def obsid_exists_in_db(database_name, obsid):
    # Connect to the database
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    # Execute the SELECT query
    cursor.execute("SELECT 1 FROM Observation WHERE obs_id=?", (obsid,))

    # Fetch the result
    result = cursor.fetchone()

    # Close the connection
    conn.close()
    # Return True if obsid exists, False otherwise
    return bool(result)

def populate_db_from_hdf5(database_name,filename, force_update=False):
    """Add a new column if it does not exist in the table"""
    obsid = extract_obsid(filename)
    if not obsid_exists_in_db(database_name, obsid) or force_update:
        comap_stats = COMAPStats(filename) 
        insert_observation_data(database_name, comap_stats.data)


def db_to_pandas(database_name, bad_value=-3123):
    """Fetch the data from the database and return a pandas DataFrame"""
    # Connect to the SQLite database
    conn = sqlite3.connect(database_name)
    
    # Fetch data
    query = "SELECT mjd, source_name, feeds, tsys, gain, ra_offset, dec_offset, flux_band0, flux_band1, flux_band2, flux_band3, cal_factor, sky_brightness, bad_observation, auto_rms, red_noise, white_noise, spectrum_noise  FROM Observation" 
    df = pd.read_sql_query(query, conn)
    
    # Close the connection
    conn.close()

    for stat in ['tsys', 'gain', 'ra_offset', 'dec_offset', 'flux_band0', 'flux_band1', 'flux_band2', 'flux_band3', 'cal_factor', 'sky_brightness', 'bad_observation', 'auto_rms', 'red_noise', 'white_noise', 'spectrum_noise']:
        df[stat] = df[stat].str.split(',').apply(lambda x: [float(v) if float(v) != bad_value else np.nan for v in x])

    for stat in ['feeds']:
        df[stat] = df[stat].str.split(',').apply(lambda x: np.arange(20)+1)

    for stat in ['mjd']:
         df[stat] = df[stat].apply(lambda x: float(x))

    df['datetime'] = df['mjd'].apply(lambda x: Time(x, format='mjd').datetime if (x != bad_value) or (x==0) else np.nan)
    df.set_index('datetime', inplace=True)

    return df

