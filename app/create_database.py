import sqlite3
import argparse
import h5py 
import numpy as np 

from scipy import constants

# Schema
OBSERVATION_TABLE = """
CREATE TABLE IF NOT EXISTS Observation (
    obs_id INTEGER PRIMARY KEY,
    source_name TEXT NOT NULL,
    mjd REAL,
    year TEXT NOT NULL,
    month TEXT NOT NULL,
    mean_Tvane REAL,
    mean_Tshroud REAL,
    mean_az REAL,
    mean_el REAL,
    mean_wind_speed REAL,
    mean_air_temp REAL,
    mean_air_pressure REAL,
    mean_relative_humidity REAL,
    mean_wind_direction REAL,
    level2_file_path TEXT NOT NULL,
    feeds TEXT,
    tsys TEXT,
    gain TEXT,
    ra_offset TEXT,
    dec_offset TEXT,
    flux_band0 TEXT,
    flux_band1 TEXT,
    flux_band2 TEXT,
    flux_band3 TEXT,
    cal_factor TEXT,
    sky_brightness TEXT,
    spike_count TEXT,
    bad_observation TEXT,
    auto_rms TEXT,
    red_noise TEXT,
    white_noise TEXT,
    spectrum_noise TEXT
    )
"""

# Schema for calibration factor tables
CALIBRATION_TABLE = """
CREATE TABLE IF NOT EXISTS {name} (
obs_id INTEGER PRIMARY KEY,
source_name TEXT NOT NULL DEFAULT 'None',
mjd REAL DEFAULT -3123,
calibration_factor_band0 TEXT DEFAULT '-3123,',
calibration_factor_error_band0 TEXT DEFAULT '-3123,',
calibration_factor_band1 TEXT DEFAULT '-3123,',
calibration_factor_error_band1 TEXT DEFAULT '-3123,',
calibration_factor_band2 TEXT DEFAULT '-3123,',
calibration_factor_error_band2 TEXT DEFAULT '-3123,',
calibration_factor_band3 TEXT DEFAULT '-3123,',
calibration_factor_error_band3 TEXT DEFAULT '-3123,',
elevation REAL DEFAULT -3123,
good_flag TEXT DEFAULT '-3123,'
)
"""

def parse_single_bit_mask(flag):
    p = np.inf 
    bit_mask_list = []
    current_flag = flag*1
    while p != 0: 
        if current_flag == 0:
            bit_mask_list.append(0)
            break
        p = int(np.floor(np.log(current_flag)/np.log(2)))
        bit_mask_list.append(p)
        current_flag -= 2**p
    return bit_mask_list

def update_bit_mask(array, flag): 
    if not any([flag in b for b in parse_single_bit_mask(array)]):
        array += 2**flag

def parse_bit_mask(flags):
    # Return a list of lists containing the bit masks for each feed 

    bit_masks = [] 
    for i in range(flags.size):
        bit_mask_list = parse_single_bit_mask(flags[i])
        bit_masks.append(bit_mask_list)
    return bit_masks 

def get_observer_flags(observer_flags_path):
    data = np.loadtxt(observer_flags_path, dtype=str, delimiter=',', skiprows=1,usecols=[0,1,2,3,4])

    data_out = []
    for row in data:
        start,end, feeds, bands, flag = row 
        start = int(start) 
        if end.strip() == '':
            end = start
        
        if (feeds.strip() == 'all') | (feeds.strip() == 'many'):
            feeds = np.arange(1,21)
        else:
            feeds = feeds.replace('"','')
            feeds = np.array([int(feed) for feed in feeds.split(',')])
        data_out.append([start, end, feeds])

    return data_out

class COMAPStats: 
    """
    Class for reading in the stats from the level 1 and level 2 files.
    The data is stored in a dictionary that matches the Observation table schema. 
    """

    def __init__(self, filename):
        self.filename = filename
        self.data = {'level2_file_path':filename} 

        self.observer_flags = get_observer_flags('app/observer_flags.csv')
        self.read_data()

    def read_data(self):
        h = h5py.File(self.filename, 'r')
        self.data['obs_id'] = int(h['comap'].attrs['obsid'])
        self.data['level2_file_path'] = self.filename 
        self.data['source_name'] = h['comap'].attrs['source'].split(',')[0]  # Adjust accordingly
        self.data['year'] = self.filename.split('-')[3]
        self.data['month'] = self.filename.split('-')[4]
        self.data['mjd'] = -3123
        self.data['feeds'] = h['spectrometer/feeds'][:].astype(int)
        self.data['tsys'] = np.zeros(20) -3123
        self.data['gain'] = np.zeros(20) -3123
        self.data['ra_offset'] = np.zeros(20) -3123
        self.data['dec_offset'] = np.zeros(20) -3123
        self.data['flux_band0'] = np.zeros(20) -3123
        self.data['flux_band1'] = np.zeros(20) -3123
        self.data['flux_band2'] = np.zeros(20) -3123
        self.data['flux_band3'] = np.zeros(20) -3123
        self.data['cal_factor'] = np.zeros(20) -3123
        self.data['sky_brightness'] = np.zeros(20) -3123
        self.data['spike_count'] = np.zeros(20) -3123
        self.data['bad_observation'] = np.ones(20) # 0 = good, 1 = bad stats, 2 = bad instrument, 3 = bad pointing, 4 = no stats, 5 = no skydip, 6 = no vane, 7 = no source fit

        self.data['auto_rms'] = np.zeros(20) -3123
        self.data['red_noise'] = np.zeros(20) -3123
        self.data['white_noise'] = np.zeros(20) -3123
        self.data['spectrum_noise'] = np.zeros(20) -3123

        # Check if the observation is in the observer_flags 
        for (start, end, obs_feeds) in self.observer_flags:
            if self.data['obs_id'] >= int(start) and self.data['obs_id'] <= int(end):
                for obs_feed in obs_feeds:
                    update_bit_mask(self.data['bad_observation'][obs_feed-1], 2) 

        for i in range(20):
            if i+1 not in self.data['feeds']:
                if any([2 in b for b in parse_single_bit_mask(self.data['bad_observation'][i])]):
                    continue
                update_bit_mask(self.data['bad_observation'][i], 2) 

        if all([2 in b for b in parse_bit_mask(self.data['bad_observation'])]):
            return

        feeds = self.data['feeds']
        source_name = self.data['source_name']
        if 'vane' in h:
            if h['vane/system_temperature'].shape[0] == 0:
                self.data['bad_observation'][:] += 2**6
                for ibad in range(self.data['bad_observation'].size):
                    update_bit_mask(self.data['bad_observation'][ibad], 6) 
                return 
            self.data['tsys'][feeds-1] = np.nanmedian(h['vane/system_temperature'][0,:,:,:],axis=(1,2))
            self.data['gain'][feeds-1] = np.nanmedian(h['vane/system_gain'][0,:,:,:],axis=(1,2))
        else:
            for ibad in range(self.data['bad_observation'].size):
                update_bit_mask(self.data['bad_observation'][ibad], 6) 
        if f'{source_name}_source_fit' in h:
            self.data['ra_offset'][feeds-1] = np.median(h[f'{source_name}_source_fit/fits'][:,:,2])
            self.data['dec_offset'][feeds-1] = np.median(h[f'{source_name}_source_fit/fits'][:,:,3])
            source_fits = h[f'{source_name}_source_fit/fits'][...] 
            frequencies = np.array([27,29,31,33])*1e9
            C = 2* constants.k * (frequencies/constants.c)**2 * 1e26 
            flux = 2*np.pi*source_fits[:,:,0]*source_fits[:,:,4]*source_fits[:,:,5] * (np.pi/180.)**2 * C[None,:]
            self.data['flux_band0'][feeds-1] = flux[:,0]
            self.data['flux_band1'][feeds-1] = flux[:,1]
            self.data['flux_band2'][feeds-1] = flux[:,2]
            self.data['flux_band3'][feeds-1] = flux[:,3]
        if 'skydip' in h:
            self.data['sky_brightness'][feeds-1] = np.nanmedian(h['skydip/fit_values'][:,:,:,1],axis=(1,2))
        else:
            for ibad in range(self.data['bad_observation'].size):
                update_bit_mask(self.data['bad_observation'][ibad], 5) 
        if 'spikes' in h:
            pass 
        if 'spectrometer' in h:
            self.data['mjd'] = float(h['spectrometer/MJD'][0]) # MJD
        if 'fnoise_fits' in h:
            try:
                self.data['auto_rms'][:] = np.nanmedian(h['fnoise_fits']['auto_rms'][:,:,:],axis=(1,2))
                self.data['white_noise'][:] = np.nanmedian(h['fnoise_fits']['fnoise_fit_parameters'][:,:,:,0],axis=(1,2))
                self.data['red_noise'][:] = np.nanmedian(h['fnoise_fits']['fnoise_fit_parameters'][:,:,:,1],axis=(1,2))
                self.data['spectrum_noise'][:] = np.nanmedian(h['fnoise_fits']['fnoise_fit_parameters'][:,:,:,2],axis=(1,2))
            except KeyError:
                for ibad in range(self.data['bad_observation'].size):
                    update_bit_mask(self.data['bad_observation'][ibad], 4) 

def create_database(database_name=':memory:'):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute(OBSERVATION_TABLE)
    conn.commit()
    conn.close()

def create_calibration_database(table_name, database_name=':memory:'):
    print(f'Creating {table_name} table in {database_name}')
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute(CALIBRATION_TABLE.format(name=table_name))
    conn.commit()
    conn.close()

def insert_observation_data(database_name, data):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO Observation (
        obs_id, level2_file_path, mjd, source_name, year, month, level2_file_path, feeds, tsys, gain, ra_offset,
        dec_offset, flux_band0, flux_band1, flux_band2, flux_band3, cal_factor, sky_brightness, spike_count, bad_observation, auto_rms, red_noise, white_noise, spectrum_noise
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data['obs_id'], data['level2_file_path'], data['mjd'],data['source_name'], data['year'], data['month'], data['level2_file_path'],
        ','.join(map(str, data['feeds'])),  # Storing as comma-separated values
        ','.join(map(str, data['tsys'])),
        ','.join(map(str, data['gain'])),
        ','.join(map(str, data['ra_offset'])),
        ','.join(map(str, data['dec_offset'])),
        ','.join(map(str, data['flux_band0'])),
        ','.join(map(str, data['flux_band1'])),
        ','.join(map(str, data['flux_band2'])),
        ','.join(map(str, data['flux_band3'])),
        ','.join(map(str, data['cal_factor'])),
        ','.join(map(str, data['sky_brightness'])),
        ','.join(map(str, data['spike_count'])),
        ','.join(map(str, data['bad_observation'])),
        ','.join(map(str, data['auto_rms'])),
        ','.join(map(str, data['red_noise'])),
        ','.join(map(str, data['white_noise'])),
        ','.join(map(str, data['spectrum_noise'])),
    ))

    conn.commit()
    conn.close()

def get_columns_from_table(database_path, table_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    
    # Create a cursor object
    cursor = conn.cursor()

    # Query to fetch all column names from the provided table
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Close the connection
    conn.close()

    return columns

def get_column_values(database_path, column_name, filter_column = None, filter_value = None, table='Observation',partial_match=True):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    
    # Create a cursor object
    cursor = conn.cursor()

    # Query to fetch all values from the specified column in the Observation table
    # Base query
    query = f"SELECT {column_name} FROM {table}"

    # If filtering is specified, add a WHERE clause to the query
    if filter_column and filter_value is not None:
        if partial_match:
            query += f" WHERE {filter_column} LIKE ?"
            cursor.execute(query, ('%' + filter_value + '%',))
        else:
            query += f" WHERE {filter_column} = ?"
            cursor.execute(query, (filter_value,))
    else:
        cursor.execute(query)
    values = [row[0] for row in cursor.fetchall()]
    try:
        if isinstance(values[0], str):
            values = np.array([[float(v) for v in value.split(',')]  for value in values])
    except ValueError:
        pass
    
    # Close the connection
    conn.close()

    return values

def check_if_obsid_exists(database_path, obsid, table='Observation'):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT 1 FROM {table} WHERE obs_id=?", (obsid,))
    exists = cursor.fetchone()
    # Commit the changes
    conn.commit()
    
    # Close the connection
    conn.close()

    return exists

def update_entry(database_path, table, obsid, entry_name, _new_value):

    allowed_columns = get_columns_from_table(database_path, table)
    
    if entry_name not in allowed_columns:
        raise ValueError(f"Invalid column name {entry_name}")



    # Make sure list values are turned into strings
    if isinstance(_new_value, list) or isinstance(_new_value, np.ndarray):
        new_value = ','.join(map(str, _new_value))
    else:
        new_value = _new_value

    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    
    # Create a cursor object
    cursor = conn.cursor()

    cursor.execute(f"SELECT 1 FROM {table} WHERE obs_id=?", (obsid,))
    exists = cursor.fetchone()

    if exists:
        cursor.execute(f'''UPDATE {table}
                        SET {entry_name} = ?
                        WHERE obs_id = ?''', (new_value, obsid))
    else:
        cursor.execute(f"INSERT INTO {table} (obs_id, {entry_name}) VALUES (?, ?)", 
                       (obsid, new_value))
    
    # Commit the changes
    conn.commit()
    
    # Close the connection
    conn.close()




