from .database_operations import db_to_pandas
from .create_database import update_entry, get_column_values, parse_bit_mask, parse_single_bit_mask
import numpy as np 
import os
from matplotlib import pyplot as plt
import json 
from astropy.time import Time
from datetime import datetime
from tqdm import tqdm
import h5py

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


def update_database_cuts(database_name, cuts_json_file):

    with open(cuts_json_file) as f:
        cuts = json.load(f)
    
    obsids = get_column_values(database_name, 'obs_id')
    mjds   = get_column_values(database_name, 'mjd')
    source_names = get_column_values(database_name, 'source_name')
    dates  = np.array([Time(mjd, format='mjd').datetime for mjd in mjds])
    bad_observations = get_column_values(database_name, 'bad_observation')

    for cut_name, cut_info in cuts.items(): 
        stats = get_column_values(database_name, cut_name)
        for obsid, bad_observation, date, stat,source_name in zip(tqdm(obsids), bad_observations, dates, stats, source_names):
            for date_block, cut_date in cut_info.items():
                start_date = datetime.strptime(cut_date['start'], '%Y-%m-%d')
                end_date   = datetime.strptime(cut_date['end'], '%Y-%m-%d')
                if (date >= start_date) and (date <= end_date):
                    if 'all' in cut_date['feeds']:
                        feeds = np.arange(1,21,dtype=int)
                    else:
                        feeds = np.array(cut_date['feeds'],dtype=int)

                    for feed in feeds:
                        if stat[feed-1] == -3123: # If the stat doesn't exist, we won't check it. 
                            continue
                        if (stat[feed-1] < cut_date['min_val'][0]) or (stat[feed-1] > cut_date['max_val'][0]):
                            if not any([1 in b for b in parse_single_bit_mask(bad_observation[feed-1])]):
                                bad_observation[feed-1] += 2**1
                update_entry(database_name, 'Observation', obsid, 'bad_observation', bad_observation)

def apply_database_cuts_to_level2(database_name):

    obsids = get_column_values(database_name, 'obs_id')
    bad_observations = get_column_values(database_name, 'bad_observation')
    filenames = get_column_values(database_name, 'level2_file_path')

    for filename, obsid, bad_observation in zip(tqdm(filenames,desc='updating level2 metadata'), obsids, bad_observations):
        if not os.path.exists(filename):
            continue
        h = h5py.File(filename, 'r+')
        h['comap'].attrs['bad_observation'] = np.array(bad_observation,dtype=int)
        h.close()