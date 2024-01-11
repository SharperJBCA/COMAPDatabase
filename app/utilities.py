import numpy as np
import h5py
import matplotlib.pyplot as plt
import os 
from .create_database import update_entry, get_column_values, create_calibration_database

def create_filelist(database_name, source, output_path='filelist.txt', ignore_flags=[]): 

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
    filelist = get_column_values(database_name, 'level2_file_path', filter_column='source_name', filter_value=source)
    bad_observation = get_column_values(database_name, 'bad_observation', filter_column='source_name', filter_value=source)

    with open(output_path, 'w') as f:
        for filename,bad in zip(filelist,bad_observation):
            if all([(b > 0) and not b in ignore_flags for b in bad]):
                continue
            f.write(f'{filename}\n')

