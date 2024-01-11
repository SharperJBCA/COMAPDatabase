from app.create_database import create_database
from app.database_operations import populate_db_from_hdf5,get_hdf5_files
from app.plotting_operations import create_plots
from app.apply_stat_cuts import update_database_cuts, apply_database_cuts_to_level2
from app.calibration_factors import create_calibration_factors, apply_calibration_factors, plot_calibration_factors
from app.utilities import create_filelist
import argparse
import signal 
import os 
from tqdm import tqdm 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manage the Flask app.')
    parser.add_argument('command', choices=['create_database', 'update_database','update_database_single','create_plots','update_database_cuts', 'apply_database_cuts','create_calibration_factors','plot_calibration_factors','apply_calibration_factors','create_filelist'], nargs='?', help='Choose a command to run.')
    parser.add_argument('--database_name',type=str, help='Name of the database to create', default='comap_manchester_database.db')
    parser.add_argument('--figure_path',type=str, help='Path to directory containing all figures',default='outputs')
    parser.add_argument('--comap_file_path',type=str, help='Path to directory containing a COMAP HDF5 file')
    parser.add_argument('--comap_data_path',type=str, help='Path to directory containing all COMAP HDF5 files')
    parser.add_argument('--force_update',action='store_true', help='Force update of database', default=False)
    parser.add_argument('--cuts_json_file',type=str, help='Path to JSON file containing cuts to apply to the database',default='app/stat_cuts.json')
    parser.add_argument('--source',type=str, help='Specify source for some functions',default='CasA')
    parser.add_argument('--filelist_path',type=str, help='Path to filelist',default='filelist.txt')
    parser.add_argument('--ignore_flags' ,type=int, nargs='+', help='Flags to ignore when creating filelist',default=[])
    args = parser.parse_args()

    if args.command == 'create_database':
        if not os.path.exists(args.database_name):
            create_database(args.database_name)
    elif args.command == 'update_database_single':
        populate_db_from_hdf5(args.database_name, args.comap_file_path, force_update=args.force_update)
    elif args.command == 'update_database':
        for hdf5_file in tqdm(get_hdf5_files(args.comap_data_path),desc='Updating database'):
            populate_db_from_hdf5(args.database_name, hdf5_file, force_update=args.force_update)
    elif args.command == 'create_plots':
        create_plots(args.database_name, args.figure_path)
    elif args.command == 'update_database_cuts':
        update_database_cuts(args.database_name, args.cuts_json_file)
    elif args.command == 'apply_database_cuts':
        apply_database_cuts_to_level2(args.database_name)
    elif args.command == 'create_calibration_factors':
        create_calibration_factors(args.database_name)
    elif args.command == 'plot_calibration_factors':
        plot_calibration_factors(args.database_name, args.figure_path)
    elif args.command == 'apply_calibration_factors':
        apply_calibration_factors(args.database_name, args.source)
    elif args.command == 'create_filelist':
        create_filelist(args.database_name, args.source, output_path=args.filelist_path, ignore_flags=args.ignore_flags)