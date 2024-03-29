import os
import sys
import dill
import glob
import math
import zarr
import shutil
import logging
import pathlib
import warnings
import subprocess
import numpy as np
import pandas as pd
import SciServer.CasJobs as cj
from threading import Thread
from collections import defaultdict
from SciServer import Authentication
from dask.distributed import Client, LocalCluster
from giverny.turbulence_gizmos.basic_gizmos import *

# installs morton-py if necessary.
try:
    import morton
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'morton-py'])
finally:
    import morton

class iso_cube():
    def __init__(self, dataset_title = '', output_path = '', auth_token = '', cube_dimensions = 3, rewrite_dataset_metadata = False, rewrite_interpolation_metadata = False):
        """
        initialize the class.
        """
        # turn off the dask warning for scattering large objects to the workers.
        warnings.filterwarnings("ignore", message = ".*Large object of size.*detected in task graph")
        
        # check that dataset_title is a valid dataset title.
        check_dataset_title(dataset_title)
        
        # cube size.
        self.N = get_dataset_resolution(dataset_title)
        # conversion factor between the cube size and a domain on [0, 2pi].
        self.dx = 2 * np.pi / self.N
        
        # setting up Morton curve.
        bits = int(math.log(self.N, 2))
        self.mortoncurve = morton.Morton(dimensions = cube_dimensions, bits = bits)
        
        # interpolation lookup table resolution.
        self.lookup_N = 10**5
        
        # turbulence dataset name, e.g. "isotropic8192" or "isotropic1024fine".
        self.dataset_title = dataset_title
        
        # get the SciServer user name.
        user = Authentication.getKeystoneUserWithToken(Authentication.getToken()).userName
        
        # set the directory for saving any output files.
        self.output_path = output_path.strip()
        if self.output_path == '':
            self.output_path = pathlib.Path(f'/home/idies/workspace/Temporary/{user}/scratch/turbulence_output')
        else:
            self.output_path = pathlib.Path(self.output_path)
        
        # create the output directory if it does not already exist.
        create_output_folder(self.output_path)
        
        # user authorization token for pyJHTDB.
        self.auth_token = auth_token
        
        # set the directory for reading the pickled files.
        self.pickle_dir = pathlib.Path(f'/home/idies/workspace/turb/data01_01/zarr/turbulence_pickled')
        
        # set the backup directory for reading the pickled files.
        self.pickle_dir_backup = pathlib.Path(f'/home/idies/workspace/turb/data02_01/zarr/turbulence_pickled_back')
        
        # set the local directory for writing the pickled metadata files if the primary and backup directories are inaccessible.
        self.pickle_dir_local = pathlib.Path(f'/home/idies/workspace/Temporary/{user}/scratch/turbulence_pickled')
        
        # create the local pickle directory if it does not already exist.
        create_output_folder(self.pickle_dir_local)
        
        # retrieve the list of datasets processed by the giverny code.
        giverny_datasets = get_giverny_datasets()
        
        """
        read/write metadata files.
        """
        # only read/write the metadata files if the dataset being queried is handled by this code.
        if self.dataset_title in giverny_datasets:
            # get a cache of the metadata for the database files.
            self.init_cache(read_metadata = True, rewrite_metadata = rewrite_dataset_metadata)

            # get map of the filepaths for all of the dataset files.
            self.init_filepaths(self.dataset_title, read_metadata = True, rewrite_metadata = rewrite_dataset_metadata)

            # get a map of the files to cornercodes for all of the dataset files.
            self.init_cornercode_file_map(self.dataset_title, self.N, read_metadata = True, rewrite_metadata = rewrite_dataset_metadata)

            # rewrite interpolation metadata files if specified.
            if rewrite_interpolation_metadata:
                # initialize the interpolation lookup table.
                self.init_interpolation_lookup_table(read_metadata = False, rewrite_metadata = rewrite_interpolation_metadata)

                # initialize the interpolation cube size lookup table.
                self.init_interpolation_cube_size_lookup_table(read_metadata = False, rewrite_metadata = rewrite_interpolation_metadata)
    
    """
    initialization functions.
    """
    def init_cache(self, read_metadata = False, rewrite_metadata = False):
        """
        pickled SQL metadata.
        """
        # pickled SQL metadata.
        pickle_filename = self.dataset_title + '_metadata.pickle'
        pickle_file_prod = self.pickle_dir.joinpath(pickle_filename)
        pickle_file_back = self.pickle_dir_backup.joinpath(pickle_filename)
        pickle_file_local = self.pickle_dir_local.joinpath(pickle_filename)
        
        # check if the pickled file is accessible.
        if not (pickle_file_prod.is_file() or pickle_file_back.is_file() or pickle_file_local.is_file()) or rewrite_metadata:
            # read SQL metadata for all of the turbulence data files into the cache.
            sql = f"""
            select dbm.ProductionDatabaseName
            , dbm.minLim
            , dbm.maxLim
            from databasemap dbm
            where dbm.datasetname = '{self.dataset_title}'
            order by minlim
            """
            df = cj.executeQuery(sql, "turbinfo")
            
            # retrieve the list of datasets that need their sql metadata manually created.
            sql_metadata_datasets = get_manual_sql_metadata_datasets()

            if self.dataset_title in sql_metadata_datasets:
                 # get the common filename prefix for all files in this dataset, e.g. "iso8192" for the isotropic8192 dataset.
                dataset_filename_prefix = get_filename_prefix(self.dataset_title)

                # adds in the missing sql metadata for the dataset in sql_metadata_datasets.
                df_update = []
                for index in range(1, 64 + 1):
                    filename_key = dataset_filename_prefix + str(index).zfill(2)
                    min_max_morton = ((index - 1) * 512**3, index * 512**3 - 1)

                    df_update.append({'ProductionDatabaseName': filename_key.strip(), 'minLim': min_max_morton[0], 'maxLim': min_max_morton[1]})

                df = pd.concat([df, pd.DataFrame(df_update)], ignore_index = True)

            x, y, z = self.mortoncurve.unpack(df['minLim'].values)
            df['x_min'] = x
            df['y_min'] = y
            df['z_min'] = z

            x, y, z = self.mortoncurve.unpack(df['maxLim'].values)
            df['x_max'] = x
            df['y_max'] = y 
            df['z_max'] = z

            tmp_cache = df

            # save tmp_cache to a pickled file.
            with open(pickle_file_local, 'wb') as pickled_filepath:
                dill.dump(tmp_cache, pickled_filepath)
                    
        if read_metadata:
            self.cache = self.read_pickle_file(pickle_filename)
        
    def init_filepaths(self, dataset_title, read_metadata = False, rewrite_metadata = False):
        """
        pickled filepaths.
        """
        # pickled production filepaths.
        pickle_filename = dataset_title + '_database_filepaths.pickle'
        pickle_file_prod = self.pickle_dir.joinpath(pickle_filename)
        pickle_file_back = self.pickle_dir_backup.joinpath(pickle_filename)
        pickle_file_local = self.pickle_dir_local.joinpath(pickle_filename)
        
        # pickled backup filepaths.
        pickle_filename_backup = dataset_title + '_database_filepaths_backup.pickle'
        pickle_file_prod_backup = self.pickle_dir.joinpath(pickle_filename_backup)
        pickle_file_back_backup = self.pickle_dir_backup.joinpath(pickle_filename_backup)
        pickle_file_local_backup = self.pickle_dir_local.joinpath(pickle_filename_backup)
        
        # check if the pickled files are accessible.
        if not (pickle_file_prod.is_file() or pickle_file_back.is_file() or pickle_file_local.is_file()) or \
            not (pickle_file_prod_backup.is_file() or pickle_file_back_backup.is_file() or pickle_file_local_backup.is_file()) or \
            rewrite_metadata:
            # specifies the folders on fileDB that should be searched for the primary and backup copies of the zarr files.
            folder_base = '/home/idies/workspace/turb/'
            folder_paths = ['data01_01/zarr/', 'data01_02/zarr/', 'data01_03/zarr/',
                            'data02_01/zarr/', 'data02_02/zarr/', 'data02_03/zarr/',
                            'data03_01/zarr/', 'data03_02/zarr/', 'data03_03/zarr/',
                            'data04_01/zarr/', 'data04_02/zarr/', 'data04_03/zarr/',
                            'data05_01/zarr/', 'data05_02/zarr/', 'data05_03/zarr/',
                            'data06_01/zarr/', 'data06_02/zarr/', 'data06_03/zarr/',
                            'data07_01/zarr/', 'data07_02/zarr/', 'data07_03/zarr/',
                            'data08_01/zarr/', 'data08_02/zarr/', 'data08_03/zarr/',
                            'data09_01/zarr/', 'data09_02/zarr/', 'data09_03/zarr/',
                            'data10_01/zarr/', 'data10_02/zarr/', 'data10_03/zarr/',
                            'data11_01/zarr/', 'data11_02/zarr/', 'data11_03/zarr/',
                            'data12_01/zarr/', 'data12_02/zarr/', 'data12_03/zarr/']
            
            turb_folders = [folder_base + folder_path for folder_path in folder_paths]
            
            # get the common filename prefix for all files in this dataset, e.g. "iso8192" for the isotropic8192 dataset.
            dataset_filename_prefix = get_filename_prefix(dataset_title)
            
            """
            production filepaths.
            """
            # map the filepaths to the part of each filename that matches "ProductionDatabaseName" in the SQL metadata for this dataset.
            tmp_filepaths = {}

            # recursively search all sub-directories in the turbulence fileDB system for the dataset zarr files.
            filepaths = []
            for turb_folder in turb_folders:
                # production metadata.
                data_filepaths = glob.glob(turb_folder + dataset_filename_prefix + '*_prod/*.zarr')

                filepaths += data_filepaths

            # map the filepaths to the filenames so that they can be easily retrieved.
            for filepath in filepaths:
                # part of the filenames that exactly matches the "ProductionDatabaseName" column stored in the SQL metadata.
                filepath_split = filepath.split(os.sep)
                folderpath = os.sep.join(filepath_split[:-1]) + os.sep
                filename = filepath.split(os.sep)[-1].split('_')[0].strip()
                # only add the filepath to the dictionary once since there could be backup copies of the files.
                if filename not in tmp_filepaths:
                    tmp_filepaths[filename] = folderpath + filename

            # save tmp_filepaths to a pickled file.
            with open(pickle_file_local, 'wb') as pickled_filepath:
                dill.dump(tmp_filepaths, pickled_filepath)
            
            """
            backup filepaths.
            """
            # map the filepaths to the part of each filename that matches "ProductionDatabaseName" in the SQL metadata for this dataset.
            tmp_filepaths_backup = {}

            # recursively search all sub-directories in the turbulence fileDB system for the dataset zarr files.
            filepaths = []
            for turb_folder in turb_folders:
                # backup metadata.
                data_filepaths = glob.glob(turb_folder + dataset_filename_prefix + '*_back/*.zarr')

                filepaths += data_filepaths

            # map the filepaths to the filenames so that they can be easily retrieved.
            for filepath in filepaths:
                # part of the filenames that exactly matches the "ProductionDatabaseName" column stored in the SQL metadata.
                filepath_split = filepath.split(os.sep)
                folderpath = os.sep.join(filepath_split[:-1]) + os.sep
                filename = filepath.split(os.sep)[-1].split('_')[0].strip()
                # only add the filepath to the dictionary once since there could be backup copies of the files.
                if filename not in tmp_filepaths_backup:
                    tmp_filepaths_backup[filename] = folderpath + filename

            # save tmp_filepaths_backup to a pickled file.
            with open(pickle_file_local_backup, 'wb') as pickled_filepath:
                dill.dump(tmp_filepaths_backup, pickled_filepath)
                    
        if read_metadata:
            self.filepaths = self.read_pickle_file(pickle_filename)
            self.filepaths_backup = self.read_pickle_file(pickle_filename_backup)
                
    def init_cornercode_file_map(self, dataset_title, N, read_metadata = False, rewrite_metadata = False):
        """
        pickled db file cornercodes to filenames map.
        """
        # pickled db file cornercodes to production filenames map.
        pickle_filename = dataset_title + f'_cornercode_file_map.pickle'
        pickle_file_prod = self.pickle_dir.joinpath(pickle_filename)
        pickle_file_back = self.pickle_dir_backup.joinpath(pickle_filename)
        pickle_file_local = self.pickle_dir_local.joinpath(pickle_filename)
        
        # check if the pickled file is accessible.
        if not (pickle_file_prod.is_file() or pickle_file_back.is_file() or pickle_file_local.is_file()) or rewrite_metadata:
            # create a map of the db file cornercodes to filenames for the whole dataset.
            tmp_cornercode_file_map = {}

            cornercode = 0
            while cornercode < N ** 3:
                # get the file info for the db file cornercode.
                f, db_minLim, db_maxLim = self.get_file_for_mortoncode(cornercode)

                tmp_cornercode_file_map[db_minLim] = f

                cornercode = db_maxLim + 1

            # save tmp_cornercode_file_map to a pickled file.
            with open(pickle_file_local, 'wb') as pickled_file_map:
                dill.dump(tmp_cornercode_file_map, pickled_file_map)
                
        if read_metadata:
            self.cornercode_file_map = self.read_pickle_file(pickle_filename)
    
    def init_interpolation_lookup_table(self, sint = 'none', read_metadata = False, rewrite_metadata = False):
        """
        pickled interpolation lookup table.
        """
        # interpolation method 'none' is omitted because there is no lookup table for 'none' interpolation.
        interp_methods = ['lag4', 'm1q4', 'lag6', 'lag8', 'm2q8',
                          'fd4noint_g', 'fd6noint_g', 'fd8noint_g', 'fd4lag4_g', 'm1q4_g', 'm2q8_g',
                          'fd4noint_l', 'fd6noint_l', 'fd8noint_l', 'fd4lag4_l',
                          'fd4noint_h', 'fd6noint_h', 'fd8noint_h', 'm2q8_h']
        
        # create the metadata files for each interpolation method if they do not already exist.
        for interp_method in interp_methods:
            # pickled file for saving the interpolation coefficient lookup table.
            pickle_filename = f'{interp_method}_lookup_table.pickle'
            pickle_file_prod = self.pickle_dir.joinpath(pickle_filename)
            pickle_file_back = self.pickle_dir_backup.joinpath(pickle_filename)
            pickle_file_local = self.pickle_dir_local.joinpath(pickle_filename)

            # check if the pickled file is accessible.
            if not (pickle_file_prod.is_file() or pickle_file_back.is_file() or pickle_file_local.is_file()) or rewrite_metadata:
                # create the interpolation coefficient lookup table.
                tmp_lookup_table = self.createInterpolationLookupTable(interp_method)

                # save tmp_lookup_table to a pickled file.
                with open(pickle_file_local, 'wb') as pickled_lookup_table:
                    dill.dump(tmp_lookup_table, pickled_lookup_table)
        
        # read in the interpolation lookup table for sint. the interpolation lookup tables are only read from
        # the get_iso_points_variable and get_iso_points_variable_visitor functions.
        if sint != 'none' and read_metadata:
            # pickled interpolation coefficient lookup table.
            self.lookup_table = self.read_pickle_file(f'{sint}_lookup_table.pickle')
            
            # read in the function interpolation lookup table that is used in the calculation of other interpolation methods.
            if sint in ['fd4lag4_g', 'm1q4_g', 'm2q8_g',
                        'fd4lag4_l',
                        'm2q8_h']:
                # convert sint to the needed function interpolation name.
                sint_name = sint.split('_')[0].replace('fd4', '')
                
                # pickled interpolation coefficient lookup table.
                self.function_lookup_table = self.read_pickle_file(f'{sint_name}_lookup_table.pickle')
                    
                # read in the gradient coefficient lookup table that is used in the calculation of the m2q8 spline hessian.
                if sint == 'm2q8_h':
                    # convert sint to the needed gradient interpolation name.
                    sint_name = sint.replace('_h', '_g')
                    
                    # pickled gradient coefficient lookup table.
                    self.gradient_lookup_table = self.read_pickle_file(f'{sint_name}_lookup_table.pickle')
            # read in the laplacian interpolation lookup table that is used in the calculation of other interpolation methods.
            elif sint in ['fd4noint_h', 'fd6noint_h', 'fd8noint_h']:
                # convert sint to the needed gradient interpolation name.
                sint_name = sint.replace('_h', '_l')
                
                # pickled laplacian coefficient lookup table.
                self.laplacian_lookup_table = self.read_pickle_file(f'{sint_name}_lookup_table.pickle')
                
    def init_interpolation_cube_size_lookup_table(self, read_metadata = False, rewrite_metadata = False):
        """
        pickled interpolation cube sizes table.
        """
        # pickled interpolation cube sizes lookup table.
        pickle_filename = 'interpolation_cube_size_lookup_table.pickle'
        pickle_file_prod = self.pickle_dir.joinpath(pickle_filename)
        pickle_file_back = self.pickle_dir_backup.joinpath(pickle_filename)
        pickle_file_local = self.pickle_dir_local.joinpath(pickle_filename)
        
        # check if the pickled file is accessible.
        if not (pickle_file_prod.is_file() or pickle_file_back.is_file() or pickle_file_local.is_file()) or rewrite_metadata:
            # create the interpolation cube size lookup table. the first number is the number of points on the left of the integer 
            # interpolation point, and the second number is the number of points on the right.
            tmp_interp_cube_sizes = {}
            tmp_interp_cube_sizes['lag4'] = [1, 2]
            tmp_interp_cube_sizes['m1q4'] = [1, 2]
            tmp_interp_cube_sizes['lag6'] = [2, 3]
            tmp_interp_cube_sizes['lag8'] = [3, 4]
            tmp_interp_cube_sizes['m2q8'] = [3, 4]
            tmp_interp_cube_sizes['fd4noint_g'] = [2, 3]
            tmp_interp_cube_sizes['fd6noint_g'] = [3, 4]
            tmp_interp_cube_sizes['fd8noint_g'] = [4, 5]
            tmp_interp_cube_sizes['m1q4_g'] = [1, 2]
            tmp_interp_cube_sizes['m2q8_g'] = [3, 4]
            tmp_interp_cube_sizes['fd4lag4_g'] = [3, 4]
            tmp_interp_cube_sizes['fd4noint_l'] = [2, 3]
            tmp_interp_cube_sizes['fd6noint_l'] = [3, 4]
            tmp_interp_cube_sizes['fd8noint_l'] = [4, 5]
            tmp_interp_cube_sizes['fd4lag4_l'] = [3, 4]
            tmp_interp_cube_sizes['fd4noint_h'] = [2, 3]
            tmp_interp_cube_sizes['fd6noint_h'] = [3, 4]
            tmp_interp_cube_sizes['fd8noint_h'] = [4, 5]
            tmp_interp_cube_sizes['m2q8_h'] = [3, 4]
            tmp_interp_cube_sizes['none'] = [0, 1]

            # save interp_cube_sizes to a pickled file.
            with open(pickle_file_local, 'wb') as pickled_lookup_table:
                dill.dump(tmp_interp_cube_sizes, pickled_lookup_table)
            
        # the interpolation cube size indices are only read when called from the init_constants function.
        if read_metadata:
            interp_cube_sizes = self.read_pickle_file(pickle_filename)

            # lookup the interpolation cube size indices.
            self.cube_min_index, self.cube_max_index = interp_cube_sizes[self.sint]
    
    def init_constants(self, var, var_original, timepoint, sint, tint, num_values_per_datapoint, c):
        """
        initialize the constants.
        """
        self.var = var
        self.var_name = var_original
        self.timepoint = timepoint
        self.sint = sint
        self.tint = tint
        self.num_values_per_datapoint = num_values_per_datapoint
        self.bytes_per_datapoint = c['bytes_per_datapoint']
        self.voxel_side_length = c['voxel_side_length']
        self.missing_value_placeholder = c['missing_value_placeholder']
        self.database_file_disk_index = c['database_file_disk_index']
        self.dask_maximum_processes = c['dask_maximum_processes']
        self.decimals = c['decimals']
        self.chunk_size = c['chunk_size']
        self.file_size = c['file_size']
        
        # set the byte order for reading the data from the files.
        self.dt = np.dtype(np.float32)
        self.dt = self.dt.newbyteorder('<')
        
        # get the number of digits in the timepoint part of the filenames, in order to add leading zeros.
        self.timepoint_digits = get_timepoint_digits(self.dataset_title)
        
        # retrieve the list of datasets which use time indices.
        time_index_datasets = get_time_index_datasets()
        
        # set the dataset name to be used in the hdf5 file. 1 is added to timepoint because the original timepoint was converted to a 0-based index.
        dataset_name_timepoint = timepoint
        if self.dataset_title in time_index_datasets:
            dataset_name_timepoint = str(dataset_name_timepoint + 1).zfill(4)
        else:
            dataset_name_timepoint = str(dataset_name_timepoint)
        
        self.dataset_name = get_output_variable_name(var_original) + '_' + dataset_name_timepoint
        
        # retrieve the list of datasets processed by the giverny code.
        giverny_datasets = get_giverny_datasets()
        
        # initialize the interpolation cube size lookup table.
        if self.dataset_title in giverny_datasets:
            self.init_interpolation_cube_size_lookup_table(read_metadata = True)
        
            # variables needed to open the zarr file for reading. variables are put together for easier passing into the dask worker functions.
            self.open_file_vars = [self.var_name, self.timepoint, self.timepoint_digits, self.dt]

            # interpolate function variables.
            self.interpolate_vars = [self.cube_min_index, self.cube_max_index, self.sint, self.lookup_N, self.dx]

            # getCutout variables.
            self.getcutout_vars = [self.file_size]

            # getData variables.
            self.getdata_vars = [self.dataset_title, self.num_values_per_datapoint, self.N, self.chunk_size, self.file_size]
    
    """
    interpolation functions.
    """
    def createInterpolationLookupTable(self, sint):
        """
        generate interpolation lookup table.
        """
        if sint in ['fd4noint_g', 'fd6noint_g', 'fd8noint_g',
                    'fd4noint_l', 'fd6noint_l', 'fd8noint_l',
                    'fd4noint_h', 'fd6noint_h', 'fd8noint_h']:
            lookup_table = self.getInterpolationCoefficients(sint)
        else:
            lookup_table = []
            
            frac = np.linspace(0, 1 - 1 / self.lookup_N, self.lookup_N)
            for fp in frac:
                lookup_table.append(self.getInterpolationCoefficients(sint, fp))

        return lookup_table
    
    def getInterpolationCoefficients(self, sint, fr = 0.0):
        """
        get interpolation coefficients.
        """
        if sint == 'fd4noint_h':
            g = np.array([-1.0 / 48.0 / self.dx / self.dx,
                          1.0 / 48.0 / self.dx / self.dx,
                          -1.0 / 48.0 / self.dx / self.dx,
                          1.0 / 48.0 / self.dx / self.dx,
                          1.0 / 3.0 / self.dx / self.dx,
                          -1.0 / 3.0 / self.dx / self.dx,
                          1.0 / 3.0 / self.dx / self.dx,
                          -1.0 / 3.0 / self.dx / self.dx])
        elif sint == 'fd6noint_h':
            g = np.array([1.0 / 360.0 / self.dx / self.dx,
                          -1.0 / 360.0 / self.dx / self.dx,
                          1.0 / 360.0 / self.dx / self.dx,
                          -1.0 / 360.0 / self.dx / self.dx,
                          -3.0 / 80.0 / self.dx / self.dx,
                          3.0 / 80.0 / self.dx / self.dx,
                          -3.0 / 80.0 / self.dx / self.dx,
                          3.0 / 80.0 / self.dx / self.dx,
                          3.0 / 8.0 / self.dx / self.dx,
                          -3.0 / 8.0/ self.dx / self.dx,
                          3.0 / 8.0/ self.dx / self.dx,
                          -3.0 / 8.0/ self.dx / self.dx])
        elif sint == 'fd8noint_h':
            g = np.array([-1.0 / 2240.0 / self.dx / self.dx,
                          1.0 / 2240.0 / self.dx / self.dx,
                          -1.0 / 2240.0 / self.dx / self.dx,
                          1.0 / 2240.0 / self.dx / self.dx,
                          2.0 / 315.0 / self.dx / self.dx,
                          -2.0 / 315.0 / self.dx / self.dx,
                          2.0 / 315.0 / self.dx / self.dx,
                          -2.0 / 315.0 / self.dx / self.dx,
                          -1.0 / 20.0 / self.dx / self.dx,
                          1.0 / 20.0 / self.dx / self.dx,
                          -1.0 / 20.0 / self.dx / self.dx,
                          1.0 / 20.0 / self.dx / self.dx,
                          14.0 / 35.0 / self.dx / self.dx,
                          -14.0 / 35.0 / self.dx / self.dx,
                          14.0 / 35.0 / self.dx / self.dx,
                          -14.0 / 35.0 / self.dx / self.dx])
        elif sint == 'm2q8_h':
            g = np.zeros(8)
            g[0] = fr * (fr * ((8.0 / 9.0) * fr - 7.0 / 5.0) + 1.0 / 2.0) + 1.0 / 90.0
            g[1] = fr * (fr * (-115.0 / 18.0 * fr + 61.0 / 6.0) - 217.0 / 60.0) - 3.0 / 20.0
            g[2] = fr * (fr * ((39.0 / 2.0) * fr - 153.0 / 5.0) + 189.0 / 20.0) + 3.0 / 2.0
            g[3] = fr * (fr * (-295.0 / 9.0 * fr + 50) - 13) - 49.0 / 18.0
            g[4] = fr * (fr * ((295.0 / 9.0) * fr - 145.0 / 3.0) + 34.0 / 3.0) + 3.0 / 2.0
            g[5] = fr * (fr * (-39.0 / 2.0 * fr + 279.0 / 10.0) - 27.0 / 4.0) - 3.0 / 20.0
            g[6] = fr * (fr * ((115.0 / 18.0) * fr - 9) + 49.0 / 20.0) + 1.0 / 90.0
            g[7] = fr * (fr * (-8.0 / 9.0 * fr + 19.0 / 15.0) - 11.0 / 30.0)
        elif sint == 'fd4noint_l':
            g = np.array([-1.0 / 12.0 / self.dx / self.dx,
                          4.0 / 3.0 / self.dx / self.dx,
                          -15.0 / 6.0 / self.dx / self.dx,
                          4.0 / 3.0 / self.dx / self.dx,
                          -1.0 / 12.0 / self.dx / self.dx])
        elif sint == 'fd6noint_l':
            g = np.array([1.0 / 90.0 / self.dx / self.dx,
                          -3.0 / 20.0 / self.dx / self.dx,
                          3.0 / 2.0 / self.dx / self.dx,
                          -49.0 / 18.0 / self.dx / self.dx,
                          3.0 / 2.0 / self.dx / self.dx,
                          -3.0 / 20.0 / self.dx / self.dx,
                          1.0 / 90.0 / self.dx / self.dx])
        elif sint == 'fd8noint_l':
            g = np.array([9.0 / 3152.0 / self.dx / self.dx,
                          -104.0 / 8865.0 / self.dx / self.dx,
                          -207.0 / 2955.0 / self.dx / self.dx,
                          792.0 / 591.0 / self.dx / self.dx,
                          -35777.0 / 14184.0 / self.dx / self.dx,
                          792.0 / 591.0 / self.dx / self.dx,
                          -207.0 / 2955.0 / self.dx / self.dx,
                          -104.0 / 8865.0 / self.dx / self.dx,
                          9.0 / 3152.0 / self.dx / self.dx])
        elif sint == 'fd4noint_g':
            g =  np.array([1.0 / 12.0 / self.dx,
                           -2.0 / 3.0 / self.dx, 
                           0.0, 
                           2.0 / 3.0 / self.dx,
                           -1.0 / 12.0 / self.dx])
        elif sint == 'fd6noint_g':
            g = np.array([-1.0 / 60.0 / self.dx,
                          3.0 / 20.0 / self.dx,
                          -3.0 / 4.0 / self.dx, 
                          0.0, 
                          3.0 / 4.0 / self.dx,
                          -3.0 / 20.0 / self.dx,
                          1.0 / 60.0 / self.dx])
        elif sint == 'fd8noint_g':
            g = np.array([1.0 / 280.0 / self.dx,
                          -4.0 / 105.0 / self.dx,
                          1.0 / 5.0 / self.dx,
                          -4.0 / 5.0 / self.dx, 
                          0.0, 
                          4.0 / 5.0 / self.dx,
                          -1.0 / 5.0 / self.dx,
                          4.0 / 105.0 / self.dx,
                          -1.0 / 280.0 / self.dx])
        elif sint in ['fd4lag4_g', 'fd4lag4_l']:
            wN = [1.,-3.,3.,-1.]
            B  = np.array([0,1.,0,0])
            # calculate weights if fr>0, and insert into g.
            if (fr>0):
                s = 0
                for n in range(4):
                    B[n] = wN[n]/(fr-n+1)
                    s += B[n]
                for n in range(4):
                    B[n] = B[n]/s

            if sint == 'fd4lag4_g':
                A =  [1.0 / 12.0 / self.dx,
                      -2.0 / 3.0 / self.dx, 
                      0.0, 
                      2.0 / 3.0 / self.dx,
                      -1.0 / 12.0 / self.dx]
            elif sint == 'fd4lag4_l':
                A = [-1.0 / 12.0 / self.dx / self.dx,
                     4.0 / 3.0 / self.dx / self.dx,
                     -15.0 / 6.0 / self.dx / self.dx,
                     4.0 / 3.0 / self.dx / self.dx,
                     -1.0 / 12.0 / self.dx / self.dx]

            g = np.zeros(8)
            g[0] =  B[0]*A[0]
            g[1] =  B[0]*A[1] + B[1]*A[0]
            g[2] =  B[0]*A[2] + B[1]*A[1] + B[2]*A[0]
            g[3] =  B[0]*A[3] + B[1]*A[2] + B[2]*A[1] + B[3]*A[0] 
            g[4] =  B[0]*A[4] + B[1]*A[3] + B[2]*A[2] + B[3]*A[1] 
            g[5] =  B[1]*A[4] + B[2]*A[3] + B[3]*A[2] 
            g[6] =  B[2]*A[4] + B[3]*A[3] 
            g[7] =  B[3]*A[4]
        elif sint == 'm1q4_g':
            g = np.zeros(4)
            g[0] = fr * (-3.0 / 2.0 * fr + 2) - 1.0 / 2.0
            g[1] = fr * ((9.0 / 2.0) * fr - 5)
            g[2] = fr * (-9.0 / 2.0 * fr + 4) + 1.0 / 2.0
            g[3] = fr * ((3.0 / 2.0) * fr - 1)
        elif sint == 'm2q8_g':
            g = np.zeros(8)
            g[0] = fr * (fr * (fr * ((2.0 / 9.0) * fr - 7.0 / 15.0) + 1.0 / 4.0) + 1.0 / 90.0) - 1.0 / 60.0
            g[1] = fr * (fr * (fr * (-115.0 / 72.0 * fr + 61.0 / 18.0) - 217.0 / 120.0) - 3.0 / 20.0) + 3.0 / 20.0
            g[2] = fr * (fr * (fr * ((39.0 / 8.0) * fr - 51.0 / 5.0) + 189.0 / 40.0) + 3.0 / 2.0) - 3.0 / 4.0
            g[3] = fr * (fr * (fr * (-295.0 / 36.0 * fr + 50.0 / 3.0) - 13.0 / 2.0) - 49.0 / 18.0)
            g[4] = fr * (fr * (fr * ((295.0 / 36.0) * fr - 145.0 / 9.0) + 17.0 / 3.0) + 3.0 / 2.0) + 3.0 / 4.0
            g[5] = fr * (fr * (fr * (-39.0 / 8.0 * fr + 93.0 / 10.0) - 27.0 / 8.0) - 3.0 / 20.0) - 3.0 / 20.0
            g[6] = fr * (fr * (fr * ((115.0 / 72.0) * fr - 3) + 49.0 / 40.0) + 1.0 / 90.0) + 1.0 / 60.0
            g[7] = fr**2 * (fr * (-2.0 / 9.0 * fr + 19.0 / 45.0) - 11.0 / 60.0)
        elif sint == 'm1q4':
            # define the weights for m1q4 spline interpolation.
            g = np.zeros(4)
            g[0] = fr * (fr * (-1.0 / 2.0 * fr + 1) - 1.0 / 2.0)
            g[1] = fr**2 * ((3.0 / 2.0) * fr - 5.0 / 2.0) + 1
            g[2] = fr * (fr * (-3.0 / 2.0 * fr + 2) + 1.0 / 2.0)
            g[3] = fr**2 * ((1.0 / 2.0) * fr - 1.0 / 2.0)
        elif sint == 'm2q8':
            # define the weights for m2q8 spline interpolation.
            g = np.zeros(8)  
            g[0] = fr * (fr * (fr * (fr * ((2.0 / 45.0) * fr - 7.0 / 60.0) + 1.0 / 12.0) + 1.0 / 180.0) - 1.0 / 60.0)
            g[1] = fr * (fr * (fr * (fr * (-23.0 / 72.0 * fr + 61.0 / 72.0) - 217.0 / 360.0) - 3.0 / 40.0) + 3.0 / 20.0)
            g[2] = fr * (fr * (fr * (fr * ((39.0 / 40.0) * fr - 51.0 / 20.0) + 63.0 / 40.0) + 3.0 / 4.0) - 3.0 / 4.0)
            g[3] = fr**2 * (fr * (fr * (-59.0 / 36.0 * fr + 25.0 / 6.0) - 13.0 / 6.0) - 49.0 / 36.0) + 1
            g[4] = fr * (fr * (fr * (fr * ((59.0 / 36.0) * fr - 145.0 / 36.0) + 17.0 / 9.0) + 3.0 / 4.0) + 3.0 / 4.0)
            g[5] = fr * (fr * (fr * (fr * (-39.0 / 40.0 * fr + 93.0 / 40.0) - 9.0 / 8.0) - 3.0 / 40.0) - 3.0 / 20.0)
            g[6] = fr * (fr * (fr * (fr * ((23.0 / 72.0) * fr - 3.0 / 4.0) + 49.0 / 120.0) + 1.0 / 180.0) + 1.0 / 60.0)
            g[7] = fr**3 * (fr * (-2.0 / 45.0 * fr + 19.0 / 180.0) - 11.0 / 180.0)
        else:
            # define the weights for the different lagrangian interpolation methods.
            if sint == 'lag4':
                wN = [1.,-3.,3.,-1.]
                g  = np.array([0,1.,0,0])
                # weight index.
                w_index = 1
            elif sint == 'lag6':
                wN = [1.,-5.,10.,-10.,5.,-1.]
                g  = np.array([0,0,1.,0,0,0])
                # weight index.
                w_index = 2
            elif sint == 'lag8':
                wN = [1.,-7.,21.,-35.,35.,-21.,7.,-1.]
                g  = np.array([0,0,0,1.,0,0,0,0])
                # weight index.
                w_index = 3

            # calculate weights if fr>0, and insert into g.
            if (fr>0):
                num_points = len(g)

                s = 0
                for n in range(num_points):
                    g[n] = wN[n] / (fr - n + w_index)
                    s += g[n]

                for n in range(num_points):
                    g[n] = g[n] / s

        return g
    
    def spatial_interpolate(self, p, u, interpolate_vars):
        """
        spatial interpolating functions to compute the kernel, extract subcube and convolve.
        
        vars:
         - p is an np.array(3) containing the three coordinates.
        """
        # assign the local variables.
        cube_min_index, cube_max_index, sint, lookup_N, dx = interpolate_vars
        
        if sint in ['lag4', 'm1q4', 'lag6', 'lag8', 'm2q8']:
            # function interpolations.
            ix = p.astype(np.int32)
            fr = p - ix
            
            # get the coefficients.
            gx = self.lookup_table[int(lookup_N * fr[0])]
            gy = self.lookup_table[int(lookup_N * fr[1])]
            gz = self.lookup_table[int(lookup_N * fr[2])]
            
            # create the 3D kernel from the outer product of the 1d kernels.
            gk = np.einsum('i,j,k', gz, gy, gx)

            return np.einsum('ijk,ijkl->l', gk, u)
        elif sint == 'none':
            # 'none' function interpolation.
            ix = np.floor(p + 0.5).astype(np.int32)
            
            return np.array(u[ix[2], ix[1], ix[0], :])
        elif sint in ['fd4noint_g', 'fd6noint_g', 'fd8noint_g',
                      'fd4noint_l', 'fd6noint_l', 'fd8noint_l']:
            # gradient and laplacian finite differences.
            ix = np.floor(p + 0.5).astype(int)
            # diagonal coefficients.
            fd_coeff = self.lookup_table
            
            # diagnoal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index : ix[0] + cube_max_index, :]
            component_y = u[ix[2], ix[1] - cube_min_index : ix[1] + cube_max_index, ix[0], :]
            component_z = u[ix[2] - cube_min_index : ix[2] + cube_max_index, ix[1], ix[0], :]

            dvdx = np.inner(fd_coeff, component_x.T)  
            dvdy = np.inner(fd_coeff, component_y.T)
            dvdz = np.inner(fd_coeff, component_z.T)
            
            # return gradient values.
            if '_g' in sint:
                return np.stack((dvdx, dvdy, dvdz), axis = 1).flatten()
            # return laplacian values.
            elif '_l' in sint:
                return dvdx + dvdy + dvdz
        elif sint in ['fd4noint_h', 'fd6noint_h', 'fd8noint_h']:
            # hessian finite differences.
            ix = np.floor(p + 0.5).astype(int)
            # diagonal coefficients.
            fd_coeff_l = self.laplacian_lookup_table
            # off-diagonal coefficients.
            fd_coeff_h = self.lookup_table
            
            # diagnoal components.
            component_x = u[ix[2], ix[1], ix[0] - cube_min_index : ix[0] + cube_max_index, :]
            component_y = u[ix[2], ix[1] - cube_min_index : ix[1] + cube_max_index, ix[0], :]
            component_z = u[ix[2] - cube_min_index : ix[2] + cube_max_index, ix[1], ix[0], :]

            uii = np.inner(fd_coeff_l, component_x.T)  
            ujj = np.inner(fd_coeff_l, component_y.T)
            ukk = np.inner(fd_coeff_l, component_z.T)

            # off-diagonal components.
            if sint == 'fd4noint_h':
                component_xy = np.array([u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
                component_xz = np.array([u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
                component_yz = np.array([u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:]])
            elif sint == 'fd6noint_h':
                component_xy = np.array([u[ix[2],ix[1]+3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]-3,:],u[ix[2],ix[1]+3,ix[0]-3,:],
                                         u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
                component_xz = np.array([u[ix[2]+3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]-3,:],u[ix[2]+3,ix[1],ix[0]-3,:],
                                         u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
                component_yz = np.array([u[ix[2]+3,ix[1]+3,ix[0],:],u[ix[2]-3,ix[1]+3,ix[0],:],u[ix[2]-3,ix[1]-3,ix[0],:],u[ix[2]+3,ix[1]-3,ix[0],:],
                                         u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:]])
            elif sint == 'fd8noint_h':
                component_xy = np.array([u[ix[2],ix[1]+4,ix[0]+4,:],u[ix[2],ix[1]-4,ix[0]+4,:],u[ix[2],ix[1]-4,ix[0]-4,:],u[ix[2],ix[1]+4,ix[0]-4,:],
                                         u[ix[2],ix[1]+3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]+3,:],u[ix[2],ix[1]-3,ix[0]-3,:],u[ix[2],ix[1]+3,ix[0]-3,:],
                                         u[ix[2],ix[1]+2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]+2,:],u[ix[2],ix[1]-2,ix[0]-2,:],u[ix[2],ix[1]+2,ix[0]-2,:],
                                         u[ix[2],ix[1]+1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]+1,:],u[ix[2],ix[1]-1,ix[0]-1,:],u[ix[2],ix[1]+1,ix[0]-1,:]])
                component_xz = np.array([u[ix[2]+4,ix[1],ix[0]+4,:],u[ix[2]-4,ix[1],ix[0]+4,:],u[ix[2]-4,ix[1],ix[0]-4,:],u[ix[2]+4,ix[1],ix[0]-4,:],
                                         u[ix[2]+3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]+3,:],u[ix[2]-3,ix[1],ix[0]-3,:],u[ix[2]+3,ix[1],ix[0]-3,:],
                                         u[ix[2]+2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]+2,:],u[ix[2]-2,ix[1],ix[0]-2,:],u[ix[2]+2,ix[1],ix[0]-2,:],
                                         u[ix[2]+1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]+1,:],u[ix[2]-1,ix[1],ix[0]-1,:],u[ix[2]+1,ix[1],ix[0]-1,:]])
                component_yz = np.array([u[ix[2]+4,ix[1]+4,ix[0],:],u[ix[2]-4,ix[1]+4,ix[0],:],u[ix[2]-4,ix[1]-4,ix[0],:],u[ix[2]+4,ix[1]-4,ix[0],:],
                                         u[ix[2]+3,ix[1]+3,ix[0],:],u[ix[2]-3,ix[1]+3,ix[0],:],u[ix[2]-3,ix[1]-3,ix[0],:],u[ix[2]+3,ix[1]-3,ix[0],:],
                                         u[ix[2]+2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]+2,ix[0],:],u[ix[2]-2,ix[1]-2,ix[0],:],u[ix[2]+2,ix[1]-2,ix[0],:],
                                         u[ix[2]+1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]+1,ix[0],:],u[ix[2]-1,ix[1]-1,ix[0],:],u[ix[2]+1,ix[1]-1,ix[0],:]])
            
            uij = np.inner(fd_coeff_h, component_xy.T) 
            uik = np.inner(fd_coeff_h, component_xz.T) 
            ujk = np.inner(fd_coeff_h, component_yz.T)
            
            return np.stack((uii,uij,uik,ujj,ujk,ukk), axis = 1).flatten()
        elif sint in ['m1q4_g', 'm2q8_g']:
            # gradient spline differentiations.
            ix = p.astype(int) 
            fr = p - ix
            
            # function spline coefficients.
            gx = self.function_lookup_table[int(lookup_N * fr[0])]
            gy = self.function_lookup_table[int(lookup_N * fr[1])]
            gz = self.function_lookup_table[int(lookup_N * fr[2])]
            
            # gradient spline coefficients.
            gx_G = self.lookup_table[int(lookup_N * fr[0])]
            gy_G = self.lookup_table[int(lookup_N * fr[1])]
            gz_G = self.lookup_table[int(lookup_N * fr[2])]
            
            gk_x = np.einsum('i,j,k', gz, gy, gx_G)
            gk_y = np.einsum('i,j,k', gz, gy_G, gx)
            gk_z = np.einsum('i,j,k', gz_G, gy, gx)
            
            d = u[ix[2] - cube_min_index : ix[2] + cube_max_index + 1,
                  ix[1] - cube_min_index : ix[1] + cube_max_index + 1,
                  ix[0] - cube_min_index : ix[0] + cube_max_index + 1, :] / dx
            
            # dudx,dvdx,dwdx.
            dvdx = np.einsum('ijk,ijkl->l', gk_x, d)
            # dudy,dvdy,dwdy.
            dvdy = np.einsum('ijk,ijkl->l', gk_y, d)
            # dudz,dvdz,dwdz.
            dvdz = np.einsum('ijk,ijkl->l', gk_z, d)
            
            return np.stack((dvdx, dvdy, dvdz), axis = 1).flatten()
        elif sint == 'm2q8_h':
            # hessian spline differentiation.
            ix = p.astype('int')
            fr = p - ix
            
            # function spline coefficients.
            gx = self.function_lookup_table[int(lookup_N * fr[0])]
            gy = self.function_lookup_table[int(lookup_N * fr[1])]
            gz = self.function_lookup_table[int(lookup_N * fr[2])]
            
            # gradient spline coefficients.
            gx_G = self.gradient_lookup_table[int(lookup_N * fr[0])]
            gy_G = self.gradient_lookup_table[int(lookup_N * fr[1])]
            gz_G = self.gradient_lookup_table[int(lookup_N * fr[2])]
            
            # hessian spline coefficients.
            gx_GG = self.lookup_table[int(lookup_N * fr[0])]
            gy_GG = self.lookup_table[int(lookup_N * fr[1])]
            gz_GG = self.lookup_table[int(lookup_N * fr[2])]

            gk_xx = np.einsum('i,j,k', gz, gy, gx_GG)
            gk_yy = np.einsum('i,j,k', gz, gy_GG, gx)
            gk_zz = np.einsum('i,j,k', gz_GG, gy, gx)
            gk_xy = np.einsum('i,j,k', gz, gy_G, gx_G)
            gk_xz = np.einsum('i,j,k', gz_G, gy, gx_G)
            gk_yz = np.einsum('i,j,k', gz_G, gy_G, gx)     

            d = u[ix[2] - cube_min_index : ix[2] + cube_max_index + 1,
                  ix[1] - cube_min_index : ix[1] + cube_max_index + 1,
                  ix[0] - cube_min_index : ix[0] + cube_max_index + 1, :] / dx / dx

            uii = np.einsum('ijk,ijkl->l', gk_xx, d)
            ujj = np.einsum('ijk,ijkl->l', gk_yy, d)
            ukk = np.einsum('ijk,ijkl->l', gk_zz, d)
            uij = np.einsum('ijk,ijkl->l', gk_xy, d)
            uik = np.einsum('ijk,ijkl->l', gk_xz, d)
            ujk = np.einsum('ijk,ijkl->l', gk_yz, d)                              

            return np.stack((uii,uij,uik,ujj,ujk,ukk), axis = 1).flatten()
        elif sint in ['fd4lag4_g', 'fd4lag4_l']:
            # gradient and laplacian finite difference with function interpolation.
            ix = p.astype(int) 
            fr = p - ix      
            
            # function interpolation coefficients.
            gx = self.function_lookup_table[int(lookup_N * fr[0])]
            gy = self.function_lookup_table[int(lookup_N * fr[1])]
            gz = self.function_lookup_table[int(lookup_N * fr[2])]
            
            # finite difference coefficients.
            gx_F = self.lookup_table[int(lookup_N * fr[0])]
            gy_F = self.lookup_table[int(lookup_N * fr[1])]
            gz_F = self.lookup_table[int(lookup_N * fr[2])]
            
            gk_x = np.einsum('i,j,k', gz, gy, gx_F)           
            gk_y = np.einsum('i,j,k', gz, gy_F, gx)           
            gk_z = np.einsum('i,j,k', gz_F, gy, gx)

            d_x = u[ix[2] - 1 : ix[2] + 3, ix[1] - 1 : ix[1] + 3, ix[0] - 3 : ix[0] + 5, :]           
            d_y = u[ix[2] - 1 : ix[2] + 3, ix[1] - 3 : ix[1] + 5, ix[0] - 1 : ix[0] + 3, :]           
            d_z = u[ix[2] - 3 : ix[2] + 5, ix[1] - 1 : ix[1] + 3, ix[0] - 1 : ix[0] + 3, :]
            
            # dudx,dvdx,dwdx.
            dvdx = np.einsum('ijk,ijkl->l', gk_x, d_x)
            # dudy,dvdy,dwdy.
            dvdy = np.einsum('ijk,ijkl->l', gk_y, d_y)
            # dudz,dvdz,dwdz.
            dvdz = np.einsum('ijk,ijkl->l', gk_z, d_z)
            
            if sint == 'fd4lag4_g':
                return np.stack((dvdx, dvdy, dvdz), axis = 1).flatten()
            elif sint == 'fd4lag4_l':
                dudxyz = dvdx[0] + dvdy[0] + dvdz[0]
                dvdxyz = dvdx[1] + dvdy[1] + dvdz[1]
                dwdxyz = dvdx[2] + dvdy[2] + dvdz[2]

                return np.array([dudxyz, dvdxyz, dwdxyz])
        
    """
    common functions.
    """
    def read_pickle_file(self, pickle_filename):
        """
        read the pickle metadata file. first, try reading from the production copy. second, try reading from the backup copy.
        """
        try:
            # pickled file production filepath.
            pickle_file = self.pickle_dir.joinpath(pickle_filename)
        
            # try reading the pickled file.
            with open(pickle_file, 'rb') as pickled_filepath:
                return dill.load(pickled_filepath)
        except:
            try:
                # pickled file backup filepath.
                pickle_file = self.pickle_dir_backup.joinpath(pickle_filename)

                # try reading the pickled file.
                with open(pickle_file, 'rb') as pickled_filepath:
                    return dill.load(pickled_filepath)
            except:
                try:
                    # pickled file backup filepath.
                    pickle_file = self.pickle_dir_local.joinpath(pickle_filename)

                    # try reading the pickled file.
                    with open(pickle_file, 'rb') as pickled_filepath:
                        return dill.load(pickled_filepath)
                except:
                    raise Exception('metadata files are not accessible.')
    
    def get_file_for_mortoncode(self, cornercode):
        """
        querying the cached SQL metadata for the file for the specified morton code.
        """
        # query the cached SQL metadata for the user-specified grid point.
        t = self.cache[(self.cache['minLim'] <= cornercode) & (self.cache['maxLim'] >= cornercode)]
        t = t.iloc[0]
        f = self.filepaths[f'{t.ProductionDatabaseName}']
        return f, t.minLim, t.maxLim
    
    def open_zarr_file(self, db_file, db_file_backup, open_file_vars):
        """
        open the zarr file for reading. first, try reading from the production copy. second, try reading from the backup copy.
        """
        # assign the local variables.
        var_name, timepoint, timepoint_digits, dt = open_file_vars
        
        try:
            # try reading from the production file.
            return zarr.open(f'{db_file}_{str(timepoint).zfill(timepoint_digits)}.zarr{os.sep}{var_name}', dtype = dt, mode = 'r')
        except:
            try:
                # try reading from the backup file.
                return zarr.open(f'{db_file_backup}_{str(timepoint).zfill(timepoint_digits)}.zarr{os.sep}{var_name}', dtype = dt, mode = 'r')
            except:
                raise Exception(f'{db_file}_{str(timepoint).zfill(timepoint_digits)}.zarr{os.sep}{var_name} and the corresponding backup file are not accessible.')
    
    """
    getCutout functions.
    """
    def identify_single_database_file_sub_boxes(self, box):
        # initially assumes the user specified box contains points in different files. the box will be split up until all the points
        # in each sub-box are from a single database file.
        box_min_xyz = [axis_range[0] for axis_range in box]
        box_max_xyz = [axis_range[1] for axis_range in box]
        
        # map of the parts of the user-specified box that are found in each database file.
        single_file_boxes = defaultdict(lambda: defaultdict(list))

        # z-value of the origin point (bottom left corner) of the box.
        current_z = box_min_xyz[2]

        # kick out of the while loops when there are no more database files along an axis.
        while current_z <= box_max_xyz[2]:
            # y-value of the origin point of the box.
            current_y = box_min_xyz[1]

            while current_y <= box_max_xyz[1]:
                # x-value of the origin point of the box.
                current_x = box_min_xyz[0]

                while current_x <= box_max_xyz[0]:
                    # database file name and corresponding minimum morton limit for the origin point of the box.  
                    min_corner_xyz = [current_x, current_y, current_z]
                    min_corner_info = self.get_file_for_mortoncode(self.mortoncurve.pack(min_corner_xyz[0] % self.N, min_corner_xyz[1] % self.N, min_corner_xyz[2] % self.N))
                    min_corner_db_file = min_corner_info[0]
                    min_corner_basename = os.path.basename(min_corner_db_file)
                    database_file_disk = min_corner_db_file.split(os.sep)[self.database_file_disk_index]
                    box_minLim = min_corner_info[1]
                    max_corner_xyz = self.mortoncurve.unpack(min_corner_info[2])
                    
                    # calculate the number of periodic cubes that the user-specified box expands into beyond the boundary along each axis.
                    cube_ms = [math.floor(float(min_corner_xyz[q]) / float(self.N)) * self.N for q in range(3)]
                    
                    # specify the box that is fully inside a database file.
                    box = [[min_corner_xyz[i], min(max_corner_xyz[i] + cube_ms[i], box_max_xyz[i])] for i in range(3)]
                    
                    # retrieve the backup filepath. '' handles datasets with with no backup file copies.
                    min_corner_db_file_backup = ''
                    if min_corner_basename in self.filepaths_backup:
                        min_corner_db_file_backup = self.filepaths_backup[min_corner_basename]
                    
                    # add the box axes ranges to the map.
                    single_file_boxes[database_file_disk][min_corner_db_file, min_corner_db_file_backup].append(box)

                    # move to the next database file origin point.
                    current_x = max_corner_xyz[0] + cube_ms[0] + 1

                current_y = max_corner_xyz[1] + cube_ms[1] + 1

            current_z = max_corner_xyz[2] + cube_ms[2] + 1
    
        return single_file_boxes
        
    def read_database_files_sequentially(self, user_single_db_boxes):
        result_output_data = []
        # iterate over the hard disk drives that the database files are stored on.
        for database_file_disk in user_single_db_boxes:
            # read in the voxel data from all of the database files on this disk.
            result_output_data += self.get_iso_points(user_single_db_boxes[database_file_disk],
                                                      self.getcutout_vars, self.open_file_vars)
        
        return result_output_data
    
    def read_database_files_in_parallel_with_dask(self, user_single_db_boxes):
        # start the dask client for parallel processing.
        # -----
        # flag specifying if the cluster is a premade distributed cluster. assumed to be True to start.
        distributed_cluster = True
        try:
            # using a premade distributed cluster.
            import SciServer.Dask
            
            # attached cluster (when a cluster is created together with a new container).
            client = SciServer.Dask.getClient()
            # deletes data on the network and restarts the workers.
            client.restart()
            
            # get the current working directory for saving the zip file of turbulence processing functions to.
            data_dir = os.getcwd() + os.sep
            
            # upload the turbulence processing functions in the giverny folder to the workers.
            shutil.make_archive(data_dir + 'giverny', 'zip', root_dir = data_dir, base_dir = 'giverny' + os.sep)
            client.upload_file(data_dir + 'giverny.zip')
        except:
            print(f'Starting a local dask cluster...')
            sys.stdout.flush()
            
            # update the distributed_cluster flag to False.
            distributed_cluster = False
            
            # using a local cluster if there is no premade distributed cluster.
            cluster = LocalCluster(n_workers = self.dask_maximum_processes, processes = True, silence_logs = logging.ERROR)
            client = Client(cluster)
        
        # available workers.
        workers = list(client.scheduler_info()['workers'])
        num_workers = len(workers)
        
        print(f'Database files are being read in parallel...')
        sys.stdout.flush()
        
        result_output_data = []
        # iterate over the hard disk drives that the database files are stored on.
        for disk_index, database_file_disk in enumerate(user_single_db_boxes):
            worker = workers[disk_index % num_workers]
            
            # read in the voxel data from all of the database files on this disk.
            result_output_data.append(client.submit(self.get_iso_points, user_single_db_boxes[database_file_disk],
                                                    self.getcutout_vars, self.open_file_vars,
                                                    workers = worker, pure = False))
        
        # gather all of the results once they are finished being run in parallel by dask.
        result_output_data = client.gather(result_output_data)        
        # flattens result_output_data to match the formatting as when the data is processed sequentially.
        result_output_data = [element for result in result_output_data for element in result]
        
        # close the dask client.
        client.close()
        
        if distributed_cluster:
            # delete the giverny.zip file if using a premade distributed cluster.
            if os.path.exists(data_dir + 'giverny.zip'):
                os.remove(data_dir + 'giverny.zip')
        else:
            # close the cluster if a local cluster was created.
            cluster.close()
        
        return result_output_data
    
    def get_iso_points(self, user_single_db_boxes_disk_data,
                       getcutout_vars, open_file_vars):
        """
        retrieve the values for the specified var(iable) in the user-specified box and at the specified timepoint.
        """
        # assign the local variables.
        file_size = getcutout_vars
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        # iterate over the database files to read the data from.
        for db_file, db_file_backup in user_single_db_boxes_disk_data:
            zm = self.open_zarr_file(db_file, db_file_backup, open_file_vars)
            
            # iterate over the user box ranges corresponding to the morton voxels that will be read from this database file.
            for user_box_ranges in user_single_db_boxes_disk_data[db_file, db_file_backup]:
                # retrieve the minimum and maximum (x, y, z) coordinates of the database file box that is going to be read in.
                min_xyz = [axis_range[0] for axis_range in user_box_ranges]
                max_xyz = [axis_range[1] for axis_range in user_box_ranges]
                # adjust the user box ranges to file size indices.
                user_box_ranges = np.array(user_box_ranges) % file_size
                
                # append the cutout into local_output_data.
                local_output_data.append((zm[user_box_ranges[2][0] : user_box_ranges[2][1] + 1,
                                             user_box_ranges[1][0] : user_box_ranges[1][1] + 1,
                                             user_box_ranges[0][0] : user_box_ranges[0][1] + 1],
                                             min_xyz, max_xyz))
        
        return local_output_data
            
    """
    getData functions.
    """
    def identify_database_file_points(self, points):
        # vectorize the mortoncurve.pack function.
        v_morton_pack = np.vectorize(self.mortoncurve.pack, otypes = [int])
        
        # chunk cube size.
        chunk_cube_size = 64**3
        # empty array for subdividing chunk groups.
        empty_array = np.array([0, 0, 0])
        # chunk size array for subdividing chunk groups.
        chunk_size_array = np.array([self.chunk_size, self.chunk_size, self.chunk_size]) - 1
        
        # convert the points to the center point position within their own bucket.
        center_points = (points / self.dx % 1) + self.cube_min_index
        # convert the points to gridded datapoints.
        datapoints = np.floor(points / self.dx).astype(int) % self.N
        # calculate the minimum and maximum chunk (x, y, z) corner point for each point in datapoints.
        chunk_min_xyzs = ((datapoints - self.cube_min_index) - ((datapoints - self.cube_min_index) % self.chunk_size)) % self.N
        chunk_max_xyzs = ((datapoints + self.cube_max_index) + (self.chunk_size - ((datapoints + self.cube_max_index) % self.chunk_size) - 1)) % self.N
        # chunk volumes.
        chunk_volumes = np.prod(chunk_max_xyzs - chunk_min_xyzs + 1, axis = 1)
        # create the chunk keys for each chunk group.
        chunk_keys = [chunk_origin_group.tobytes() for chunk_origin_group in np.stack([chunk_min_xyzs, chunk_max_xyzs], axis = 1)]
        # convert chunk_min_xyzs and chunk_max_xyzs to indices in a single database file.
        chunk_min_mods = chunk_min_xyzs % self.file_size
        chunk_max_mods = chunk_max_xyzs % self.file_size
        # calculate the minimum and maximum chunk morton codes for each point in chunk_min_xyzs and chunk_max_xyzs.
        chunk_min_mortons = v_morton_pack(chunk_min_xyzs[:, 0], chunk_min_xyzs[:, 1], chunk_min_xyzs[:, 2])
        chunk_max_mortons = v_morton_pack(chunk_max_xyzs[:, 0], chunk_max_xyzs[:, 1], chunk_max_xyzs[:, 2])
        # calculate the db file cornercodes for each morton code.
        db_min_cornercodes = (chunk_min_mortons >> 27) << 27
        db_max_cornercodes = (chunk_max_mortons >> 27) << 27
        # identify the database files that will need to be read for each chunk.
        db_min_files = [self.cornercode_file_map[morton_code] for morton_code in db_min_cornercodes]
        db_max_files = [self.cornercode_file_map[morton_code] for morton_code in db_max_cornercodes]
        # retrieve the backup filepaths. '' handles datasets with with no backup file copies.
        db_min_files_backup = [self.filepaths_backup[os.path.basename(db_min_file)] if os.path.basename(db_min_file) in self.filepaths_backup else '' \
                               for db_min_file in db_min_files]
        
        # save the original indices for points, which corresponds to the orderering of the user-specified
        # points. these indices will be used for sorting output_data back to the user-specified points ordering.
        original_points_indices = [q for q in range(len(points))]
        # zip the data. sort by volume first so that all fully overlapped chunk groups can be easily found.
        zipped_data = sorted(zip(chunk_volumes, chunk_min_mortons, chunk_keys, db_min_files, db_max_files, db_min_files_backup, points, datapoints, center_points,
                                 chunk_min_xyzs, chunk_max_xyzs, chunk_min_mods, chunk_max_mods, original_points_indices), key = lambda x: (-1 * x[0], x[1], x[2]))
        
        # map the native bucket points to their respective db files and chunks.
        db_native_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # store an array of visitor bucket points.
        db_visitor_map = []
        # chunk key map used for storing all subdivided chunk groups to find fully overlapped chunk groups.
        chunk_map = {}
        
        for chunk_volume, chunk_min_morton, chunk_key, db_min_file, db_max_file, db_min_file_backup, point, datapoint, center_point, \
            chunk_min_xyz, chunk_max_xyz, chunk_min_mod, chunk_max_mod, original_point_index in zipped_data:
            # update the database file info if the morton code is outside of the previous database fil maximum morton limit.
            db_disk = db_min_file.split(os.sep)[self.database_file_disk_index]
            
            if db_min_file == db_max_file:
                # update the chunk key if the chunk group is fully contained in another larger chunk group.
                updated_chunk_key = chunk_key
                if chunk_key in chunk_map:
                    updated_chunk_key = chunk_map[chunk_key]
                elif chunk_volume != chunk_cube_size:
                    chunk_map = self.subdivide_chunk_group(chunk_map, chunk_key, chunk_min_xyz, chunk_max_xyz, chunk_size_array, empty_array)
                
                # assign to native map.
                if updated_chunk_key not in db_native_map[db_disk][db_min_file, db_min_file_backup]:
                    db_native_map[db_disk][db_min_file, db_min_file_backup][updated_chunk_key].append((chunk_min_xyz, chunk_max_xyz, chunk_min_mod, chunk_max_mod))
    
                db_native_map[db_disk][db_min_file, db_min_file_backup][updated_chunk_key].append((point, datapoint, center_point, original_point_index))
            else:
                # assign to the visitor map.
                db_visitor_map.append((point, datapoint, center_point, original_point_index))
        
        return db_native_map, np.array(db_visitor_map, dtype = 'object')
    
    def subdivide_chunk_group(self, chunk_map, chunk_key, chunk_min_xyz, chunk_max_xyz, chunk_size_array, empty_array): 
        chunk_mins = []
        chunk_maxs = []

        # axes that are 2 chunks in length.
        chunk_diffs = np.where(chunk_max_xyz - chunk_min_xyz + 1 == 2 * self.chunk_size)[0]
        num_long_axes = len(chunk_diffs)

        # 1-cubes, which are needed for all chunk groups (2, 4, or 8 chunks).
        # long axis 1, first 1-cube.
        chunk_mins.append(chunk_min_xyz)
        new_max = chunk_min_xyz + chunk_size_array
        chunk_maxs.append(new_max)

        # long axis 1, second 1-cube.
        new_min = chunk_min_xyz + empty_array
        new_min[chunk_diffs[0]] += self.chunk_size
        new_max = chunk_min_xyz + chunk_size_array
        new_max[chunk_diffs[0]] += self.chunk_size
        chunk_mins.append(new_min)
        chunk_maxs.append(new_max)
        
        # add additional sub-chunks chunk group contains 4 or 8 chunks.
        if num_long_axes == 2 or num_long_axes == 3:
            # 1-cubes, additional.
            # long axis 2, first 1-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[1]] += self.chunk_size
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[1]] += self.chunk_size
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)

            # long axis 2, second 1-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[0]] += self.chunk_size
            new_min[chunk_diffs[1]] += self.chunk_size
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size
            new_max[chunk_diffs[1]] += self.chunk_size
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)

            # 2-cubes.
            # long axis 1, first 2-cube.
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size
            chunk_mins.append(chunk_min_xyz)
            chunk_maxs.append(new_max)

            # long axis 1, second 2-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[1]] += self.chunk_size
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size
            new_max[chunk_diffs[1]] += self.chunk_size
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)
            
            # long axis 2, first 2-cube.
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[1]] += self.chunk_size
            chunk_mins.append(chunk_min_xyz)
            chunk_maxs.append(new_max)

            # long axis 2, second 2-cube.
            new_min = chunk_min_xyz + empty_array
            new_min[chunk_diffs[0]] += self.chunk_size
            new_max = chunk_min_xyz + chunk_size_array
            new_max[chunk_diffs[0]] += self.chunk_size
            new_max[chunk_diffs[1]] += self.chunk_size
            chunk_mins.append(new_min)
            chunk_maxs.append(new_max)
        
            if num_long_axes == 3:
                # 1-cubes, additional.
                # long axis 3, first 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, second 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, third 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, fourth 1-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_min[chunk_diffs[1]] += self.chunk_size
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # 2-cubes, additional.
                # long axis 1, third 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 1, fourth 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)
                
                # long axis 2, third 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 2, fourth 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)
                
                # long axis 3, first 2-cube.
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(chunk_min_xyz)
                chunk_maxs.append(new_max)

                # long axis 3, second 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, third 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axis 3, fourth 2-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_min[chunk_diffs[1]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # 4-cubes.
                # long axes 1 and 2, first 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 1 and 2, second 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[2]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 1 and 3, first 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 1 and 3, second 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[1]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 2 and 3, first 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

                # long axes 2 and 3, second 4-cube.
                new_min = chunk_min_xyz + empty_array
                new_min[chunk_diffs[0]] += self.chunk_size
                new_max = chunk_min_xyz + chunk_size_array
                new_max[chunk_diffs[0]] += self.chunk_size
                new_max[chunk_diffs[1]] += self.chunk_size
                new_max[chunk_diffs[2]] += self.chunk_size
                chunk_mins.append(new_min)
                chunk_maxs.append(new_max)

        # whole cube.
        chunk_mins.append(chunk_min_xyz)
        chunk_maxs.append(chunk_max_xyz)

        # convert to numpy arrays.
        chunk_mins = np.array(chunk_mins)
        chunk_maxs = np.array(chunk_maxs)

        # update chunk_map with all of the new keys.
        chunk_keys = [chunk_origin_group.tobytes() for chunk_origin_group in np.stack([chunk_mins, chunk_maxs], axis = 1)]
        for key in chunk_keys:
            chunk_map[key] = chunk_key

        return chunk_map
    
    def get_chunk_origin_groups(self, chunk_min_x, chunk_min_y, chunk_min_z, chunk_max_x, chunk_max_y, chunk_max_z, N, chunk_size):
        # get arrays of the chunk origin points for each bucket.
        return np.array([[x, y, z]
                         for z in range(chunk_min_z, (chunk_max_z if chunk_min_z <= chunk_max_z else N + chunk_max_z) + 1, chunk_size)
                         for y in range(chunk_min_y, (chunk_max_y if chunk_min_y <= chunk_max_y else N + chunk_max_y) + 1, chunk_size)
                         for x in range(chunk_min_x, (chunk_max_x if chunk_min_x <= chunk_max_x else N + chunk_max_x) + 1, chunk_size)])
    
    def read_natives_sequentially_variable(self, db_native_map, native_output_data):
        # native data.
        # iterate over the data volumes that the database files are stored on.
        for database_file_disk in db_native_map:
            # read in the voxel data from all of the database files on this disk.
            native_output_data += self.get_iso_points_variable(db_native_map[database_file_disk],
                                                               self.open_file_vars, self.interpolate_vars)
            
    def read_visitors_in_parallel_variable(self, db_visitor_map, visitor_output_data):
        # visitor data.
        if len(db_visitor_map) != 0:
            distributed_cluster = True
            try:
                # using a premade distributed cluster.
                import SciServer.Dask

                # attached cluster (when a cluster is created together with a new container).
                client = SciServer.Dask.getClient()
                # deletes data on the network and restarts the workers.
                client.restart()

                # get the current working directory for saving the zip file of turbulence processing functions to.
                data_dir = os.getcwd() + os.sep

                # upload the turbulence processing functions in the giverny folder to the workers.
                shutil.make_archive(data_dir + 'giverny', 'zip', root_dir = data_dir, base_dir = 'giverny' + os.sep)
                client.upload_file(data_dir + 'giverny.zip')
            except:
                print(f'Starting a local dask cluster...')
                sys.stdout.flush()
            
                # update the distributed_cluster flag to False.
                distributed_cluster = False

                # using a local cluster if there is no premade distributed cluster.
                cluster = LocalCluster(n_workers = self.dask_maximum_processes, processes = True, silence_logs = logging.ERROR)
                client = Client(cluster)
    
            # available workers.
            workers = list(client.scheduler_info()['workers'])
            num_workers = len(workers)
            
            print(f'Database files are being read in parallel...')
            sys.stdout.flush()
            
            # calculate how many chunks to use for splitting up the visitor map data.
            num_visitor_points = len(db_visitor_map)
            num_chunks = num_workers
            if num_visitor_points < num_workers:
                num_chunks = num_visitor_points

            # chunk db_visitor_map.
            db_visitor_map_split = np.array(np.array_split(db_visitor_map, num_chunks), dtype = object)

            temp_visitor_output_data = []
            # scatter the chunks to their own worker and submit each chunk for parallel processing.
            for db_visitor_map_chunk, worker in zip(db_visitor_map_split, workers):
                # submit the chunk for parallel processing.
                temp_visitor_output_data.append(client.submit(self.get_iso_points_variable_visitor, db_visitor_map_chunk,
                                                              self.getdata_vars, self.open_file_vars, self.interpolate_vars,
                                                              workers = worker, pure = False))
                
            # gather all of the results once they are finished being run in parallel by dask.
            temp_visitor_output_data = client.gather(temp_visitor_output_data)
            # flattens result_output_data to match the formatting as when the data is processed sequentially.
            temp_visitor_output_data = [element for result in temp_visitor_output_data for element in result]

            # update visitor_output_data.
            visitor_output_data += temp_visitor_output_data
            
            # close the dask client.
            client.close()

            if distributed_cluster:
                # delete the giverny.zip file if using a premade distributed cluster.
                if os.path.exists(data_dir + 'giverny.zip'):
                    os.remove(data_dir + 'giverny.zip')
            else:
                # close the cluster if a local cluster was created.
                cluster.close() 
            
    def read_database_files_sequentially_variable(self, db_native_map, db_visitor_map):
        if len(db_visitor_map) == 0:
            print('Database files are being read sequentially...')
            sys.stdout.flush()
        
        # create empty lists for filling the output data.
        native_output_data = []
        visitor_output_data = []
        
        # create threads for parallel processing of the native and visitor data.
        native_thread = Thread(target = self.read_natives_sequentially_variable, args = (db_native_map, native_output_data))
        visitor_thread = Thread(target = self.read_visitors_in_parallel_variable, args = (db_visitor_map, visitor_output_data))
            
        # start the threads.
        native_thread.start()
        visitor_thread.start()

        # wait for the threads to complete.
        native_thread.join()
        visitor_thread.join()
            
        result_output_data = native_output_data + visitor_output_data
            
        return result_output_data
    
    def read_database_files_in_parallel_with_dask_variable(self, db_native_map, db_visitor_map):
        # start the dask client for parallel processing.
        # -----
        # flag specifying if the cluster is a premade distributed cluster. assumed to be True to start.
        distributed_cluster = True
        try:
            # using a premade distributed cluster.
            import SciServer.Dask
            
            # attached cluster (when a cluster is created together with a new container).
            client = SciServer.Dask.getClient()
            # deletes data on the network and restarts the workers.
            client.restart()
            
            # get the current working directory for saving the zip file of turbulence processing functions to.
            data_dir = os.getcwd() + os.sep
            
            # upload the turbulence processing functions in the giverny folder to the workers.
            shutil.make_archive(data_dir + 'giverny', 'zip', root_dir = data_dir, base_dir = 'giverny' + os.sep)
            client.upload_file(data_dir + 'giverny.zip')
        except:
            print(f'Starting a local dask cluster...')
            sys.stdout.flush()
            
            # update the distributed_cluster flag to False.
            distributed_cluster = False
            
            # using a local cluster if there is no premade distributed cluster.
            cluster = LocalCluster(n_workers = self.dask_maximum_processes, processes = True, silence_logs = logging.ERROR)
            client = Client(cluster)
        
        # available workers.
        workers = list(client.scheduler_info()['workers'])
        num_workers = len(workers)
        
        print(f'Database files are being read in parallel...')
        sys.stdout.flush()
        
        result_output_data = []
        # native buckets.
        # -----
        if len(db_native_map) != 0:
            # iterate over the db volumes.
            for disk_index, database_file_disk in enumerate(db_native_map):
                worker = workers[disk_index % num_workers] 
                
                # submit the data for parallel processing.
                result_output_data.append(client.submit(self.get_iso_points_variable, db_native_map[database_file_disk],
                                                        self.open_file_vars, self.interpolate_vars,
                                                        workers = worker, pure = False))
        
        # visitor buckets.
        # -----
        if len(db_visitor_map) != 0:
            # calculate how many chunks to use for splitting up the visitor map data.
            num_visitor_points = len(db_visitor_map)
            num_chunks = num_workers
            if num_visitor_points < num_workers:
                num_chunks = num_visitor_points

            # chunk db_visitor_map.
            db_visitor_map_split = np.array(np.array_split(db_visitor_map, num_chunks), dtype = object)

            # scatter the chunks to their own worker and submit the chunk for parallel processing.
            for db_visitor_map_chunk, worker in zip(db_visitor_map_split, workers):
                # submit the data for parallel processing.
                result_output_data.append(client.submit(self.get_iso_points_variable_visitor, db_visitor_map_chunk,
                                                        self.getdata_vars, self.open_file_vars, self.interpolate_vars,
                                                        workers = worker, pure = False))
        
        # gather all of the results once they are finished being run in parallel by dask.
        result_output_data = client.gather(result_output_data)        
        # flattens result_output_data to match the formatting as when the data is processed sequentially.
        result_output_data = [element for result in result_output_data for element in result]
             
        # close the dask client.
        client.close()
        
        if distributed_cluster:
            # delete the giverny.zip file if using a premade distributed cluster.
            if os.path.exists(data_dir + 'giverny.zip'):
                os.remove(data_dir + 'giverny.zip')
        else:
            # close the cluster if a local cluster was created.
            cluster.close()
                
        return result_output_data
    
    def get_iso_points_variable(self, db_native_map_data,
                                open_file_vars, interpolate_vars):
        """
        reads and interpolates the user-requested native points in a single database volume.
        """
        # assign the local variables.
        cube_min_index, cube_max_index, sint = interpolate_vars[:3]
        
        # initialize the interpolation lookup table.
        self.init_interpolation_lookup_table(sint = sint, read_metadata = True)
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        # iterate over the database files and morton sub-boxes to read the data from.
        for db_file, db_file_backup in db_native_map_data:
            zs = self.open_zarr_file(db_file, db_file_backup, open_file_vars)
            
            db_file_data = db_native_map_data[db_file, db_file_backup]
            for chunk_key in db_file_data:
                chunk_data = db_file_data[chunk_key]
                chunk_min_xyz = chunk_data[0][0]
                chunk_max_xyz = chunk_data[0][1]

                # read in the necessary chunks.
                zm = zs[chunk_data[0][2][2] : chunk_data[0][3][2] + 1,
                        chunk_data[0][2][1] : chunk_data[0][3][1] + 1,
                        chunk_data[0][2][0] : chunk_data[0][3][0] + 1]
                
                # iterate over the points to interpolate.
                for point, datapoint, center_point, original_point_index in chunk_data[1:]:
                    bucket_min_xyz = datapoint - chunk_min_xyz - cube_min_index
                    bucket_max_xyz = datapoint - chunk_min_xyz + cube_max_index + 1

                    bucket = zm[bucket_min_xyz[2] : bucket_max_xyz[2],
                                bucket_min_xyz[1] : bucket_max_xyz[1],
                                bucket_min_xyz[0] : bucket_max_xyz[0]]
            
                    # interpolate the points and use a lookup table for faster interpolations.
                    local_output_data.append((original_point_index, (point, self.spatial_interpolate(center_point, bucket, interpolate_vars))))
        
        return local_output_data
    
    def get_iso_points_variable_visitor(self, visitor_data,
                                        getdata_vars, open_file_vars, interpolate_vars):
        """
        reads and interpolates the user-requested visitor points.
        """
        # assign the local variables.
        cube_min_index, cube_max_index, sint = interpolate_vars[:3]
        dataset_title, num_values_per_datapoint, N, chunk_size, file_size = getdata_vars
        
        # get map of the filepaths for all of the dataset files.
        self.init_filepaths(dataset_title)
        
        # get a map of the files to cornercodes for all of the dataset files.
        self.init_cornercode_file_map(dataset_title, N)
        
        # initialize the interpolation lookup table.
        self.init_interpolation_lookup_table(sint = sint, read_metadata = True)
        
        # vectorize the mortoncurve.pack function.
        v_morton_pack = np.vectorize(self.mortoncurve.pack, otypes = [int])

        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []

        # empty chunk group array (up to eight 64-cube chunks).
        zm = np.zeros((128, 128, 128, num_values_per_datapoint))

        datapoints = np.array([datapoint for datapoint in visitor_data[:, 1]])
        # calculate the minimum and maximum chunk group corner point (x, y, z) for each datapoint.
        chunk_min_xyzs = ((datapoints - cube_min_index) - ((datapoints - cube_min_index) % chunk_size)) % N
        chunk_max_xyzs = ((datapoints + cube_max_index) + (chunk_size - ((datapoints + cube_max_index) % chunk_size) - 1)) % N
        # calculate the morton codes for the minimum (x, y, z) point of each chunk group.
        chunk_min_mortons = v_morton_pack(chunk_min_xyzs[:, 0], chunk_min_xyzs[:, 1], chunk_min_xyzs[:, 2])
        # create the chunk keys for each chunk group.
        chunk_keys = [chunk_origin_group.tobytes() for chunk_origin_group in np.stack([chunk_min_xyzs, chunk_max_xyzs], axis = 1)]
        # calculate the minimum and maximum bucket corner point (x, y, z) for each datapoint.
        bucket_min_xyzs = (datapoints - chunk_min_xyzs - cube_min_index) % N
        bucket_max_xyzs = (datapoints - chunk_min_xyzs + cube_max_index + 1) % N
        # create the bucket keys for each interpolation bucket.
        bucket_keys = [bucket_origin_group.tobytes() for bucket_origin_group in np.stack([bucket_min_xyzs, bucket_max_xyzs], axis = 1)]

        current_chunk = ''
        current_bucket = ''
        for point_data, chunk_min_morton, chunk_min_xyz, chunk_max_xyz, chunk_key, bucket_min_xyz, bucket_max_xyz, bucket_key in \
            sorted(zip(visitor_data, chunk_min_mortons, chunk_min_xyzs, chunk_max_xyzs, chunk_keys, bucket_min_xyzs, bucket_max_xyzs, bucket_keys),
                   key = lambda x: (x[1], x[4], x[7])):
            if current_chunk != chunk_key:
                # get the origin points for each voxel in the bucket.
                chunk_origin_groups = self.get_chunk_origin_groups(chunk_min_xyz[0], chunk_min_xyz[1], chunk_min_xyz[2],
                                                                   chunk_max_xyz[0], chunk_max_xyz[1], chunk_max_xyz[2],
                                                                   N, chunk_size)
                # adjust the chunk origin points to the chunk domain size for filling the empty chunk group array.
                chunk_origin_points = chunk_origin_groups - chunk_origin_groups[0]

                # get the chunk origin group inside the dataset domain.
                chunk_origin_groups = chunk_origin_groups % N
                # calculate the morton codes for the minimum point in each chunk of the chunk groups.
                morton_mins = v_morton_pack(chunk_origin_groups[:, 0], chunk_origin_groups[:, 1], chunk_origin_groups[:, 2])
                # get the chunk origin group inside the file domain.
                chunk_origin_groups = chunk_origin_groups % file_size

                # calculate the db file cornercodes for each morton code.
                db_cornercodes = (morton_mins >> 27) << 27
                # identify the database files that will need to be read for this bucket.
                db_files = [self.cornercode_file_map[morton_code] for morton_code in db_cornercodes]
                # retrieve the backup filepaths. '' handles datasets with with no backup file copies.
                db_files_backup = [self.filepaths_backup[os.path.basename(db_file)] if os.path.basename(db_file) in self.filepaths_backup else '' \
                                   for db_file in db_files]

                current_file = ''
                # iterate of the db files.
                for db_file, db_file_backup, chunk_origin_point, chunk_origin_group in sorted(zip(db_files, db_files_backup, chunk_origin_points, chunk_origin_groups),
                                                                                              key = lambda x: x[0]):
                    if db_file != current_file:
                        # create an open file object of the database file.
                        zs = self.open_zarr_file(db_file, db_file_backup, open_file_vars)

                        # update current_file.
                        current_file = db_file

                    zm[chunk_origin_point[2] : chunk_origin_point[2] + chunk_size,
                       chunk_origin_point[1] : chunk_origin_point[1] + chunk_size,
                       chunk_origin_point[0] : chunk_origin_point[0] + chunk_size] = zs[chunk_origin_group[2] : chunk_origin_group[2] + chunk_size,
                                                                                        chunk_origin_group[1] : chunk_origin_group[1] + chunk_size,
                                                                                        chunk_origin_group[0] : chunk_origin_group[0] + chunk_size]

                # update current_chunk.
                current_chunk = chunk_key

            if current_bucket != bucket_key:
                bucket = zm[bucket_min_xyz[2] : bucket_max_xyz[2],
                            bucket_min_xyz[1] : bucket_max_xyz[1],
                            bucket_min_xyz[0] : bucket_max_xyz[0]]

                # update current_bucket.
                current_bucket = bucket_key

            # interpolate the point and use a lookup table for faster interpolation.
            local_output_data.append((point_data[3], (point_data[0], self.spatial_interpolate(point_data[2], bucket, interpolate_vars))))

        return local_output_data
    