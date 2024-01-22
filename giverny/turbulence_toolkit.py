import time
import tracemalloc
import numpy as np
import pandas as pd
from giverny.isotropic_cube import *
from giverny.turbulence_toolkit import *
from giverny.turbulence_gizmos.constants import *
from giverny.turbulence_gizmos.basic_gizmos import *

"""
retrieve a cutout of the isotropic cube.
"""
def getCutout(cube, var_original, timepoint_original, axes_ranges_original, strides,
              trace_memory = False):
    from giverny.turbulence_gizmos.getCutout import getCutout_process_data
    
    # housekeeping procedures.
    # -----
    var, axes_ranges, timepoint = \
        getCutout_housekeeping_procedures(cube, axes_ranges_original, strides, var_original, timepoint_original)
    
    # process the data.
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)

    # parse the database files, generate the output_data matrix, and write the matrix to an hdf5 file.
    output_data = getCutout_process_data(cube, axes_ranges, var, timepoint,
                                         axes_ranges_original, strides, var_original, timepoint_original)
    
    # closing the tracemalloc library.
    if trace_memory:
        # memory used during processing as calculated by tracemalloc.
        tracemem_end = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_end = tracemalloc.get_tracemalloc_memory() / (1024**3)
        # stopping the tracemalloc library.
        tracemalloc.stop()

        # see how much memory was used during processing.
        # memory used at program start.
        print(f'\nstarting memory used in GBs [current, peak] = {tracemem_start}')
        # memory used by tracemalloc.
        print(f'starting memory used by tracemalloc in GBs = {tracemem_used_start}')
        # memory used during processing.
        print(f'ending memory used in GBs [current, peak] = {tracemem_end}')
        # memory used by tracemalloc.
        print(f'ending memory used by tracemalloc in GBs = {tracemem_used_end}')
    
    return output_data

"""
complete all of the getCutout housekeeping procedures before data processing.
    - convert 1-based axes ranges to 0-based.
    - format the variable name and get the variable identifier.
    - convert 1-based timepoint to 0-based.
"""
def getCutout_housekeeping_procedures(cube, axes_ranges_original, strides, var_original, timepoint_original):
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(var_original, cube.dataset_title)
    # check that the user-input timepoint is a valid timepoint for the dataset.
    check_timepoint(timepoint_original, cube.dataset_title)
    # check that the user-input x-, y-, and z-axis ranges are all specified correctly as [minimum, maximum] integer values.
    check_axes_ranges(axes_ranges_original)
    # check that the user-input strides are all positive integers.
    check_strides(strides)
    
    # pre-processing steps.
    # -----
    # converts the 1-based axes ranges above to 0-based axes ranges, and truncates the ranges if they are longer than 
    # the cube resolution (cube.N) since the boundaries are periodic. output_data will be filled in with the duplicate data 
    # for the truncated data points after processing so that the data files are not read redundantly.
    axes_ranges = convert_to_0_based_ranges(axes_ranges_original, cube.N)

    # convert the variable name from var_original into a variable identifier.
    var = get_variable_identifier(var_original)
    
    # converts the 1-based timepoint above to a 0-based timepoint.
    timepoint = convert_to_0_based_value(timepoint_original)
    
    return (var, axes_ranges, timepoint)

"""
interpolate/differentiate the variable for the specified points from the various JHTDB datasets.
"""
def getData(cube, var_original, timepoint_original, temporal_method_original, spatial_method_original, spatial_operator_original, points,
            option = None, trace_memory = False):
    print('\n' + '-' * 25 + '\ngetData is processing...')
    sys.stdout.flush()
    
    # calculate how much time it takes to run the code.
    start_time = time.perf_counter()
    
    # set attributes.
    dataset_title = cube.dataset_title
    lJHTDB = cube.lJHTDB
    
    # -----
    # housekeeping procedures. will handle multiple variables, e.g. 'pressure' and 'velocity'.
    var, timepoint, temporal_method, spatial_method, spatial_operator = getData_housekeeping_procedures(dataset_title, points, var_original, timepoint_original,
                                                                                                        temporal_method_original, spatial_method_original, spatial_operator_original, option)
    
    # data constants.
    c = get_constants()

    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)
    
    # initialize cube constants.
    cube.init_constants(var, var_original, timepoint, spatial_method, temporal_method, num_values_per_datapoint, c)
    
    # get the full variable name for determining the datatype.
    datatype_var = get_output_variable_name(var_original)
    
    # remove 'function' from operator for determining the datatype.
    datatype_operator = spatial_operator if spatial_operator != 'function' else ''
    
    # define datatype from the datatype_var and datatype_operator variables.
    datatype = f'{datatype_var}{datatype_operator.title()}'
    
    # spatial interpolation map for legacy datasets.
    spatial_map = { 
        'none': 0, 'lag4': 4, 'lag6': 6, 'lag8': 8,
        'fd4noint': 40, 'fd6noint': 60, 'fd8noint': 80,
        'fd4lag4': 44,
        'm1q4': 104, 'm1q6': 106, 'm1q8': 108, 'm1q10': 110, 'm1q12': 112, 'm1q14': 114,
        'm2q4': 204, 'm2q6': 206, 'm2q8': 208, 'm2q10': 210, 'm2q12': 212, 'm2q14': 214,
        'm3q4': 304, 'm3q6': 306, 'm3q8': 308, 'm3q10': 310, 'm3q12': 312, 'm3q14': 314,
        'm4q4': 404, 'm4q6': 406, 'm4q8': 408, 'm4q10': 410, 'm4q12': 412, 'm4q14': 414
    }

    # temporal interpolation map for legacy datasets.
    temporal_map = {
        'none': 0,
        'pchip': 1
    }
    
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)
    
    # process the data query.
    if datatype == 'Position':
        timepoint_end, delta_t = option
        
        # set the number of steps to keep to 1. for now this will not be a user-modifiable parameter.
        steps_to_keep = 1
        
        # formatting the output since getPosition prints output, whereas lJHTDB.getData does not.
        print()
        
        # only returning the position array ('result') to keep consistent with other getData variables. the time array can be calculated in the notebook if needed
        # as t = np.linspace(timepoint, timepoint_end, steps_to_keep + 1).astype(np.float32).
        result, t = lJHTDB.getPosition(data_set = dataset_title,
                                       starttime = timepoint, endtime = timepoint_end, dt = delta_t,
                                       point_coords = points, steps_to_keep = steps_to_keep)
        
        # only return the final point positions to keep consistent with the other "get" functions.
        result = result[-1]
    else:
        # retrieve the list of datasets processed by the giverny code.
        giverny_datasets = get_giverny_datasets()
        
        # retrieve interpolation/differentiation results for the various datasets.
        if dataset_title in giverny_datasets:
            from giverny.turbulence_gizmos.getData import getData_process_data
            
            # get the results.
            result = getData_process_data(cube, points, var, timepoint, temporal_method, spatial_method, var_original, timepoint_original)
        else:
            # get the spatial and temporal interpolation integers for the legacy datasets.
            sint = spatial_map[spatial_method_original]
            tint = temporal_map[temporal_method_original]
    
            # get the results.
            result = lJHTDB.getData(timepoint, points, data_set = dataset_title, sinterp = sint, tinterp = tint, getFunction = f'get{datatype}')
    
    # insert the output header at the beginning of output_data.
    output_header = get_interpolation_tsv_header(cube.dataset_title, cube.var_name, timepoint_original, cube.sint, cube.tint)
    result_header = np.array(output_header.split('\n')[1].strip().split('\t'))[3:]
    result = pd.DataFrame(data = result, columns = result_header)
    
    # -----
    end_time = time.perf_counter()
    
    print(f'\ntotal time elapsed = {end_time - start_time:0.3f} seconds ({(end_time - start_time) / 60:0.3f} minutes)')
    sys.stdout.flush()
    
    print('\nData processing pipeline has completed successfully.\n' + '-' * 5)
    sys.stdout.flush()
    
    # closing the tracemalloc library.
    if trace_memory:
        # memory used during processing as calculated by tracemalloc.
        tracemem_end = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_end = tracemalloc.get_tracemalloc_memory() / (1024**3)
        # stopping the tracemalloc library.
        tracemalloc.stop()

        # see how much memory was used during processing.
        # memory used at program start.
        print(f'\nstarting memory used in GBs [current, peak] = {tracemem_start}')
        # memory used by tracemalloc.
        print(f'starting memory used by tracemalloc in GBs = {tracemem_used_start}')
        # memory used during processing.
        print(f'ending memory used in GBs [current, peak] = {tracemem_end}')
        # memory used by tracemalloc.
        print(f'ending memory used by tracemalloc in GBs = {tracemem_used_end}')
    
    return result

"""
complete all of the housekeeping procedures before data processing.
    - format the variable name and get the variable identifier.
    - convert 1-based timepoint to 0-based.
"""
def getData_housekeeping_procedures(dataset_title, points, var_original, timepoint_original, temporal_method, spatial_method, spatial_operator, option):
    # validate user-input.
    # -----
    # check that dataset_title is a valid dataset title.
    check_dataset_title(dataset_title)
    # check that the points are all within axes domain for the dataset.
    check_points_domain(dataset_title, points)
    # check that the user-input variable is a valid variable name.
    check_variable(var_original, dataset_title)
    # check that the user-input timepoint is a valid timepoint for the dataset.
    check_timepoint(timepoint_original, dataset_title)
    # check that the user-input interpolation spatial operator (spatial_operator) is a valid interpolation operator.
    check_operator(spatial_operator, var_original)
    # check that the user-input spatial interpolation (spatial_method) is a valid spatial interpolation method.
    spatial_method = check_spatial_interpolation(dataset_title, var_original, spatial_method, spatial_operator)
    # check that the user-input temporal interpolation (temporal_method) is a valid temporal interpolation method.
    check_temporal_interpolation(dataset_title, var_original, temporal_method)
    # check that option != None if var_original = 'position'.
    if var_original == 'position':
        check_option_parameter(option, timepoint_original)
    
    # pre-processing steps.
    # -----
    # convert the variable name from var_original into a variable identifier.
    var = get_variable_identifier(var_original)
    
    # retrieve the list of datasets which use time indices.
    time_index_datasets = get_time_index_datasets()
        
    # converts the 1-based timepoint above to a 0-based timepoint if the timepoint is for a dataset with time indices.
    timepoint = timepoint_original
    if dataset_title in time_index_datasets:
        timepoint = convert_to_0_based_value(timepoint_original)
    
    return (var, timepoint, temporal_method, spatial_method, spatial_operator)
