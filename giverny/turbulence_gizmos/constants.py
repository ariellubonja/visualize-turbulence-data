"""
data constants.
"""
def get_constants():
    """
    data constants:
        - database_file_disk_index: index that pulls out the parent folder for a database file when the filepath is split on each 
                                    folder separator. the parent folder references the hard disk drive that the database file is stored on.
        - dask_maximum_processes: the maximum number of python processes that dask will be allowed to create when using a local cluster.
        - missing_value_placeholder: placeholder for missing values that will be used to fill the output_data array when it is initialized.
        - bytes_per_datapoint: bytes per value associated with a datapoint.
        - max_data_size: maximum data size allowed to be retrieved, in gigabytes (GB).
        - decimals: number of decimals to round to for precision of xcoor, ycoor, and zcoor in the xarray output.
        - voxel_side_length: smallest sub-box size to recursively shrink to. the value is the length of one side of the cube.
        - chunk_size: cube size of the zarr chunks. the value is the length of one side of the cube.
        - file_size: cube size of an entire file. the value is the length of one side of the cube.
    """
    return {
        'database_file_disk_index':-4,
        'dask_maximum_processes':4,
        'missing_value_placeholder':-999.9,
        'bytes_per_datapoint':4,
        'max_data_size':3.0,
        'decimals':7,
        'voxel_side_length':8,
        'chunk_size':64,
        'file_size':512
    }