def create_dtm_iterative_height_filter(points,
                                       crs,
                                       output_path,
                                       origin = [0,0],
                                       window_sizes=[5, 2.5, 1, .5],
                                       height_thresholds=[2.5, 1.25, .5, .25],
                                       ground_quantile = 0.):
    """Create a digital elevation model from a point cloud using iterative height filtering

    A ground surface is estimated using the minimum elevation within 2D grid cells with gaps filled using 2D linear
    interpolation. From this surface, height-above-ground is calculated for all lidar points. Points above a
    height-above-ground threshold are filtered out. This process is repeated using progressively smaller window sizes
    and height thresholds.

    Written by Johnathan Tenny (jt893@nau.edu) based on Caster et al 2021 https://doi.org/10.1016/j.geoderma.2021.115369

    Args:
        points: a numpy array or polars dataframe where the first three columns are x, y, z coordinates of a point cloud
        crs: coordinate system of point cloud
        output_path: filepath to write output raster
        origin: origin of the output DEM
        window_sizes: list of progressively smaller xy window resolutions
        height_thresholds: list of progressively smaller height thresholds
        ground_quantile: statistical quantile of elevation within bin to use as ground elevation.
            Set to 0. to use minimum elevation or adjust if ground elevations seem biased.
    """

    import polars as pl
    import numpy as np
    import rasterio

    # Format point cloud as polars dataframe
    try:
        points = points.xyz
    except:
        points = points[:, 0:3]

    points_df = pl.DataFrame({'X': points[:, 0], 'Y': points[:, 1], 'Z': points[:, 2]})

    # Iterative height filtering
    for window_size, height_thresh in zip(window_sizes, height_thresholds):
        # Get ground surface based on low point in window
        bin_df = bin2D(points_df, pl.quantile('Z',ground_quantile), window_size, origin)
        bin_df = bin_df.rename({'Z':'Ground'})

        # Interpolate missing values
        mask_valid = bin_df['Ground'].is_finite().is_not_null()
        if mask_valid.sum() != len(bin_df):
            points_valid = bin_df.filter(mask_valid).select(['YBin','XBin'])
            values_valid = bin_df.filter(mask_valid)['Ground']
            points_missing = bin_df.filter(~mask_valid).select(['YBin','XBin'])
            values_missing = interp2D_w_nearest_neighbor_extrapolation(points_valid, values_valid, points_missing)
            new_vals = bin_df['Ground'].to_numpy().copy()
            new_vals[~mask_valid] = values_missing
            bin_df = bin_df.with_columns(pl.lit(new_vals).alias('Ground'))

        # Get height-above-ground
        points_df = points_df.with_columns(pl.col('Y').sub(origin[1]).floordiv(window_size).cast(pl.Int32).alias('YBin'),
                                            pl.col('X').sub(origin[0]).floordiv(window_size).cast(pl.Int32).alias('XBin'))

        points_df = points_df.drop('Ground',strict=False).join(bin_df, ['YBin', 'XBin'], 'left')

        # Remove points above the height threshold
        points_df = points_df.filter(pl.col('Z') <= pl.col('Ground').add(height_thresh))

    # Convert to raster
    nx = bin_df['XBin'].n_unique()
    ny = bin_df['YBin'].n_unique()
    grid = bin_df.sort(['YBin','XBin'],descending=[True,False])['Ground'].to_numpy().reshape([ny,nx])

    # Get coordinates of upper left corner
    ul_x = bin_df['XBin'].min() * window_size + origin[0]
    ul_y = bin_df['YBin'].max() * window_size + window_size + origin[1]

    transform = rasterio.transform.from_origin(ul_x, ul_y, window_size, window_size)

    metadata = {
        'driver': 'GTiff',
        'dtype': rasterio.float32,
        'nodata': None,
        'width': nx,
        'height': ny,
        'count': 1,  # Number of bands
        'crs': crs,  # Coordinate reference system
        'transform': transform,
    }

    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(grid.astype(rasterio.float32), 1)  # Write to band 1

    
def bin2D(points_df, function, cell_size, origin=(0, 0)) -> 'polars.DataFrame':
    """Aggregate point cloud to a 2D grid and apply a polars function

    points_df: a polars dataframe with columns 'X', 'Y', 'Z'
    function: a function compatible with polars.DataFrame.aggregate(); may need to specify col name e.g. pl.min('z')
    cell_size: float value for output raster resolution
    origin: origin of grid relative to coordinates
    """
    # Function should be from polars and

    import polars as pl
    import numpy as np

    # Get function value for each bin
    points_df = points_df.with_columns(pl.col('X').sub(origin[0]).floordiv(cell_size).cast(pl.Int32).alias('XBin'),
                                       pl.col('Y').sub(origin[1]).floordiv(cell_size).cast(pl.Int32).alias('YBin'))

    bin_vals = points_df.group_by(['XBin', 'YBin']).agg(function)

    # Get df containing all possible bins
    binminx = points_df['XBin'].min()
    binminy = points_df['YBin'].min()
    binmaxx = points_df['XBin'].max()
    binmaxy = points_df['YBin'].max()

    ybins,xbins = np.meshgrid(np.arange(binminy,binmaxy+1),
                              np.arange(binminx,binmaxx+1),
                              indexing='ij')

    bins_df = pl.DataFrame({'YBin': ybins.flatten().astype(np.int32),
                            'XBin': xbins.flatten().astype(np.int32)})

    return bins_df.join(bin_vals,['YBin','XBin'],'left')


def interp2D_w_nearest_neighbor_extrapolation(xy_train, values_train, xy_predict):
    import scipy
    import numpy as np

    xy_train = np.array(xy_train)
    values_train = np.array(values_train).flatten()
    xy_predict = np.array(xy_predict)

    f = scipy.interpolate.LinearNDInterpolator(xy_train, values_train)
    # evaluate the original interpolator. Out-of-bounds values are nan.
    values_predict = f(xy_predict)
    nans = np.isnan(values_predict)
    if nans.any():
        # Build a KD-tree for efficient nearest neighbor search.
        tree = scipy.spatial.cKDTree(xy_train)
        # Find the nearest neighbors for the NaN points.
        distances, indices = tree.query(xy_predict[nans], k=1)
        # Replace NaN values with the values from the nearest neighbors.
        values_predict[nans] = values_train[indices]
    return values_predict

def read_pdal(path):
    import pdal
    import polars as pl
    pipeline = pdal.Pipeline([pdal.Reader(path)])
    pipeline.execute()
    return pl.DataFrame(pipeline.arrays[0]), pipeline.srswkt2