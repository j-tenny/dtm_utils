# dtm_utils
Iterative height filter algorithm and other utilities for working with digital terrain models from lidar

## Installation
```
conda create -n dem_utils pdal rasterio polars numpy scipy
conda activate dem_utils
pip install git+https://github.com/j-tenny/dtm_utils
```

## Usage

Create a digital elevation model from a point cloud using iterative height filtering algorithm

Written by Johnathan Tenny (jt893@nau.edu) based on Caster et al 2021 https://doi.org/10.1016/j.geoderma.2021.115369

Algorithm Description: A ground surface is estimated using the minimum elevation within 2D grid cells with gaps filled using 2D linear
interpolation. From this surface, height-above-ground is calculated for all lidar points. Points above a
height-above-ground threshold are filtered out. This process is repeated using progressively smaller window sizes
and height thresholds.

Args:

- `points`: a numpy array or polars dataframe where the first three columns are x, y, z coordinates of a point cloud
    
- `output_path`: filepath to write output raster
    
- `output_resolution`: resolution of the output DEM
    
- `origin`: origin of the output DEM
    
- `window_sizes`: list of progressively smaller xy window resolutions
    
- `height_thresholds`: list of progressively smaller height thresholds
    
- `ground_quantile`: statistical quantile of elevation within bin to use as ground elevation.
Set to 0. to use minimum elevation or adjust if ground elevations seem biased.

```
input_filepath = 'test_data/als_creek.las'
output_path = 'test_data/als_creek_dtm.tif'
origin = (0,0)
window_sizes      = [5, 2.5, 1.5, 1., .5]
height_thresholds = [4, 2.0, 1.2, .7, .3]
ground_quantile = 0

import dtm_utils
import os
if os.path.exists(output_path):
    os.remove(output_path)
points, crs = dtm_utils.read_pdal(input_filepath)
dtm_utils.create_dtm_iterative_height_filter(points,crs,output_path,origin,
                                             window_sizes,height_thresholds,ground_quantile)
```
