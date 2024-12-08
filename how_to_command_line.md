# Command Line Tool
## Configuration

VAPC uses JSON configuration files to define processing parameters. Below are examples of various configuration templates.

### Compute Attributes

**`compute_attributes_config.json`**
```json
{
     "infile": "./tests/test_data/vapc_in.laz",
     "outdir": "./tests/test_data",
     "voxel_size": 0.2,
     "vapc_command": {
          "tool": "compute",
          "args": {
                "compute": [
                     "big_int_index",
                     "hash_index",
                     "point_density",
                     "point_count",
                     "percentage_occupied",
                     "covariance_matrix",
                     "eigenvalues",
                     "geometric_features",
                     "center_of_gravity",
                     "distance_to_center_of_gravity",
                     "std_of_cog",
                     "closest_to_center_of_gravity",
                     "center_of_voxel",
                     "corner_of_voxel"
                ]
          }
     },
     "tile": false,
     "reduce_to": "center_of_voxel",
     "save_as": ".laz"
}
```

### Filter and Compute Attributes

**`filter_and_compute_config.json`**
```json
{
     "infile": "./tests/test_data/vapc_in.laz",
     "outdir": "./tests/test_data",
     "voxel_size": 0.05,
     "vapc_command": {
          "tool": "filter_and_compute",
          "args": {
                "filters": {
                     "point_count": {
                          "bigger": 5
                     }
                },
                "compute": [
                     "big_int_index",
                     "hash_index",
                     "point_density",
                     "point_count",
                     "percentage_occupied",
                     "covariance_matrix",
                     "eigenvalues",
                     "geometric_features",
                     "center_of_gravity",
                     "distance_to_center_of_gravity",
                     "std_of_cog",
                     "closest_to_center_of_gravity",
                     "center_of_voxel",
                     "corner_of_voxel"
                ]
          }
     },
     "tile": false,
     "reduce_to": "closest_to_center_of_gravity",
     "save_as": ".ply"
}
```

### Mask Configuration

**`mask_config.json`**
```json
{
     "infile": "./tests/test_data/vapc_in.laz",
     "outdir": "./tests/test_data",
     "voxel_size": 0.2,
     "vapc_command": {
          "tool": "mask",
          "args": {
                "maskfile": "./tests/test_data/small_mask.laz",
                "segment_in_or_out": "in",
                "buffer_size": 0
          }
     },
     "tile": 5,
     "reduce_to": false,
     "save_as": ".laz"
}
```

### Subsample Configuration

**`subsample_config.json`**
```json
{
     "infile": "./tests/test_data/vapc_in.laz",
     "outdir": "./tests/test_data",
     "voxel_size": 0.5,
     "vapc_command": {
          "tool": "subsample",
          "args": {
                "sub_sample_method": "closest_to_center_of_gravity"
          }
     },
     "tile": 5,
     "reduce_to": false,
     "save_as": ".laz"
}
```

## Running VAPC

Use the `run.py` script to execute VAPC with your configuration file.

### Command Syntax

```bash
python run.py <config_file.json>
```

### Example

To compute attributes using the `compute_attributes_config.json`:

```bash
python run.py compute_attributes_config.json
```

## Advanced Configurations

### Statistics Computation

**`compute_statistics_config.json`**
```json
{
     "infile": "./tests/test_data/vapc_in.laz",
     "outdir": "./tests/test_data",
     "voxel_size": 1,
     "vapc_command": {
          "tool": "statistics",
          "args": {
                "statistics": {
                     "red": ["mean", "min", "max", "mode", "median", "sum", "mode_count,0.1"],
                     "green": ["mean", "min", "max", "mode", "median", "sum", "mode_count,0.1"],
                     "blue": ["mean", "min", "max", "mode", "median", "sum", "mode_count,0.1"]
                }
          }
     },
     "tile": false,
     "reduce_to": "center_of_voxel",
     "save_as": ".ply"
}
```

Run the statistics computation:

```bash
python run.py compute_statistics_config.json
```

### Filtering by Attributes

**`filter_by_attributes_config.json`**
```json
{
     "infile": "./tests/test_data/vapc_in.laz",
     "outdir": "./tests/test_data",
     "voxel_size": 1,
     "vapc_command": {
          "tool": "filter",
          "args": {
                "filters": {
                     "point_count": {
                          "bigger": 2
                     }
                }
          }
     },
     "tile": false,
     "reduce_to": "center_of_voxel",
     "save_as": ".laz"
}
```

Execute the filter:

```bash
python run.py filter_by_attributes_config.json
```

## Conclusion

By following this guide, you can effectively utilize VAPC as a command line tool to process and analyze your point cloud data. Customize the JSON configuration files to suit your specific processing needs and leverage the various tools provided by VAPC to achieve optimal results.
