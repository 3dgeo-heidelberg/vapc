<p align="center">
  <a href="#structure">Structure</a> •
  <a href="#the-data-handler">Data handler</a> •
  <a href="#vasp">Vasp</a> •
  <a href="#full-example-of-a-config-file">Full config file example</a> •
  <a href="#command-line-workflow-example">Run example</a> •
  <a href="#how-to-filter">Filtering</a> •
  <a href="#how-to-mesh">Meshing</a>
</p>

# Structure
Vasp is structured in two parts. The data handler and the vasp computation part.
## The data handler
is an addition for easy reading and saving of data. It always read a list of point clouds and saves them as a single dataframe. Whenever calling save from the data handler, the dataframe currently stored in it will be saved. So here the workflow would be:
* Initiate data handler
* Load data with data handler
* Give dataframe to vasp
* Return dataframe after computations back to data handler
* Call save from data handler  

Required parameters for the data handling can be set like this:

```json
"infiles":["v2_vasp/test_data/vasp_in.laz"],
"outfile":"v2_vasp/test_data/vasp_out.laz",
"voxel_size":1.0,
"origin":[0,0,0]
```


## vasp...
is used to compute statistics of existing attributes and compute new attribute. This takes place in a voxel space where the voxel size ("voxel_size") and the origin of the voxel space have to be defined ("origin"). 

### For the computation of statistics for existing attributes
the following methods are currently implemented:
* min
* mean
* median
* max
* sum
* modus

Where the respective value within each voxel will be computed for any attribute. If the mean intensity should be calculated, one can require to do so by writing the following into the config file:
```json
"attributes":{
    "intensity":"mean"
    }
```
Similar the mean intensity and the mode for the hitObjectId (e.g. for VLS data) can be computed by this:

```json
"attributes":{
    "intensity":"mean",
    "hitObjectId":"mode"
    }
```
Warning: mode is currently only implemented with a significantly slower method.


### For the computatino of new attributes
the following options exist:

#### Indices:
* big_int_index
* hash_index

#### Point releted:
* point_count
* point_density
* percentage_occupied
* center_of_gravity
* center_of_voxel
* corner_of_voxel


#### Geometry related:
  * covariance_matrix 
  * eigenvalues
  * geometric_features
  
To compute new attributes one could define a list similar to this for the config file:

```json
"calculate":[   
    "point_count",
    "geometric_features",
    "center_of_gravity"
    ]
```
As we can see in this example we want to get the point count, geometric features, and the center of gravity. In order to compute geometric features we require the eigenvalues, which on the other hand require the covariance matrix. By only asking for the geometric features the eigenvalues and covariance matrix will also be calculated but not stored in the result. If one wants the covariance matrix in the output file, it has to be written in the compute list.

## What are the new coordinates?
Here three options are currently implemented:
* Center of gravity     ("center_of_gravity")
* Center of voxel       ("center_of_voxel")
* Corner of voxel       ("corner_of_voxel")

This can be set like this:
```json
"return_at":"center_of_voxel"
```

## Full example of a config file
Lets define a scenario:

We want to use vasp to compute the mean intensity of my input point cloud. The voxel size should be 1 m. Additionally we want to compute the point count and the big int index. The result of each voxel shall be stored in a point cloud where each point is located at the center of gravity of its corresponding voxel. Here the config file would look something like this:

```json
{
"infiles":["v2_vasp/test_data/vasp_in.laz"],
"outfile":"v2_vasp/test_data/vasp_out.laz",
"voxel_size":1.0,
"origin":[0,0,0],

"attributes":{
    "intensity":"mean"
    },
    
"calculate":[   
    "point_count",
    "big_int_index"
    ],
"return_at":"center_of_gravity"
}
```

## Command line workflow example:
* Clone and install the dependencies as described in the [readme](readme.md).
* Navigate into the [demo](./demo) folder and from there ...
* Run main.py and select the config.py file as input.
  
```bash 
cd demo
python main.py config.py
```

## How to filter?
A common use case might be connected to filtering the result by a minimum number of points per voxel. A filter can also be defined in the following way:

```json 
attr_filter = {
    "filter_attribute":"point_count",
    "filter_value":50,
    "min_max_eq":"min"}
```

As we can see we have to define the operator (min, min_eq, max, max_eq,eq), the filter value and the filter attribute.

!!! This method is not yet implemented in an automated workflow but can be called in a script by:

```python
vasp.filter_attributes(
    filter_attribute=attr_filter["filter_attribute"],
    filter_value=attr_filter["filter_value"],
    min_max_eq=attr_filter["min_max_eq"]
    )
```

## How to mesh
Visualisations might require to generate a mesh of a voxelized point cloud. The data handler currently has a method implemented, that stores the point cloud as a voxel mesh in .ply format (readable with Bender). The option shift to center is usefull for loading data into Blender, as it does not shift by default. True coordinates are lost by this.

!!! This method is not yet implemented in an automated workflow but can be called in a script by:

```python
data_handler.save_as_ply(
    outfile=config["outfile_ply"],
    voxel_size=vasp.voxel_size,
    shift_to_center=False
    ) 
```
Where it might make sense to add an additional option in the config file similar to the regular outfile.