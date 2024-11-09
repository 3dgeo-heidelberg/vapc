<p align="center">Welcome to ...
  <img src="./img/vasp_logo_temp.png" height="150px">
</p>
<!-- <h1 align="center">
  <br>
  VASP
  <br>
</h1> -->
<h1 align="center"><strong>V</strong>oxel <strong>A</strong>ttributes and <strong>S</strong>tatistics for <strong>P</strong>oint clouds</h4>

`vasp` is a `Python` library for voxel-based point cloud operations.

3D/4D point clouds are used in many fields and applications. Efficient processing of dense time series of point clouds or large study sites are require tools for automatic analysis. Moreover, methods considering the full 4D (3D space + time) data are being developed in research and need to be made available in an accessible way with flexible integration into existent workflows.

The **main objective** of `vasp` is to bundle and provide different methods of 3D/4D point cloud processing using a voxel-based structure in a dedicated, comprehensive Python library.

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>

# Tools provided by vasp
* Voxelisation of point clouds
* Computation of [voxel based attributes](#list-of-additional-attributes)
* Computation of [voxel based statistics](#list-of-statistics-for-existing-attribues) for existing attributes
* Voxel based [3D masking of point clouds](#temp)


### List of additional attributes:
  * point_count,  point_density,  percentage_occupied
  * covariance_matrix,  eigenvalues,  geometric_features
  * big_int_index,hash_index
  * center_of_gravity, center_of_voxel, corner_of_voxel

### List of statistics for existing attribues:
  * Min, Mean, Max
  * Mode, Median, Sum
  
# Installation
Clone and run this application:
```bash
# Clone this repository
$ git clone https://github.com/3dgeo-heidelberg/vasp.git

# Go into the repository
$ cd vasp

# Installing the release version using pip
$ python -m pip install .
```

# Documentation of software usage<br>
Examplary [Jupyter Notebooks](./jupyter) are available. <br>
Using vasp from the [command line](./cmd) with configure files is explained in the [how-to](./cmd/how_to_command_line.md). <br>

# Download
A small test dataset is currently provided in [test_data](./test_data).

# Credits
This software uses the following [python](https://www.python.org/) packages:

- [NumPy](https://numpy.org/)
- [Laspy](https://github.com/laspy/laspy)
- [pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [plyfile](https://python-plyfile.readthedocs.io/)

# Related
- [3DGeo Research Group, Heidelberg University](https://github.com/3dgeo-heidelberg) - Focused on the development of methods for the geographic analysis of 3D/4D point clouds.

# License
!!!ADD LICENSE

# 
> [Homepage](https://www.geog.uni-heidelberg.de/3dgeo/index.html) &nbsp;&middot;&nbsp; E-Mail [ronald.tabernig@uni-heidelberg.de](ronald.tabernig@uni-heidelberg.de)

