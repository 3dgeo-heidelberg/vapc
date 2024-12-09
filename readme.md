<!-- <p align="center">Welcome to ...
  <img src="https://github.com/3dgeo-heidelberg/vapc/blob/main/img/vapc_logo_temp.png?raw=true" height="150px">
</p>
<!-- <h1 align="center">
  <br>
  vapc
  <br>
</h1> -->
<h1 align="center"><strong>V</strong>oxel <strong>A</strong>nalysis for <strong>P</strong>oint <strong>C</strong>louds</h4>

`vapc` is a `Python` library for voxel-based point cloud operations.

3D/4D point clouds are used in many fields and applications. Efficient processing of dense time series of point clouds or large study sites requires tools for automatic analysis. Moreover, methods considering the full 4D (3D space + time) data are being developed in research and need to be made available in an accessible way with flexible integration into existing workflows.

The **main objective** of `vapc` is to bundle and provide different methods of 3D/4D point cloud processing using a voxel-based structure in a dedicated, comprehensive Python library.

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#documentation-of-software-usage">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#related">Related</a>
</p>

# Installation

## Creating Conda Environments
To avoid negative interactions between installed packages and version conflicts, you should create a conda environment for each new project. You do so by executing:
```bash
# First, create new environment
$ conda create --name vapc python=3.10

# Then activate the environment using:
$ conda activate vapc

```

Using vapc requires Python 3.10 or higher.
Clone and run this application:

```bash

# Clone this repository
$ git clone https://github.com/3dgeo-heidelberg/vapc.git

# Go into the repository
$ cd vapc

# Installing the release version using pip
$ python -m pip install .

#OR if editable needed
$ python -m pip install -v --editable .

```

# Documentation of software usage
## Jupyter Notebooks
Examplary [Jupyter Notebooks](./jupyter) are available.

### Some useful tools provided by `vapc`
* Subsampling of point clouds
* Voxelisation of point clouds
* Computation of voxel based attributes
* Computation of voxel based statistics for existing attributes
* Voxel based 3D masking of point clouds
* Voxel attribute based filtering of point clouds

## Command line `vapc`
Using vapc from the command line with config files is explained in the [how to command line](how_to_command_line.md). 

# Download
A small test dataset is currently provided in [test data](./test_data).

# Related
[3DGeo Research Group, Heidelberg University](https://github.com/3dgeo-heidelberg) - Focused on the development of methods for the geographic analysis of 3D/4D point clouds.

# License
See [LICENSE](LICENSE).

# 
> [Homepage](https://www.geog.uni-heidelberg.de/3dgeo/index.html) &nbsp;&middot;&nbsp; E-Mail [ronald.tabernig@uni-heidelberg.de](ronald.tabernig@uni-heidelberg.de)

