<p align="center">Welcome to ...
  <img src="https://github.com/3dgeo-heidelberg/vasp/blob/main/img/vasp_logo_temp.png?raw=true" height="150px">
</p>
<!-- <h1 align="center">
  <br>
  VASP
  <br>
</h1> -->
<h1 align="center"><strong>V</strong>oxel <strong>A</strong>ttributes and <strong>S</strong>tatistics for <strong>P</strong>oint clouds</h4>

`vasp` is a `Python` library for voxel-based point cloud operations.

3D/4D point clouds are used in many fields and applications. Efficient processing of dense time series of point clouds or large study sites requires tools for automatic analysis. Moreover, methods considering the full 4D (3D space + time) data are being developed in research and need to be made available in an accessible way with flexible integration into existing workflows.

The **main objective** of `vasp` is to bundle and provide different methods of 3D/4D point cloud processing using a voxel-based structure in a dedicated, comprehensive Python library.

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>

# Installation
Clone and run this application:
```bash
# Clone this repository
$ git clone https://github.com/3dgeo-heidelberg/vasp.git

# Go into the repository
$ cd vasp

# Installing the release version using pip
$ python -m pip install .

#OR if editable needed
python -m pip install -v --editable .

```
# Documentation of software usage
## Jupyter Notebooks
Examplary [Jupyter Notebooks](./jupyter) are available.

### Some usefull tools provided by `vasp`
* Subsampling of point clouds
* Voxelisation of point clouds
* Computation of voxel based attributes
* Computation of voxel based statistics for existing attributes
* Voxel based 3D masking of point clouds
* Voxel attribute based filtering of point clouds

### Examples of setting up you own pipeline with `vasp`
* First filter by point count and than compute statistics

## Command line `vasp`
Using vasp from the [command line](./cmd) with configure files is already possible and will soon be explained in the [how-to](./cmd/how_to_command_line.md). 

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

