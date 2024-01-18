
<p align="center">
  <img src="./img/vasp_logo_temp.png" height="300px">
</p>
<!-- <h1 align="center">
  <br>
  VASP
  <br>
</h1> -->
<h1 align="center"><strong>V</strong>oxel <strong>A</strong>ttributes and <strong>S</strong>tatistics for <strong>P</strong>oint clouds</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>

# Key Features
* Voxelize point clouds
* Compute [additional attributes](#list-of-additional-attributes)
* Compute [statistics for existing attributes](#list-of-statistics-for-existing-attribues)  
* Read from las/laz
* Save to las/laz and ply (as cubes)

### List of additional attributes:
  * point_count,  point_density,  percentage_occupied
  * covariance_matrix,  eigenvalues,  geometric_features
  * big_int_index,hash_index
  * center_of_gravity, center_of_voxel, corner_of_voxel

### List of statistics for existing attribues:
  * Min, Mean, Max
  * Mode, Median, Sum
  
# How To Use
Clone and run this application:
```bash
# Clone this repository
$ git clone https://github.com/3dgeo-heidelberg/vasp.git

# Go into the repository
$ cd vasp

# Install dependencies
$ conda env create -f vasp.yml
```
Exemplary workflows can be found in [jupyter](./jupyter)

# Download
Test data can be downloaded from [!!!ADD LINK](!!!link_here)

# Credits
This software uses the following [python](https://www.python.org/) packages:

- [NumPy](https://numpy.org/)
- [Laspy](https://github.com/laspy/laspy)
- [pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)

# Related
- [3DGeo Research Group, Heidelberg University](https://github.com/3dgeo-heidelberg) - Focused on the development of methods for the geographic analysis of 3D/4D point clouds.

# License
!!!ADD LICENSE

# 
> [Homepage](https://www.geog.uni-heidelberg.de/3dgeo/index.html) &nbsp;&middot;&nbsp; E-Mail [ronald.tabernig@uni-heidelberg.de](ronald.tabernig@uni-heidelberg.de)

