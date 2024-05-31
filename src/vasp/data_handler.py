#For Data Handler:
# import OSToolBox as ost #add installer info to yml pip install OSToolBox
import laspy
import os
import numpy as np
import pandas as pd
from utilities import trace,timeit
from scipy import stats

class DATA_HANDLER:
    def __init__(self,
                 infiles:list
                 ):
        """
        Contains the functionality for opening and saving data. 
        The tool is currently being optimized to simplify workflows with the voxelizer.

        Parameters:
        - infiles (list): This list contains one or more laz files that will be converted into a single dataframe.
        - attributes (dict): This dictionary contains information about which attributes to read and 
                                what statistics to carry out on them.
        """
        self.files = infiles

    @trace
    @timeit
    def load_las_files(self):
        """
        This function opens one or more laz files and reads them.  
        The data is stored in a Pandas dataframe.
        """

        all_data = []
        print(self.files)
        if type(self.files) is str:
            self.files = [self.files]

        for filepath in self.files:
            print("Loading ... %s"%os.path.basename(filepath))
            with laspy.open(filepath) as lf:
                las = lf.read()
                self.las_header = las.header
                self.attributes = list(las.points.array.dtype.names)
                list(map(self.attributes.remove,["X","Y","Z"])) #remove XYZ as xyz should be read directly to not scale and shift the points in an extra step.
                df = pd.DataFrame(
                                data = np.array([las.x,las.y,las.z]+[las[attr] for attr in self.attributes]).T,
                                columns = ["X","Y","Z"]+[attr for attr in self.attributes])
                all_data.append(df)
        # Merge all data frames in the list and store in self.df
        self.df = pd.concat(all_data, ignore_index=True)
    
    ### Chunkwise reading works but requires some updates
    # @trace
    # @timeit
    # def load_las_files_chunkwise(self,
    #                                   sub_voxel_size:float,
    #                                   origin:list = [0,0,0],
    #                                   chunk_reader_size:int = 10_000):
    #     """
    #     This function opens one or more laz files and reads them using the chunk iterator.
    #     This approach uses less RAM. A chunk size of approximately 10_000 currently delivers 
    #     the highest performance. The data is stored in a Pandas dataframe.

    #     Parameters:
    #     - sub_voxel_size (float): Size of subvoxels the original data is reduced to.
    #     - origin (list): Origin of the voxel space.
    #     - chunk_reader_size (int): Number of points read per iteration.
    #     """
    #     false_attributes = []
    #     filled_voxel_dict = {}
    #     filled_voxel_attr_dict = {}
    #     self.origin = origin
    #     for filepath in self.files:
    #         with laspy.open(filepath) as lf:
    #             print("Loading ... %s"%os.path.basename(filepath))
    #             self.las_header = lf.header
    #             for sub_las in lf.chunk_iterator(chunk_reader_size):
    #                 x = sub_las.X * self.las_header.scales[0] + self.las_header.offsets[0]
    #                 y = sub_las.Y * self.las_header.scales[1] + self.las_header.offsets[1]
    #                 z = sub_las.Z * self.las_header.scales[2] + self.las_header.offsets[2]
    #                 df_content = {'X': x, 'Y': y, 'Z': z}
    #                 for rel_attr in self.attributes:
    #                     if hasattr(sub_las,rel_attr):
    #                         df_content[rel_attr] = sub_las[rel_attr]
    #                     else:
    #                         print("Could not find %s"%rel_attr)
    #                 x_voxels = np.floor((x - self.origin[0]) / sub_voxel_size).astype(int)
    #                 y_voxels = np.floor((y - self.origin[1]) / sub_voxel_size).astype(int)
    #                 z_voxels = np.floor((z - self.origin[2]) / sub_voxel_size).astype(int)
    #                 xyz_voxel_coords = np.column_stack((x_voxels, y_voxels, z_voxels))
    #                 unique_voxel_coords, indices, counts = np.unique(xyz_voxel_coords, axis=0, return_counts=True, return_inverse=True)

    #                 xyz_coords = np.column_stack((x, y, z))
    #                 for i, voxel_coord in enumerate(unique_voxel_coords):
    #                     voxel_str = str(tuple(voxel_coord))
    #                     mean_coord = np.mean(xyz_coords[indices == i], axis=0)
    #                     attr_data_dic = {}
    #                     for attr in self.attributes:
    #                         attr_data_at_voxel = sub_las[attr][indices == i]
    #                         if self.attributes[attr] == "mode":
    #                             attr_stat = stats.mode(attr_data_at_voxel)[0]
    #                         elif self.attributes[attr] == "sum":
    #                             attr_stat = attr_data_at_voxel.sum()
    #                         elif self.attributes[attr] == "mean":
    #                             attr_stat = attr_data_at_voxel.mean()
    #                         elif self.attributes[attr] == "median":
    #                             attr_stat = np.median(attr_data_at_voxel)
    #                         elif self.attributes[attr] == "min":
    #                             attr_stat = attr_data_at_voxel.min()
    #                         elif self.attributes[attr] == "max":
    #                             attr_stat = attr_data_at_voxel.max()
    #                         else:
    #                             false_attributes.append(attr)
    #                             self.attributes.pop(attr)
    #                             print("Aggregation type unknown for %s"%attr)
    #                             continue
    #                         attr_data_dic[attr] = attr_stat
    #                     if voxel_str in filled_voxel_dict:
    #                         filled_voxel_dict[voxel_str][0].append(mean_coord)
    #                         filled_voxel_dict[voxel_str][1].append(counts[i])
    #                         filled_voxel_dict[voxel_str][2] += counts[i]
    #                         for attr in attr_data_dic.keys():
    #                             filled_voxel_attr_dict[voxel_str][attr].append(attr_data_dic[attr])
    #                     else:
    #                         filled_voxel_dict[voxel_str] = [[mean_coord], [counts[i]],counts[i]]
    #                         filled_voxel_attr_dict[voxel_str] = {}
    #                         for attr in attr_data_dic.keys():
    #                             filled_voxel_attr_dict[voxel_str][attr] = [attr_data_dic[attr]]
    #         out_dic_pts = {}
    #         out_dic_nr_of_pts = {}
    #         out_dic_attributes = {}
    #         for attr in self.attributes.keys():
    #             out_dic_attributes[attr] = {}

    #         for key in filled_voxel_dict.keys():
    #             out_dic_pts[key] = np.sum(np.array(filled_voxel_dict[key][0]).T/ filled_voxel_dict[key][2] * np.array(filled_voxel_dict[key][1]),axis = 1)
    #             out_dic_nr_of_pts[key] = filled_voxel_dict[key][2]
    #             for attr in attr_data_dic.keys():
    #                 attr_data_at_key = filled_voxel_attr_dict[key][attr]
    #                 if self.attributes[attr] == "mode":
    #                     out_dic_attributes[attr][key] = stats.mode(attr_data_at_key)[0]
    #                 elif self.attributes[attr] == "sum":
    #                     out_dic_attributes[attr][key] = np.sum(attr_data_at_key)
    #                 elif self.attributes[attr] == "mean":
    #                     out_dic_attributes[attr][key] = np.mean(attr_data_at_key)
    #                 elif self.attributes[attr] == "median":
    #                     out_dic_attributes[attr][key] = np.median(attr_data_at_key)
    #                 elif self.attributes[attr] == "min":
    #                     out_dic_attributes[attr][key] = np.min(attr_data_at_key)
    #                 elif self.attributes[attr] == "max":
    #                     out_dic_attributes[attr][key] = np.max(attr_data_at_key)
    #                 else:
    #                     print("Aggregation type unknown for %s"%attr)
    #                     continue

    #         pts = np.array([*out_dic_pts.values()])
    #         nr_of_pts = np.array([*out_dic_nr_of_pts.values()])
    #         # ints = np.array([*out_dic_ints.values()])
    #         # rgbs = np.array([*out_dic_rgbs.values()],dtype = int)
    #         combined_data = {'X': pts[:,0], 'Y': pts[:,1], 'Z': pts[:,2],"point_count_subvoxel":nr_of_pts}
    #         for attr in out_dic_attributes.keys():
    #             combined_data.update({attr:np.array([*out_dic_attributes[attr].values()])})
    #         self.df = pd.DataFrame(combined_data)
                
                
                
    #         #     ,"intensity":ints,"red":rgbs[:,0],"green":rgbs[:,1],"blue":rgbs[:,2]})
    #         # elif self.intensity:
    #         #     self.df = pd.DataFrame({'X': pts[:,0], 'Y': pts[:,1], 'Z': pts[:,2],"point_count_subvoxel":nr_of_pts,"intensity":ints})
    #         # elif self.colorized:
    #         #     self.df = pd.DataFrame({'X': pts[:,0], 'Y': pts[:,1], 'Z': pts[:,2],"point_count_subvoxel":nr_of_pts,"red":rgbs[:,0],"green":rgbs[:,1],"blue":rgbs[:,2]})
    #         # else:
    #         #     self.df = pd.DataFrame({'X': pts[:,0], 'Y': pts[:,1], 'Z': pts[:,2],"point_count_subvoxel":nr_of_pts})

    @trace
    @timeit
    def save_as_las(self,
                    outfile:str):
        """
        Function used to save data stored in the dataframe at an given path.

        Parameters:
        - outfile (str): Path where laz file is stored.
        """
        if not hasattr(self, 'las_header'):
            new_header = laspy.LasHeader(point_format = 6,version = "1.4")
            new_header.offsets = [self.df["X"].mean(),self.df["Y"].mean(),self.df["Z"].mean()]
            new_header.scales = [0.00025,0.00025,0.00025]
            # raise ValueError("LAS header not found. Ensure a LAS file has been read.")
        else:
            new_header = laspy.LasHeader(point_format=self.las_header.point_format, version=self.las_header.version)
            new_header.offsets = self.las_header.offsets
            new_header.scales = self.las_header.scales 
        self.lasFile = laspy.LasData(new_header)
        self.lasFile.x = self.df["X"]
        self.lasFile.y = self.df["Y"]
        self.lasFile.z = self.df["Z"]
        #for VLS data...
        try:
            self.lasFile.remove_extra_dims(["ExtraBytes"])
        except:
            pass
        # Add other attributes to output:
        for name in self.df.columns:
            if name not in ["X", "Y", "Z"]:
                try:
                    self.lasFile[name] = self.df[name]
                except Exception as e:
                    self._addDimensionToLaz(self.df[name].astype(np.float32),name)
                    print("Adding new dimension %s"%name)
        self.lasFile.write(outfile)

    @trace
    @timeit
    def save_as_ply(self, 
                        outfile:str, 
                        voxel_size:float, 
                        shift_to_center:bool = False):
        """
        Saves the voxeldata as "cubes" in a .ply file. 
        !!!Fixed shift as parameter might be interesting.

        Parameters:
        - outfile (str): Path where laz file is stored.
        - voxel_size (float): Edgelength of the voxels will be created with.
        - shift_to_center (bool): Shift data to origin. Usefull for visualisations in Blender to avoid shifting data.
        """
        

        self.voxel_size = voxel_size
        self.scalars = self.df.columns
        self.scalars = self.scalars.drop(["X","Y","Z"])
        try:
            self.scalars = self.scalars.drop(["red","green","blue"])
        except:
            pass

        #verts, faces, vert_colors, attributes = self._generate_mesh_data()
        verts, faces = self._generate_mesh_data()
        if shift_to_center:
            min_x = self.df.X.mean() + self.voxel_size / 2
            min_y = self.df.Y.mean() + self.voxel_size / 2
            min_z = self.df.Z.mean() + self.voxel_size / 2
            min = np.array([min_x, min_y, min_z])
            verts[:,:3] -= np.tile(min, (verts.shape[0], 1))


        ost.write_ply(filename=outfile, field_list=verts.astype(float), field_names=self.df.columns, triangular_faces=faces)
        
        
    def _addDimensionToLaz(self,
                           array,
                           name):
        """
        Write new attribute to laz file.
        """
        self.lasFile.add_extra_dim(laspy.ExtraBytesParams(
        name=name,
        type=array.dtype,
        description=name
        ))
        self.lasFile[name] = array


    @trace
    @timeit
    def _generate_mesh_data(self):
        """
        Generate mesh data from voxel information.
        Returns a tuple of vertices, faces, and vertex colors.
        """
        
        df_values = np.c_[self.df['X'].values, self.df['Y'].values, self.df['Z'].values,  self.df.iloc[:,3:].values]

        # Calculate voxel corners
        corners = self._calculate_voxel_corners(df_values)

        # Add faces for one cube (a mesh voxel made of triangles)
        face_indices_0 = np.array([
            [0, 1, 2],  # Front
            [4, 5, 6],  # Back
            [0, 1, 5],  # Bottom
            [2, 3, 7],  # Top
            [1, 2, 6],  # Right
            [3, 0, 4],  # Left
            [0, 3, 2],  # Front
            [4, 7, 6],  # Back
            [0, 4, 5],  # Bottom
            [2, 6, 7],  # Top
            [1, 5, 6],  # Right
            [3, 7, 4]   # Left
        ])

        # Number of corners
        corners_shp = corners.shape[0]

        # Number of voxels (number of corners divided by 8)
        vxl_shp = int(corners_shp/8)
        
        # Tile the face indices array so the number of cubes is the same as the number of voxels
        face_indices = np.tile(face_indices_0, (vxl_shp,1))

        # Making an array to add to the face indices which will then be able to fetch the corresponding corner index for each face
        n = np.arange(0, corners_shp, 8)
        n = np.repeat(n, face_indices_0.shape[0])
        n = np.tile(n, (face_indices_0.shape[1], 1)).T

        # Adding up the array
        face_indices += n

        return corners, face_indices
    

    def _calculate_voxel_corners(self, df_values):
        """
        Compute corners points for all the cubes.
        Return corners points with their respective scalars.
        """
        offset = self.voxel_size/2.
        xyz_offsets = np.array([
            [-offset, -offset, -offset],
            [offset, -offset, -offset],
            [offset, offset, -offset],
            [-offset, offset, -offset],
            [-offset, -offset, offset],
            [offset, -offset, offset],
            [offset, offset, offset],
            [-offset, offset, offset]
        ])

        # Fetch scalars array and repeat each line the number of lines in xyz_offsets
        scalars = df_values[:,3:]
        scalars = np.repeat(scalars, xyz_offsets.shape[0], axis=0)

        # Add offsets for each points
        xyz = df_values[:,:3][:, np.newaxis] + xyz_offsets
        xyz = np.vstack(xyz)

        # Concatenate the points with their respective scalars
        xyz_scalars = np.c_[xyz, scalars]

        return xyz_scalars
    
    
