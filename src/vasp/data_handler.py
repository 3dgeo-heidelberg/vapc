#For Data Handler:
from plyfile import PlyData, PlyElement

import laspy
import os
import numpy as np
import pandas as pd
from .utilities import trace,timeit
from scipy import stats

class DATA_HANDLER:
    def __init__(self, infiles):
        """
        Contains the functionality for opening and saving data.
        The tool is currently being optimized to simplify workflows with the voxelizer.

        Parameters:
        - infiles (str or list): A file path or a list of file paths to laz files that will be 
        converted into a single dataframe.
        """
        if isinstance(infiles, list):
            self.files = infiles
        else:
            self.files = [infiles]

    @trace
    @timeit
    def load_las_files(self):
        """
        Opens one or more LAZ or LAS files and reads them into a Pandas DataFrame.
        The data from each file is concatenated into a single DataFrame stored in `self.df`.
        """
        all_data = []
        if type(self.files) != list:
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
                                columns = ["X", "Y", "Z"] + self.attributes)
                all_data.append(df)

        # Merge all data frames and append to existing or create new df
        if hasattr(self, 'df'):
            self.df = pd.concat([self.df] + all_data, ignore_index=True)
        else:
            self.df = pd.concat(all_data, ignore_index=True)


    @trace
    @timeit
    def save_as_las(self,
                    outfile:str,
                    las_point_format = 6,
                    las_version = "1.4",
                    las_scales = [0.00025,0.00025,0.00025],
                    ):
        """
        Function used to save data stored in the dataframe at an given path.

        Parameters:
        - outfile (str): Path where laz file is stored.
        """
        if not hasattr(self, 'las_header'):
            new_header = laspy.LasHeader(point_format = las_point_format,version = las_version)
            new_header.offsets = [self.df["X"].mean(),self.df["Y"].mean(),self.df["Z"].mean()]
            new_header.scales = las_scales
            # raise ValueError("LAS header not found. Ensure a LAS file has been read.")
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


    # @trace
    # @timeit
    # def _generate_mesh_data_old(self):
    #     """
    #     Generate mesh data from voxel information.
    #     Returns a tuple of vertices, faces, and vertex colors.
    #     """
        
    #     df_values = np.c_[self.df['X'].values, self.df['Y'].values, self.df['Z'].values,  self.df.iloc[:,3:].values]

    #     # Calculate voxel corners
    #     corners = self._calculate_voxel_corners(df_values)

    #     # Add faces for one cube (a mesh voxel made of triangles)
    #     face_indices_0 = np.array([
    #         [0, 1, 2],  # Front
    #         [4, 5, 6],  # Back
    #         [0, 1, 5],  # Bottom
    #         [2, 3, 7],  # Top
    #         [1, 2, 6],  # Right
    #         [3, 0, 4],  # Left
    #         [0, 3, 2],  # Front
    #         [4, 7, 6],  # Back
    #         [0, 4, 5],  # Bottom
    #         [2, 6, 7],  # Top
    #         [1, 5, 6],  # Right
    #         [3, 7, 4]   # Left
    #     ])

    #     # Number of corners
    #     corners_shp = corners.shape[0]

    #     # Number of voxels (number of corners divided by 8)
    #     vxl_shp = int(corners_shp/8)
        
    #     # Tile the face indices array so the number of cubes is the same as the number of voxels
    #     face_indices = np.tile(face_indices_0, (vxl_shp,1))

    #     # Making an array to add to the face indices which will then be able to fetch the corresponding corner index for each face
    #     n = np.arange(0, corners_shp, 8)
    #     n = np.repeat(n, face_indices_0.shape[0])
    #     n = np.tile(n, (face_indices_0.shape[1], 1)).T

    #     # Adding up the array
    #     face_indices += n

    #     return corners, face_indices
    

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
    
    
    @trace
    @timeit
    def save_as_ply(self, outfile: str, voxel_size: float, shift_to_center: bool = False):
        """
        Saves the voxel data as "cubes" in a .ply file.

        Parameters:
        - outfile (str): Path where PLY file is stored.
        - voxel_size (float): Edge length of the voxels to be created.
        - shift_to_center (bool): Shift data to origin. Useful for visualizations in Blender to avoid shifting data.
        """
        # Adjust color values if necessary
        if "red" in self.df.columns and "green" in self.df.columns and "blue" in self.df.columns:
            if self.df.red.max() > 255 or self.df.green.max() > 255 or self.df.blue.max() > 255:
                self.df.red = (self.df.red / 65535.0 * 255).astype(np.uint8)
                self.df.green = (self.df.green / 65535.0 * 255).astype(np.uint8)
                self.df.blue = (self.df.blue / 65535.0 * 255).astype(np.uint8)

        self.voxel_size = voxel_size

        # Generate mesh data
        verts, faces = self._generate_mesh_data()

        if shift_to_center:
            min_x = self.df.X.mean() + self.voxel_size / 2
            min_y = self.df.Y.mean() + self.voxel_size / 2
            min_z = self.df.Z.mean() + self.voxel_size / 2
            min_coord = np.array([min_x, min_y, min_z])
            verts[:, :3] -= min_coord

        # Prepare vertex data for PLY file
        # Build the data type list for the structured array
        vertex_dtype = []
        for i, field_name in enumerate(self.df.columns):
            field_name_lower = field_name.lower() if field_name in ['X', 'Y', 'Z'] else field_name
            if field_name in ['red', 'green', 'blue']:
                vertex_dtype.append((field_name, 'u1'))  # colors as uint8
            elif field_name in ['X', 'Y', 'Z']:
                vertex_dtype.append((field_name_lower, 'f4'))  # x, y, z as float32
            else:
                # For other attributes, assume float32
                vertex_dtype.append((field_name, 'f4'))

        # Create structured array for vertices
        vertex_array = np.zeros(len(verts), dtype=vertex_dtype)

        # Assign data to the structured array
        for i, field_name in enumerate(self.df.columns):
            field_name_lower = field_name.lower() if field_name in ['X', 'Y', 'Z'] else field_name
            if field_name in ['red', 'green', 'blue']:
                # Ensure data is uint8
                data = verts[:, i].astype(np.uint8)
            else:
                data = verts[:, i]
            vertex_array[field_name_lower] = data

        # Prepare face data
        face_dtype = [('vertex_indices', 'i4', (3,))]
        face_array = np.empty(len(faces), dtype=face_dtype)
        face_array['vertex_indices'] = faces

        # Create PlyElements
        vertex_element = PlyElement.describe(vertex_array, 'vertex')
        face_element = PlyElement.describe(face_array, 'face')

        # Write to PLY file

        with open(outfile, 'wb') as ply_file:
            PlyData(PlyData([vertex_element, face_element], text=False)).write(ply_file)
        

    def _generate_mesh_data(self):
        """
        Generate mesh data from voxel information.
        Returns a tuple of vertices and faces.
        """
        df_values = np.c_[self.df['X'].values, self.df['Y'].values, self.df['Z'].values, self.df.iloc[:, 3:].values]

        # Calculate voxel corners
        corners = self._calculate_voxel_corners(df_values)

        # Define the face indices for a cube
        face_indices_0 = np.array([
            [0, 1, 2], [0, 2, 3],  # Front face
            [4, 5, 6], [4, 6, 7],  # Back face
            [0, 1, 5], [0, 5, 4],  # Bottom face
            [2, 3, 7], [2, 7, 6],  # Top face
            [1, 2, 6], [1, 6, 5],  # Right face
            [3, 0, 4], [3, 4, 7]   # Left face
        ])

        # Total number of voxels
        num_voxels = int(corners.shape[0] / 8)

        # Repeat face indices for each voxel
        faces = np.tile(face_indices_0, (num_voxels, 1))

        # Offset face indices for each voxel
        offset = np.arange(0, num_voxels * 8, 8).repeat(12)
        faces += offset[:, np.newaxis]

        return corners, faces