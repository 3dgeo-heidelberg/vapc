import os
from pathlib import Path
import warnings
from plyfile import PlyData, PlyElement
import laspy
from laspy.errors import LaspyException
import numpy as np
import pandas as pd
from .utilities import trace, timeit


class DataHandler:
    """
    Handles loading, processing, and saving of LAS/LAZ point cloud data files.

    The `DataHandler` class provides functionality to load multiple LAS/LAZ files,
    convert them into a unified pandas DataFrame, and save the processed data in various
    formats such as LAS, LAZ, and PLY. It is optimized to simplify workflows with the voxelizer
    by managing data efficiently.

    Parameters
    ----------
    infiles : str or list of str
        A file path or a list of file paths to LAS/LAZ files that will be converted into a single DataFrame.

    Attributes
    ----------
    files : list of str
        List of input LAS/LAZ file paths.
    df : pandas.DataFrame
        DataFrame containing the loaded and processed point cloud data.
    las_file : laspy.LasData
        LasData object containing the LAS file data.
    las_header : laspy.LasHeader
        LasHeader object containing the LAS file header information.
    attributes : list of str
        List of attributes present in the LAS file.
    voxel_size : float
        Edge length of the voxels to be created if saving mesh to ply.
    """
    def __init__(self, infiles):
        if isinstance(infiles, list):
            self.files = infiles
        else:
            self.files = [infiles]
        self.df = None
        self.las_file = None
        self.las_header = None
        self.attributes = None
        self.voxel_size = None

    @trace
    @timeit
    def load_las_files(self):
        """
        Loads LAS/LAZ files into a pandas DataFrame.

        Opens one or more LAZ or LAS files and reads them into a Pandas DataFrame.
        The data from each file is concatenated into a single DataFrame stored in `self.df`.

        Returns
        -------
        None
        """
        all_data = []
        if not isinstance(self.files, list):
            self.files = [self.files]

        for filepath in self.files:
            # print("Loading ... %s"%os.path.basename(filepath))
            with laspy.open(filepath) as lf:
                las = lf.read()
                self.las_header = las.header
                self.attributes = list(las.point_format.dimension_names)
                list(
                    map(self.attributes.remove, ["X", "Y", "Z"])
                )  # remove XYZ as xyz should be read directly to not scale and shift the points in an extra step.
                # using vls data one has to remove the ExtraBytes dim
                # if "HELIOS++" in self.las_header.generating_software:
                #     self.attributes.remove("ExtraBytes")
                if "ExtraBytes" in self.attributes:
                    self.attributes.remove("ExtraBytes")
                df = pd.DataFrame(
                    data=np.array(
                        [las.x, las.y, las.z] + [las[attr] for attr in self.attributes]
                    ).T,
                    columns=["X", "Y", "Z"] + self.attributes,
                )
                all_data.append(df)

        # Merge all data frames and append to existing or create new df
        if self.df is not None:
            self.df = pd.concat([self.df] + all_data, ignore_index=True)
        else:
            self.df = pd.concat(all_data, ignore_index=True)

    @trace
    @timeit
    def save_as_las(
        self,
        outfile: str,
        las_point_format=7,
        las_version="1.4",
        las_scales=None,
        las_offset=None,
    ):
        """
        Saves the data stored in the DataFrame to a LAS file at the specified path.

        Parameters
        ----------
        outfile : str
            Path where the LAS or LAZ file will be stored.
        las_point_format : int, optional
            Point format for the LAS file (default is 7).
        las_version : str, optional
            LAS file version (default is "1.4").
        las_scales : list of float, optional
            Scale factors for X, Y, Z (default is [0.00025, 0.00025, 0.00025]).
        las_offset : list of float, optional
            Offset values for X, Y, Z (default is [X.min(), Y.min(), Z.min()]).

        Raises
        ------
        AttributeError
            If the DataFrame `self.df` does not exist.

        Returns
        -------
        None
        """
        if self.df is None:
            raise AttributeError("DataFrame not found. Please load data before saving.")
        if las_scales is None:
            las_scales = [0.00025, 0.00025, 0.00025]
        if las_offset is None:
            las_offset = np.min(self.df[["X", "Y", "Z"]].values, axis=0)

        new_header = laspy.LasHeader(point_format=las_point_format, version=las_version)
        new_header.offsets = las_offset
        new_header.scales = las_scales

        self.las_file = laspy.LasData(new_header)
        self.las_file.x = self.df["X"]
        self.las_file.y = self.df["Y"]
        self.las_file.z = self.df["Z"]
        # for VLS data
        if "HELIOS++" in self.las_file.header.generating_software:
            try:
                self.las_file.remove_extra_dims(["ExtraBytes"])
            except LaspyException:
                pass  # 'ExtraBytes' dimension does not exist
        # Add other attributes to output:
        for name in self.df.columns:
            if name not in ["X", "Y", "Z", "scanner_channel", "overlap"]: # Not adding scanner_channel and overlap to laz files
                try:
                    self.las_file[name] = self.df[name].astype(np.float32)
                except TypeError:
                    self.las_file[name] = self.df[name].astype(np.int16)
                except ValueError:
                    self._add_dimension_to_laz(self.df[name].astype(np.float32), name)
        if not os.path.exists(Path(outfile).parent):
            os.makedirs(Path(outfile).parent)

        self.las_file.write(outfile)

    def _add_dimension_to_laz(self, array, name):
        """
        Adds a new attribute dimension to the LAZ file.

        Writes a new attribute to the LAZ file by adding an extra dimension with the specified name and data type.

        Parameters
        ----------
        array : np.ndarray
            Array of values to be added as the new dimension.
        name : str
            Name of the new dimension.

        Returns
        -------
        None
        """
        self.las_file.add_extra_dim(
            laspy.ExtraBytesParams(name=name, type=array.dtype, description=name)
        )
        self.las_file[name] = array

    def _calculate_voxel_corners(self, df_values):
        """
        Computes corner points for all voxels.

        Calculates the corner points for each voxel based on the voxel size and the provided DataFrame values.
        Returns corner points along with their respective scalar attributes.

        Parameters
        ----------
        df_values : np.ndarray
            An array where the first three columns are X, Y, Z coordinates, and the remaining columns are scalar attributes.

        Returns
        -------
        np.ndarray
            Array of voxel corner positions and their associated scalar attributes.
        """
        offset = self.voxel_size / 2.0
        xyz_offsets = np.array(
            [
                [-offset, -offset, -offset],
                [offset, -offset, -offset],
                [offset, offset, -offset],
                [-offset, offset, -offset],
                [-offset, -offset, offset],
                [offset, -offset, offset],
                [offset, offset, offset],
                [-offset, offset, offset],
            ]
        )

        # Fetch scalars array and repeat each line the number of lines in xyz_offsets
        scalars = df_values[:, 3:]
        scalars = np.repeat(scalars, xyz_offsets.shape[0], axis=0)

        # Add offsets for each points
        xyz = df_values[:, :3][:, np.newaxis] + xyz_offsets
        xyz = np.vstack(xyz)

        # Concatenate the points with their respective scalars
        xyz_scalars = np.c_[xyz, scalars]

        return xyz_scalars

    @trace
    @timeit
    def save_as_ply(
        self, outfile: str, voxel_size: float, shift_to_center: bool = False
    ):
        """
        Saves the voxel data as cubes in a PLY file.

        Converts the voxelized DataFrame into a mesh representation and writes it to a PLY file.
        Optionally shifts the data to bbox center for better visualization.

        Parameters
        ----------
        outfile : str
            Path where the PLY file will be stored.
        voxel_size : float
            Edge length of the voxels to be created.
        shift_to_center : bool, optional
            Shift data to bbox center. Useful for visualizations to center the object. Defaults to False.

        Returns
        -------
        None
        """
        assert voxel_size > 0, "Voxel size must be greater than 0"
        if self.df is None:
            raise AttributeError("DataFrame not found. Please load data before saving.")
        self.voxel_size = voxel_size
        self._validate_is_voxelized()

        for enum, scalar in enumerate(self.df.columns.values):
            self.df.columns.values[enum] = scalar.replace(" ", "_")

        # Adjust color values if necessary
        if (
            "red" in self.df.columns
            and "green" in self.df.columns
            and "blue" in self.df.columns
        ):
            if (
                self.df.red.max() > 255
                or self.df.green.max() > 255
                or self.df.blue.max() > 255
            ):
                self.df.red = (self.df.red / 65535.0 * 255).astype(np.uint8)
                self.df.green = (self.df.green / 65535.0 * 255).astype(np.uint8)
                self.df.blue = (self.df.blue / 65535.0 * 255).astype(np.uint8)

        # Generate mesh data
        verts, faces = self._generate_mesh_data()

        if shift_to_center:
            center_x = self.df.X.min() + (self.df.X.max() - self.df.X.min()) / 2
            center_y = self.df.Y.min() + (self.df.Y.max() - self.df.Y.min()) / 2
            center_z = self.df.Z.min() + (self.df.Z.max() - self.df.Z.min()) / 2
            bbox_center = np.array([center_x, center_y, center_z])
            verts[:, :3] -= bbox_center

        # Prepare vertex data for PLY file
        # Build the data type list for the structured array
        vertex_dtype = []
        for i, field_name in enumerate(self.df.columns):
            field_name_lower = (
                field_name.lower() if field_name in ["X", "Y", "Z"] else field_name
            )
            if field_name in ["red", "green", "blue"]:
                vertex_dtype.append((field_name, "u1"))  # colors as uint8
            elif field_name in ["X", "Y", "Z"]:
                vertex_dtype.append((field_name_lower, "f4"))  # x, y, z as float32
            else:
                # For other attributes, assume float32
                vertex_dtype.append((field_name, "f4"))

        # Create structured array for vertices
        vertex_array = np.zeros(len(verts), dtype=vertex_dtype)

        # Assign data to the structured array
        for i, field_name in enumerate(self.df.columns):
            field_name_lower = (
                field_name.lower() if field_name in ["X", "Y", "Z"] else field_name
            )
            if field_name in ["red", "green", "blue"]:
                # Ensure data is uint8
                data = verts[:, i].astype(np.uint8)
            else:
                data = verts[:, i]
            vertex_array[field_name_lower] = data

        # Prepare face data
        face_dtype = [("vertex_indices", "i4", (3,))]
        face_array = np.empty(len(faces), dtype=face_dtype)
        face_array["vertex_indices"] = faces

        # Create PlyElements
        vertex_element = PlyElement.describe(vertex_array, "vertex")
        face_element = PlyElement.describe(face_array, "face")

        # Write to PLY file
        with open(outfile, "wb") as ply_file:
            PlyData([vertex_element, face_element], text=False).write(ply_file)

    def _generate_mesh_data(self):
        """
        Generates mesh data from voxel information.

        Creates vertex positions and face indices based on the voxel data stored in the DataFrame.
        This mesh data is used for exporting to 3D file formats like PLY.

        Returns
        -------
        tuple
            A tuple containing:
                - np.ndarray: Array of vertex positions and scalar attributes.
                - np.ndarray: Array of face indices.
        """
        df_values = self.df[["X", "Y", "Z"] + list(self.df.columns[3:])].values

        # Calculate voxel corners
        corners = self._calculate_voxel_corners(df_values)

        # Define the face indices for a cube
        face_indices_0 = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Front face
                [4, 5, 6],
                [4, 6, 7],  # Back face
                [0, 1, 5],
                [0, 5, 4],  # Bottom face
                [2, 3, 7],
                [2, 7, 6],  # Top face
                [1, 2, 6],
                [1, 6, 5],  # Right face
                [3, 0, 4],
                [3, 4, 7],  # Left face
            ]
        )

        # Total number of voxels
        num_voxels = int(corners.shape[0] / 8)

        # Repeat face indices for each voxel
        faces = np.tile(face_indices_0, (num_voxels, 1))

        # Offset face indices for each voxel
        offset = np.arange(0, num_voxels * 8, 8).repeat(12)
        faces += offset[:, np.newaxis]

        return corners, faces

    def _validate_is_voxelized(self):
        """
        Validates if the data is voxelized properly.

        If the coordinates do not correspond to voxel centers, a warning is issued.

        Returns
        -------
        None
        """
        if self.df is None:
            raise AttributeError("DataFrame not found. Please load data before saving.")
        if self.voxel_size is None:
            raise AttributeError("Voxel size not provided. Cannot validate if voxelized.")
        # do "X", "Y" and "Z" correspond to voxel centers for the given voxel_size?
        # We check if coords mod voxel size is the same for all points
        if not len(np.unique(self.df[["X", "Y", "Z"]].values % self.voxel_size)) == 1:
            warnings.warn(
                "Caution: Data is not voxelized properly. Output may contain overlapping voxels. \n \
                Please voxelize data using 'reduce_at'=='center_of_voxel' before saving as PLY",
                UserWarning
            )

    ###### Mesh functions ######
    def open_obj_mesh(self,skiprows = 2): #TODO: Implement tests
        """
        Reads an OBJ mesh file and processes its vertices, vertex normals, and faces.
        Parameters:
        skiprows (int): Number of rows to skip at the beginning of the file. Default is 2.
        Returns:
        None: The function sets the following attributes on the object:
            - self.face_df: A DataFrame containing the processed face data with vertex and normal coordinates.
            - self.vertex_df: A DataFrame containing the processed vertex data with renamed columns for compatibility with vapc.
        Notes:
        - The function assumes that the OBJ file contains vertices ('v'), vertex normals ('vn'), and faces ('f').
        - Faces are assumed to be defined in the format: f v1//vn1 v2//vn2 v3//vn3.
        - Only one file can be read at a time. If more than one file is provided, the function will print a message and return.
        """
        if len(self.files)>1:
            print('Only one file can be read at a time')
            return
        df = pd.read_csv(self.files[0], sep=' ', header=None, names=['prefix','x', 'y', 'z'], skiprows=skiprows)
        df = df[df['prefix'].isin(['v','vn','f'])]
        v_df  = df[df['prefix'] == 'v'].reset_index(drop=True)
        vn_df = df[df['prefix'] == 'vn'].reset_index(drop=True)
        f_df  = df[df['prefix'] == 'f'].reset_index(drop=True)
        v_df['vertex_id'] = v_df.index + 1
        vn_df['normal_id'] = vn_df.index + 1

        # Assume faces are defined as:
        #   f 1//1 2//2 3//3
        f_df[['v1', 'vn1']] = f_df['x'].str.split('//', expand=True)
        f_df[['v2', 'vn2']] = f_df['y'].str.split('//', expand=True)
        f_df[['v3', 'vn3']] = f_df['z'].str.split('//', expand=True)


        for col in ['v1', 'vn1', 'v2', 'vn2', 'v3', 'vn3']:
            f_df[col] = f_df[col].astype(int)

        # Work on a copy of f_df for merging:
        face = f_df.copy()

        # --- Merge for vertex 1 and its normal ---
        # Prepare vertex data for vertex 1
        v1_df = v_df[['vertex_id', 'x', 'y', 'z']].rename(
            columns={'x': 'v1_x', 'y': 'v1_y', 'z': 'v1_z'}
        )
        face = face.merge(v1_df, left_on='v1', right_on='vertex_id', how='left')
        face.drop(columns='vertex_id', inplace=True)  # drop the temporary merge key
        # Create mapping Series for vertices and normals
        v_map_x = v_df.set_index('vertex_id')['x']
        v_map_y = v_df.set_index('vertex_id')['y']
        v_map_z = v_df.set_index('vertex_id')['z']

        vn_map_x = vn_df.set_index('normal_id')['x']
        vn_map_y = vn_df.set_index('normal_id')['y']
        vn_map_z = vn_df.set_index('normal_id')['z']

        # Map the values for each vertex of each face
        face['v1_x'] = face['v1'].map(v_map_x)
        face['v1_y'] = face['v1'].map(v_map_y)
        face['v1_z'] = face['v1'].map(v_map_z)
        face['vn1_x'] = face['vn1'].map(vn_map_x)
        face['vn1_y'] = face['vn1'].map(vn_map_y)
        face['vn1_z'] = face['vn1'].map(vn_map_z)

        face['v2_x'] = face['v2'].map(v_map_x)
        face['v2_y'] = face['v2'].map(v_map_y)
        face['v2_z'] = face['v2'].map(v_map_z)
        face['vn2_x'] = face['vn2'].map(vn_map_x)
        face['vn2_y'] = face['vn2'].map(vn_map_y)
        face['vn2_z'] = face['vn2'].map(vn_map_z)

        face['v3_x'] = face['v3'].map(v_map_x)
        face['v3_y'] = face['v3'].map(v_map_y)
        face['v3_z'] = face['v3'].map(v_map_z)
        face['vn3_x'] = face['vn3'].map(vn_map_x)
        face['vn3_y'] = face['vn3'].map(vn_map_y)
        face['vn3_z'] = face['vn3'].map(vn_map_z)
        self.face_df = face
        #Make input suitable for vapc
        self.vertex_df = v_df.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'})
        
        self.vertex_df["X"] = self.vertex_df["X"].astype(float)
        self.vertex_df["Y"] = self.vertex_df["Y"].astype(float)
        self.vertex_df["Z"] = self.vertex_df["Z"].astype(float)
        #remove prefix as it is clear from the name
        self.vertex_df = self.vertex_df.drop(columns=['prefix'])

    def save_obj_mesh(self, outfile):#TODO: Implement tests
        """
        Save the faces DataFrame (returned by open_obj) to a new OBJ file.
        
        This function extracts unique vertex and normal definitions from the face_df,
        re-indexes them, and writes out vertex (v), vertex normal (vn), and face (f)
        definitions to outfile.
        
        Parameters:
        face_df: DataFrame returned by open_obj. It must include columns:
                'v1_x', 'v1_y', 'v1_z', 'v2_x', 'v2_y', 'v2_z', 'v3_x', 'v3_y', 'v3_z',
                'vn1_x', 'vn1_y', 'vn1_z', 'vn2_x', 'vn2_y', 'vn2_z', 'vn3_x', 'vn3_y', 'vn3_z',
                as well as the face index columns 'v1', 'vn1', 'v2', 'vn2', 'v3', 'vn3'.
        outfile: String path for the new OBJ file to be written.
        """
        face_df = self.face_df
        #If face df is empty, we can not save anything
        if face_df.shape[0] == 0:
            print("No faces to save")
            return
        # --- Extract Vertex Data ---
        # Rename columns for each face vertex group to a common format
        v1 = face_df[['v1_x', 'v1_y', 'v1_z']].rename(columns={'v1_x':'x', 'v1_y':'y', 'v1_z':'z'})
        v2 = face_df[['v2_x', 'v2_y', 'v2_z']].rename(columns={'v2_x':'x', 'v2_y':'y', 'v2_z':'z'})
        v3 = face_df[['v3_x', 'v3_y', 'v3_z']].rename(columns={'v3_x':'x', 'v3_y':'y', 'v3_z':'z'})
        # Concatenate and remove duplicates
        vertices = pd.concat([v1, v2, v3], ignore_index=True)
        unique_vertices = vertices.drop_duplicates().reset_index(drop=True)
        unique_vertices['new_index'] = unique_vertices.index + 1  # OBJ indices start at 1

        # --- Extract Normal Data ---
        n1 = face_df[['vn1_x', 'vn1_y', 'vn1_z']].rename(columns={'vn1_x':'x', 'vn1_y':'y', 'vn1_z':'z'})
        n2 = face_df[['vn2_x', 'vn2_y', 'vn2_z']].rename(columns={'vn2_x':'x', 'vn2_y':'y', 'vn2_z':'z'})
        n3 = face_df[['vn3_x', 'vn3_y', 'vn3_z']].rename(columns={'vn3_x':'x', 'vn3_y':'y', 'vn3_z':'z'})
        normals = pd.concat([n1, n2, n3], ignore_index=True)
        unique_normals = normals.drop_duplicates().reset_index(drop=True)
        unique_normals['new_index'] = unique_normals.index + 1

        # --- Create Mapping Dictionaries ---
        # Map coordinate tuples to the new index
        vertex_map = { (row['x'], row['y'], row['z']): row['new_index'] 
                    for _, row in unique_vertices.iterrows() }
        normal_map = { (row['x'], row['y'], row['z']): row['new_index'] 
                    for _, row in unique_normals.iterrows() }

        # --- Update Face DataFrame with New Indices ---
        # We map each face's original vertex coordinates to the new index.
        # Note: Using apply is simple (though not super fast for huge files)
        face_df = face_df.copy()  # avoid modifying the original
        face_df['nv1'] = face_df.apply(lambda r: vertex_map[(r['v1_x'], r['v1_y'], r['v1_z'])], axis=1)
        face_df['nv2'] = face_df.apply(lambda r: vertex_map[(r['v2_x'], r['v2_y'], r['v2_z'])], axis=1)
        face_df['nv3'] = face_df.apply(lambda r: vertex_map[(r['v3_x'], r['v3_y'], r['v3_z'])], axis=1)
        
        face_df['nvn1'] = face_df.apply(lambda r: normal_map[(r['vn1_x'], r['vn1_y'], r['vn1_z'])], axis=1)
        face_df['nvn2'] = face_df.apply(lambda r: normal_map[(r['vn2_x'], r['vn2_y'], r['vn2_z'])], axis=1)
        face_df['nvn3'] = face_df.apply(lambda r: normal_map[(r['vn3_x'], r['vn3_y'], r['vn3_z'])], axis=1)

        # --- Write the New OBJ File ---
        with open(outfile, 'w') as f:
            # Write vertex definitions
            for _, row in unique_vertices.iterrows():
                f.write(f"v {row['x']} {row['y']} {row['z']}\n")
            # Write vertex normal definitions
            for _, row in unique_normals.iterrows():
                f.write(f"vn {row['x']} {row['y']} {row['z']}\n")
            # Write face definitions.
            # Each face line follows the format: f v_index//vn_index v_index//vn_index v_index//vn_index
            for _, row in face_df.iterrows():
                f.write(f"f {row['nv1']}//{row['nvn1']} {row['nv2']}//{row['nvn2']} {row['nv3']}//{row['nvn3']}\n")
