#For VASP:
import numpy as np
import math
from scipy import stats
import pandas as pd
from utilities import trace,timeit

class VASP:
    def __init__(self,
                 voxel_size:float,
                 origin:list,
                 attributes:dict,
                 compute:list = [],
                 return_at:str = "center_of_gravity"):
        """
        VASP [Under construction].

        Parameters:
        - voxel_size (float): Defines voxel size..
        - origin (list): Defines origin of voxel_space.
        - attributes (dict): This dictionary contains information about which attributes to read 
                                and what statistics to carry out on them.
        - compute (list): Optional list containing name of attributes that will be calculated if 
                                calling 'compute_requested_attributes'
        - return_at (str): Specifies what point the data will be reduced to if calling 'reduce_to_voxels'. 
                                Determines the location of each output voxel.
        """
        #Relevant input:
        self.voxel_size = voxel_size
        self.origin = origin
        self.attributes = attributes
        self.compute = compute
        self.return_at = return_at
        #Calculations not applied yet:
        self.voxelized = False
        self.big_int_index = False
        self.hash_index = False
        self.point_count = False
        self.point_density = False
        self.percentage_occupied = False
        self.covariance_matrix = False
        self.eigenvalues = False
        self.geometric_features = False
        self.center_of_gravity = False
        self.center_of_voxel = False
        self.corner_of_voxel = False
        self.attributes_per_voxel = False
        self.drop_columns = []
        self.new_column_names = {}
        self.reduced = False

    @trace
    @timeit
    def get_data_from_data_handler(self,data_handler):
        """
        Gets dataframe from the data handler.
        """
        self.df = data_handler.df

    @trace
    @timeit
    def voxelize(self):
        """
        Computes the voxel coordinates for each point.
        """
        for i,dim in enumerate(["X", "Y", "Z"]):
            self.df[f"voxel_{dim.lower()}"] = np.floor((self.df[dim] - self.origin[i]) / self.voxel_size).astype(int)
        self.voxelized = True

    @trace
    @timeit
    def compute_requested_attributes(self):
        """
        Computes attributes based on the calculate input list. [optional]
        """
        if "big_int_index" in self.compute:       self.compute_big_int_index()
        if "hash_index" in self.compute:          self.compute_hash_index()
        if "point_count" in self.compute:         self.compute_point_count()
        if "point_density" in self.compute:       self.compute_point_density()
        if "percentage_occupied" in self.compute: self.compute_percentage_occupied()
        if "covariance_matrix" in self.compute:   self.compute_covariance_matrix()
        if "eigenvalues" in self.compute:         self.compute_eigenvalues()
        if "geometric_features" in self.compute:  self.compute_geometric_features()
        if "center_of_gravity" in self.compute:   self.compute_center_of_gravity()
        if "center_of_voxel" in self.compute:     self.compute_center_of_voxel()
        if "corner_of_voxel" in self.compute:     self.compute_corner_of_voxel()

    @trace
    @timeit
    def compute_requested_statistics_per_attributes(self):
        """
        Computes the statistics requested per existing attribute. Can not 
        be used for calculating mode (use numpy implementation for that)
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"]).agg(self.attributes).reset_index()
        self.new_column_names = {}
        for attr in self.attributes:
            self.new_column_names.update({attr:"statistics_%s"%attr})
            self.drop_columns += ["statistics_%s"%attr]
        grouped.rename(columns=self.new_column_names, inplace=True)
        self.df = self.df.merge(grouped, how="left", on=["voxel_x", "voxel_y", "voxel_z"])
        self.attributes_per_voxel = True

    @trace
    @timeit
    def compute_requested_statistics_per_attributes_numpy(self):
        """
        Computes the statistics requested per existing attribute. Can also 
        be used for calculating mode. However needs more testing for now.
        If works, will replace pandas based method.
        Current options:
            - Mean
            - Median
            - Mode
            - Min
            - Max
            - Sum
        """

        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        df_temp_subset = self.df[["voxel_x", "voxel_y", "voxel_z"]+list(self.attributes.keys())]
        dtypes = [(col, df_temp_subset[col].dtype) for col in df_temp_subset.columns]
        data = np.array([tuple(row) for row in df_temp_subset.values], dtype=dtypes)
        sorted_indices = np.lexsort((data["voxel_z"], data["voxel_y"], data["voxel_x"]))
        sorted_data = data[sorted_indices]
        groups, indices = np.unique(sorted_data[["voxel_x", "voxel_y", "voxel_z"]], return_index=True, axis=0)
        all_aggregated_data = {}
        all_aggregated_datalist = []
        final_dtype = [
            ("voxel_x", groups["voxel_x"].dtype), 
            ("voxel_y", groups["voxel_y"].dtype), 
            ("voxel_z", groups["voxel_z"].dtype)
            ]
        
        self.new_column_names = {}
        local_names = []
        for attr in self.attributes.keys():
            if self.attributes[attr] == "mode":
                aggregated_data = [stats.mode(sorted_data[attr][indices[i]:indices[i + 1]])[0] for i in range(len(indices) - 1)]
                aggregated_data.append(stats.mode(sorted_data[attr][indices[-1]:])[0])
            elif self.attributes[attr] == "sum":
                aggregated_data = [sorted_data[attr][indices[i]:indices[i + 1]].sum() for i in range(len(indices) - 1)]
                aggregated_data.append(sorted_data[attr][indices[-1]:].sum())
            elif self.attributes[attr] == "mean":
                aggregated_data = [sorted_data[attr][indices[i]:indices[i + 1]].mean() for i in range(len(indices) - 1)]
                aggregated_data.append(sorted_data[attr][indices[-1]:].mean())
            elif self.attributes[attr] == "median":
                aggregated_data = [np.median(sorted_data[attr][indices[i]:indices[i + 1]]) for i in range(len(indices) - 1)]
                aggregated_data.append(np.median(sorted_data[attr][indices[-1]:]))
            elif self.attributes[attr] == "min":
                aggregated_data = [sorted_data[attr][indices[i]:indices[i + 1]].min() for i in range(len(indices) - 1)]
                aggregated_data.append(sorted_data[attr][indices[-1]:].min())
            elif self.attributes[attr] == "max":
                aggregated_data = [sorted_data[attr][indices[i]:indices[i + 1]].max() for i in range(len(indices) - 1)]
                aggregated_data.append(sorted_data[attr][indices[-1]:].max())
            else:
                print("Aggregation type unknown for %s"%attr)
                self.drop_columns(attr)
                continue

            all_aggregated_data[attr] = aggregated_data
            all_aggregated_datalist.append(aggregated_data)
            final_dtype += [(attr,np.array(aggregated_data).dtype)]
            self.new_column_names.update({attr:"statistics_%s"%attr})
            self.drop_columns += ["statistics_%s"%attr]
            local_names.append(attr)

        combined_data = [tuple(list(group) + [agg[i] for agg in all_aggregated_datalist]) for i, group in enumerate(groups)]
        result_array = np.array(combined_data, dtype=final_dtype)
        grouped = pd.DataFrame(result_array,columns = ["voxel_x", "voxel_y", "voxel_z"]+local_names)
        grouped.rename(columns=self.new_column_names, inplace=True)
        self.df = self.df.merge(grouped, how="left", on=["voxel_x", "voxel_y", "voxel_z"])
        self.attributes_per_voxel = True

    @trace
    @timeit
    def compute_big_int_index(self):
        """
        Computes a big int index for all occupied voxels (as a str).
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        self.df["big_int_index"] = (self.df["voxel_x"].astype(str)+ self.df["voxel_y"].astype(str)+ self.df["voxel_z"].astype(str))
        self.big_int_index = True
    
    @trace
    @timeit
    def compute_hash_index(self,
                             p1 = 76690892503, 
                             p2 = 15752609759, 
                             p3 = 27174879103, 
                             n = 2**100):
        """
        Computes the hash index for all occupied voxels.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        self.df["hash_index"] = (self.df["voxel_x"] * p1 ^ self.df["voxel_y"] * p2 ^ self.df["voxel_z"] * p3) % n
        self.hash_index = True

    @trace
    @timeit
    def compute_point_count(self):
        """
        Computes the point count for all occupied voxels. (Number of points within each voxel)
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        points_per_voxel = grouped.size().reset_index(name="point_count")
        self.df = self.df.merge(points_per_voxel, how="left", on=["voxel_x", "voxel_y", "voxel_z"])
        self.point_count = True

    @trace
    @timeit
    def compute_point_density(self):
        """
        Computes the point density for all occupied voxels.
        """
        if self.point_count is False:
            self.compute_point_count()
            self.drop_columns += ["point_count"]
        self.df["point_density"] = self.df["point_count"]/(self.voxel_size**3)
        self.point_density = True

    @trace
    @timeit
    def compute_percentage_occupied(self):
        """
        Computes the space occupied by voxels within the voxel space bounding box.
        Percentage occupied = Number of voxels occupied / Number of voxels within bounding box * 100
        """
        if self.big_int_index is False:
            self.compute_big_int_index()
            self.drop_columns += ["big_int_index"]
        x_min, x_max = self.df["voxel_x"].min(),self.df["voxel_x"].max()
        y_min, y_max = self.df["voxel_y"].min(),self.df["voxel_y"].max()
        z_min, z_max = self.df["voxel_z"].min(),self.df["voxel_z"].max()
        x_extent,y_extent,z_extent = x_max - x_min, y_max - y_min, z_max - z_min
        nr_of_voxels_within_bounding_box = x_extent*y_extent*z_extent
        nr_of_occupied_voxels = len(np.unique(self.df["big_int_index"]))
        self.percentage_occupied = round(nr_of_occupied_voxels/nr_of_voxels_within_bounding_box*100,2)
        print("%s of the voxel space is occupied"%self.percentage_occupied)

    @trace
    @timeit
    def compute_center_of_gravity(self):
        """
        Computes the center of gravity for all occupied voxels.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        self.voxel_cog = grouped[["X", "Y", "Z"]].mean().reset_index()
        self.voxel_cog.rename(columns={"X": "cog_x", "Y": "cog_y", "Z": "cog_z"}, inplace=True)
        self.df = self.df.merge(self.voxel_cog, how="left", on=["voxel_x", "voxel_y", "voxel_z"])
        self.center_of_gravity = True

    @trace
    @timeit
    def compute_center_of_voxel(self):
        """
        Computes the voxel center for all occupied voxels.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        self.df[["center_x", "center_y", "center_z"]] = (self.df[["voxel_x", "voxel_y", "voxel_z"]] * self.voxel_size) + self.voxel_size / 2 + self.origin
        self.center_of_voxel = True

    @trace
    @timeit
    def compute_corner_of_voxel(self):
        """
        Computes the minx, miny, minz corner for all occupied voxels.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        self.df[["corner_x", "corner_y", "corner_z"]] = (self.df[["voxel_x", "voxel_y", "voxel_z"]] * self.voxel_size) + self.origin
        self.corner_of_voxel = True

    @trace
    @timeit
    def compute_covariance_matrix(self):
        """
        Computes the covarianve matrix for all occupied voxels.
        """
        def _covariance(df):
            cov_matrix = df[["X", "Y", "Z"]].cov()
            return cov_matrix.values.flatten()
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        cov_df = grouped.apply(_covariance)
        col_names = ["cov_xx", "cov_xy", "cov_xz", "cov_yx", "cov_yy", "cov_yz", "cov_zx", "cov_zy", "cov_zz"]
        covariance_df = pd.DataFrame(cov_df.tolist(), index=cov_df.index, columns=col_names).reset_index()
        self.df = self.df.merge(covariance_df, how="left", on=["voxel_x", "voxel_y", "voxel_z"])
        self.covariance_matrix = True

    @trace
    @timeit
    def compute_eigenvalues(self):
        """
        Computes eigenvalues for all occupied voxels.
        !!! Eigenvectors are also calculated, might be interesting to add
        """
        def _eigenvalues(df):
            cov_matrix = df[["X", "Y", "Z"]].cov()
            if cov_matrix.isna().any().any():
                eigenValues =  np.array([np.nan,np.nan,np.nan])
            else:
                eigenValues, eigenVectors = np.linalg.eig(cov_matrix)
                # Currently sort eigenvalues but not the vectors

                idx = eigenValues.argsort()[::-1]   
                eigenValues = eigenValues[idx]
            return eigenValues.flatten()
        if self.covariance_matrix is False:
            self.compute_covariance_matrix()
            self.drop_columns += ["cov_xx", "cov_xy", "cov_xz", "cov_yx", "cov_yy", "cov_yz", "cov_zx", "cov_zy", "cov_zz"]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        eig_df = grouped.apply(_eigenvalues)
        col_names = ["Eigenvalue 1", "Eigenvalue 2","Eigenvalue 3"]
        eigenvalue_df = pd.DataFrame(eig_df.tolist(), index=eig_df.index, columns=col_names).reset_index()
        self.df = self.df.merge(eigenvalue_df, how="left", on=["voxel_x", "voxel_y", "voxel_z"])
        self.eigenvalues = True

    @trace
    @timeit
    def compute_geometric_features(self):
        """
        Computes geometric features for all occupied voxels.
        """
        if self.eigenvalues is False:
            self.compute_eigenvalues()
            self.drop_columns += ["Eigenvalue 1", "Eigenvalue 2","Eigenvalue 3"]
        self.df["Sum of Eigenvalues"] = self.df["Eigenvalue 1"]+self.df["Eigenvalue 2"]+self.df["Eigenvalue 3"]
        self.df["Omnivariance"] = (self.df["Eigenvalue 1"]*self.df["Eigenvalue 2"]*self.df["Eigenvalue 3"])**(1/3)
        try:
            self.df["Eigentropy"] = -1*(self.df["Eigenvalue 1"]*math.log(self.df["Eigenvalue 1"])+self.df["Eigenvalue 2"]*math.log(self.df["Eigenvalue 2"])+self.df["Eigenvalue 3"]*math.log(self.df["Eigenvalue 3"]))
        except:
            self.df["Eigentropy"] = np.nan
        self.df["Anisotropy"] = (self.df["Eigenvalue 1"]-self.df["Eigenvalue 3"])/self.df["Eigenvalue 1"]
        self.df["Planarity"] = (self.df["Eigenvalue 2"]-self.df["Eigenvalue 3"])/self.df["Eigenvalue 1"]
        self.df["Linearity"] = (self.df["Eigenvalue 1"]-self.df["Eigenvalue 2"])/self.df["Eigenvalue 1"]
        self.df["Surface Variation"] = self.df["Eigenvalue 3"]/self.df["Sum of Eigenvalues"]
        self.df["Sphericity"] = self.df["Eigenvalue 3"]/self.df["Eigenvalue 1"]
        self.geometric_features = True

    @trace
    @timeit
    def reduce_to_voxels(self): 
        """
        Reduce the DataFrame to only on value per Voxel. return_at defines what the X,Y, and Z coordinate 
        of the output will be.
        return_at overwrites X,Y,Z with:
            The center of each voxel containing points ("center_of_voxel") 
            The minx,miny,minz corner of each voxel containing points a("corner_of_voxel") 
            The center of gravity computed within each voxel containing points ("center_of_gravity") 
        """
        if self.return_at == "center_of_voxel":
            if self.corner_of_voxel is False:
                self.compute_center_of_voxel()
                self.drop_columns+=["center_x", "center_y", "center_z"]
            self.new_column_names.update({"X":"center_x","Y":"center_y","Z":"center_z"})

        elif self.return_at == "corner_of_voxel":
            if self.corner_of_voxel is False:
                self.compute_corner_of_voxel()
                self.drop_columns+=["corner_x", "corner_y", "corner_z"]
            self.new_column_names.update({"X":"corner_x","Y":"corner_y","Z":"corner_z"})
        
        elif self.return_at == "center_of_gravity":
            if not self.center_of_gravity:
                self.compute_center_of_gravity()
                self.drop_columns+=["cog_x", "cog_y", "cog_z"]
            self.new_column_names.update({"X":"cog_x","Y":"cog_y","Z":"cog_z"})
        else:
            print(f"Voxels cannot be reduced to {self.return_at},try 'center_of_gravity', 'center_of_voxel', or 'corner_of_voxel'")
            return

        #Update columns with their required values
        for col_name in self.new_column_names.keys():
            self.df[col_name] = self.df[self.new_column_names[col_name]]
        self.df = self.df.drop(set(self.drop_columns),axis = 1)
        self.df = self.df.drop_duplicates()
        self.reduced = True

    @trace
    @timeit
    def filter_attributes(self,
                          filter_attribute:str,
                          min_max_eq:str,
                          filter_value,
                          ):
        """
        Filters a DataFrame attribute based on specified criteria.
        This method modifies the DataFrame `self.df` by applying a filter condition based on the specified attribute, value, and filter type. 
        Filters include:
            equality ('eq') 
            minimum ('min') 
            minimum or equal ('min_eq') 
            maximum ('max')
            maximum or equal ('max_eq')

        Parameters:
        - filter_attribute (str): The attribute (column name) of the DataFrame to apply the filter on.
        - filter_value (Comparable): The value to compare the attribute against. Must be compatible with the type of the DataFrame attribute.
        - min_max_eq (str): A string specifying the type of filter to apply.

        Example:
        ```
        # Assuming `self.df` is a DataFrame with a column 'point_count'
        self.filter_attributes('point_count', 'min_eq', 30)
        # This will modify `self.df` to include only rows where 'point_count' is 30 or more.
        ```
        """
        min_max_eq = min_max_eq.lower()
        if min_max_eq == "eq":
            self.df = self.df[self.df[filter_attribute]==filter_value]
        elif min_max_eq == "min":
            self.df = self.df[self.df[filter_attribute]>filter_value]
        elif min_max_eq == "min_eq":
            self.df = self.df[filter_attribute>=filter_value]
        elif min_max_eq == "max":
            self.df = self.df[filter_attribute>filter_value]
        elif min_max_eq == "max_eq":
            self.df = self.df[filter_attribute>=filter_value]
        else:
            print("Filter invalid, use eq, min, min_eq, max, and max_eq only.")


