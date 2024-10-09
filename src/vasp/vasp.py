#For VASP:
import numpy as np
import math
from scipy import stats
from scipy.spatial import KDTree
import pandas as pd
# import pandasql as ps
from .utilities import trace,timeit
from .data_handler import DATA_HANDLER
from itertools import combinations
import time
import sys



class VASP:
    def __init__(self,
                 voxel_size:float,
                 origin:list = {},
                 attributes:dict = {},
                 compute:list = [],
                 return_at:str = "closest_to_center_of_gravity"):
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
        self.attributes_up_to_data = False
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
        self.distance_to_center_of_gravity = False
        self.closest_to_center_of_gravity = False
        self.center_of_voxel = False
        self.corner_of_voxel = False
        self.attributes_per_voxel = False
        self.drop_columns = []
        self.new_column_names = {}
        self.reduced = False
        self.reduction_point = False
        self.offset_applied = False
        
    @trace
    @timeit
    def get_data_from_data_handler(self,data_handler):
        """
        Gets dataframe from the data handler.
        """
        self.df = data_handler.df
        self.original_attributes = self.df.columns
        
    @trace
    @timeit
    def compute_reduction_point(self):
        self.reduction_point = [int(self.df["X"].min()),int(self.df["Y"].min()),int(self.df["Z"].min())]

    @trace
    @timeit
    def compute_offset(self):
        if self.reduction_point is False:
            self.compute_reduction_point()
        if self.offset_applied:
            self.df["X"] += self.reduction_point[0]
            self.df["Y"] += self.reduction_point[1]
            self.df["Z"] += self.reduction_point[2]
            self.offset_applied = False
        else:
            self.df["X"] -= self.reduction_point[0]
            self.df["Y"] -= self.reduction_point[1]
            self.df["Z"] -= self.reduction_point[2]
            self.offset_applied = True
        
        
    @trace
    @timeit
    def voxelize(self):
        """
        Computes the voxel coordinates for each point.
        """
        for i,dim in enumerate(["X", "Y", "Z"]):
            self.df[f"voxel_{dim.lower()}"] = np.floor((self.df[dim] - self.origin[i]) / self.voxel_size).astype(int)

        #self.df["voxel_id"] = self.df.iloc[:, -3:].astype(str).apply('_'.join, axis=1)
        self.voxelized = True

    @trace
    @timeit
    def compute_requested_attributes(self):
        """
        Computes attributes based on the calculate input list. [optional]
        """
        if "big_int_index" in self.compute:                 self.compute_big_int_index()
        if "hash_index" in self.compute:                    self.compute_hash_index()
        if "point_count" in self.compute:                   self.compute_point_count()
        if "point_density" in self.compute:                 self.compute_point_density()
        if "percentage_occupied" in self.compute:           self.compute_percentage_occupied()
        if "covariance_matrix" in self.compute:             self.compute_covariance_matrix()
        if "eigenvalues" in self.compute:                   self.compute_eigenvalues()
        if "geometric_features" in self.compute:            self.compute_geometric_features()
        if "center_of_gravity" in self.compute:             self.compute_center_of_gravity()
        if "distance_to_center_of_gravity" in self.compute: self.compute_distance_to_center_of_gravity()
        if "std_of_cog" in self.compute:                    self.compute_std_of_cog()
        if "closest_to_center_of_gravity" in self.compute:  self.compute_closest_to_center_of_gravity()
        if "center_of_voxel" in self.compute:               self.compute_center_of_voxel()
        if "corner_of_voxel" in self.compute:               self.compute_corner_of_voxel()

    # @trace
    # @timeit
    # def compute_requested_statistics_per_attributes_old(self):
    #     """
    #     Computes the statistics requested per existing attribute. Can not 
    #     be used for calculating mode (use numpy implementation for that)
    #     """
    #     if self.voxelized is False:
    #         self.voxelize()
    #         self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
    #     grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"]).agg(self.attributes).reset_index()
    #     self.new_column_names = {}
    #     for attr in self.attributes:
    #         self.new_column_names.update({attr:"statistics_%s"%attr})
    #         self.drop_columns += ["statistics_%s"%attr]
    #     grouped.rename(columns=self.new_column_names, inplace=True)
    #     self.df = self.df.merge(grouped, how="left", on=["voxel_x", "voxel_y", "voxel_z"])
    #     self.attributes_per_voxel = True

    def update_attribute_dictionary(self, 
                                    remove_cols = ["X","Y","Z",'bit_fields', 
                                              'raw_classification','scan_angle_rank', 
                                              'user_data', 'point_source_id']):
        """
        If not specified, for each attribute the mean will be computed if point cloud is voxelized. If specified, the selected statistic will me computed
        """
        # list(map(self.attributes.remove,["X","Y","Z"])) #remove XYZ as xyz should be read directly to not scale and shift the points in an extra step.

        original_attributes = list(self.original_attributes)
        for dc in remove_cols:
            try:
                original_attributes.remove(dc)
            except:
                pass
        # original_attributes = list(map(original_attributes.remove,remove_cols))
        attributes = {}
        for col in original_attributes:
            attributes[col] = "mean"
        for attr in self.attributes.keys():
            attributes[attr] = self.attributes[attr]
        self.attributes = attributes
        self.attributes_up_to_data = True

    @trace
    @timeit
    def compute_requested_statistics_per_attributes(self):
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
            - mode_count
        """

        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        if self.attributes_up_to_data is False:
            self.update_attribute_dictionary()
        df_temp_subset = self.df[["voxel_x", "voxel_y", "voxel_z"]+list(self.attributes.keys())]
        dtypes = [(col, df_temp_subset[col].dtypes) for col in df_temp_subset.columns]
        data = np.array([tuple(row) for row in df_temp_subset.values], dtype=dtypes)
        sorted_indices = np.lexsort((data["voxel_z"], data["voxel_y"], data["voxel_x"]))
        sorted_data = data[sorted_indices]
        groups, indices = np.unique(sorted_data[["voxel_x", "voxel_y", "voxel_z"]], return_index=True)#, axis=0)
        all_aggregated_data = {}
        all_aggregated_datalist = []
        final_dtype = [
            ("voxel_x", groups["voxel_x"].dtype), 
            ("voxel_y", groups["voxel_y"].dtype), 
            ("voxel_z", groups["voxel_z"].dtype)
            ]
        local_names_col_names = {}
        local_names = []
        print("Computing stats")
        for attr in self.attributes.keys():
            if not type(self.attributes[attr]) is list:
                self.attributes[attr] = [self.attributes[attr]]
            for enum,stat_request in enumerate(self.attributes[attr]):
                if "mode_count" in stat_request:
                    print(self.attributes[attr])
                    percentage = float(stat_request.split(",")[-1])
                    start = time.time()
                    sorted_data_attr = sorted_data[attr]
                    sorted_data_attr = np.array(sorted_data_attr, dtype=int)
                    split_arr = np.array_split(sorted_data_attr, indices[1:])
                    bc = list(map(np.bincount, split_arr))
                    bc_sum = list(map(sum, bc))
                    bc_prop = list(map(np.divide, bc, bc_sum))
                    comparaison = list(map(lambda sublist: sublist > percentage, bc_prop))
                    aggregated_data = list(map(sum, comparaison))
                    end = time.time()
                    print(f"Computing 'mode_count' for attribute '{attr}' took: {end - start:.4f} sec")
                elif stat_request == "mode":
                    start = time.time()
                    aggregated_data = list(map(lambda i: np.apply_along_axis(lambda x: [np.bincount(x.astype(int)).argmax()], axis=0, arr=sorted_data[attr][indices[i]:indices[i + 1]])[0], range(len(indices) - 1)))
                    aggregated_data.append(pd.Series(sorted_data[attr][indices[-1]:]).mode()[0])
                    end = time.time()
                    print(f"Computing 'mode' for attribute '{attr}' took: {end - start:.4f} sec")
                elif stat_request == "sum":
                    aggregated_data = [sorted_data[attr][indices[i]:indices[i + 1]].sum() for i in range(len(indices) - 1)]
                    aggregated_data.append(sorted_data[attr][indices[-1]:].sum())
                elif stat_request == "mean":
                    aggregated_data = [sorted_data[attr][indices[i]:indices[i + 1]].mean() for i in range(len(indices) - 1)]
                    aggregated_data.append(sorted_data[attr][indices[-1]:].mean())
                elif stat_request == "median":
                    aggregated_data = [np.median(sorted_data[attr][indices[i]:indices[i + 1]]) for i in range(len(indices) - 1)]
                    aggregated_data.append(np.median(sorted_data[attr][indices[-1]:]))
                elif stat_request == "min":
                    aggregated_data = [sorted_data[attr][indices[i]:indices[i + 1]].min() for i in range(len(indices) - 1)]
                    aggregated_data.append(sorted_data[attr][indices[-1]:].min())
                elif stat_request == "max":
                    aggregated_data = [sorted_data[attr][indices[i]:indices[i + 1]].max() for i in range(len(indices) - 1)]
                    aggregated_data.append(sorted_data[attr][indices[-1]:].max())
                else:
                    print("Aggregation type unknown for %s"%attr)
                    self.drop_columns(attr)
                    continue

                # all_aggregated_data[attr+"_%s"%enum] = aggregated_data
                all_aggregated_datalist.append(aggregated_data)
                final_dtype += [(attr+"_%s"%enum,np.array(aggregated_data).dtype)]
                local_names_col_names.update({attr+"_%s"%enum:"%s_%s"%(attr,stat_request)})
                local_names.append(attr+"_%s"%enum)
                #If attributes should not be saved with stat indication activate lower line.
                # self.drop_columns += ["%s_%s"%attr,self.attributes[attr]]
                # local_names.append(attr)
        combined_data = [tuple(list(group) + [agg[i] for agg in all_aggregated_datalist]) for i, group in enumerate(groups)]

        result_array = np.array(combined_data, dtype=final_dtype)
        grouped = pd.DataFrame(result_array,columns = ["voxel_x", "voxel_y", "voxel_z"]+local_names)
        grouped.rename(columns=local_names_col_names, inplace=True)
        self.df = self.df.merge(grouped, how="left", on=["voxel_x", "voxel_y", "voxel_z"])
        self.attributes_per_voxel = True

    @trace
    @timeit
    def compute_big_int_index_old(self):
        """
        Computes a big int index for all occupied voxels (as a str).
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        
        self.df.loc[:,"big_int_index"] = (self.df.loc[:, 'voxel_x'].astype(str)+ self.df.loc[:, 'voxel_y'].astype(str)+self.df.loc[:, 'voxel_z'].astype(str)).astype(int)
        self.big_int_index = True
       
    @trace
    @timeit
    def compute_big_int_index(self,
                            n = 1000000000):
        """
        Computes a big int index for all occupied voxels (as a int).
        Do not set n > 1000000000, errors will occur.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        
        self.df.loc[:,"big_int_index"] = self.df.loc[:, 'voxel_x']*n**2+ self.df.loc[:, 'voxel_y']*n+self.df.loc[:, 'voxel_z']
        self.big_int_index = True
       

   
    @trace
    @timeit
    def mask_by_voxels(self,
                     mask_file
                     ):
        if self.big_int_index is False:
            self.compute_big_int_index()
            self.drop_columns += ["big_int_index"]
        #Load mask file
        dh = DATA_HANDLER([mask_file],
                        attributes={})
        dh.load_las_files()
        vasp_mask = VASP(self.voxel_size,
                        self.origin,
                        {})
        vasp_mask.get_data_from_data_handler(dh)
        vasp_mask.voxelize()
        self.compute_hash_index()
        #mask_indices = np.unique(vasp_mask.df[["voxel_x", "voxel_y", "voxel_z"]],axis = 1)
        mask_indices = np.unique(vasp_mask.df["hash_index"])
        vasp_mask.df = vasp_mask.df[self.df["hash_index"].isin(mask_indices)]
        #vasp_mask.df[["voxel_x", "voxel_y", "voxel_z"]] = mask_indices
        vasp_mask.compute_big_int_index()
        vasp_mask.compute_hash_index()
        hash_indices = np.unique(vasp_mask.df["hash_index"])
        print(hash_indices)
        #mask by hash index
        print(self.df)
        print("before filtering:",self.df.shape)
        self.df = self.df[self.df["hash_index"].isin(hash_indices)]
        print("after hash filtering:",self.df.shape)
        #remove wrong values by big int index
        big_int_indices = np.unique(vasp_mask.df["big_int_index"])
        self.compute_big_int_index()
        self.df = self.df[self.df["big_int_index"].isin(big_int_indices)]
        print("after big int filtering:",self.df.shape)
        self.df = self.df.drop(["voxel_x", "voxel_y", "voxel_z","big_int_index","hash_index"],axis = 1)

    @trace
    @timeit
    def compute_voxel_buffer(self, buffer_size:int = 1):
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
            
        coords = np.array((self.df["voxel_x"],self.df["voxel_y"],self.df["voxel_z"])).T
        offsets = np.arange(-buffer_size, buffer_size + 1)
        all_combinations = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1, 3)
        expanded_coords = coords[:, np.newaxis, :] + all_combinations
        result = expanded_coords.reshape(-1, 3)
        self.df = pd.DataFrame(np.array(result), columns = ["voxel_x", "voxel_y", "voxel_z"])


    @trace
    @timeit
    def select_by_mask_old(self,
                     vasp_mask):
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        if self.hash_index is False:
            self.compute_hash_index()
            self.drop_columns += ["hash_index"]
        if vasp_mask.voxelized is False:
            vasp_mask.voxelize()
            
        vasp_mask.compute_hash_index()
        #mask by hash index
        print("Points before filtering:",self.df.shape)
        self.df = self.df[self.df["hash_index"].isin(vasp_mask.df["hash_index"])]
        print("Points after filtering:",self.df.shape)
        self.df = self.df.drop(["voxel_x", "voxel_y", "voxel_z","hash_index"],axis = 1)
        
    @trace
    @timeit
    def select_by_mask(self,
                     vasp_mask,
                     mask_attribute = "big_int_index",
                     segment_in_or_out = "in"):
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        if vasp_mask.voxelized is False:
            vasp_mask.voxelize()
        if mask_attribute not in self.df.columns:
            self.compute = [mask_attribute]
            self.compute_requested_attributes()
        if mask_attribute not in vasp_mask.df.columns:
            vasp_mask.compute = [mask_attribute]
            vasp_mask.compute_requested_attributes()
       
        #mask by attribute
        print("Points before filtering:",self.df.shape)
        if segment_in_or_out == "in":
            self.df = self.df[self.df[mask_attribute].isin(vasp_mask.df[mask_attribute])]
        elif segment_in_or_out == "out":
            self.df = self.df[~self.df[mask_attribute].isin(vasp_mask.df[mask_attribute])]
        else:
            print("Mask either in (keep overlap) our out (remove overlap)")
            return
        print("Points after filtering:",self.df.shape)
        self.df = self.df.drop(["voxel_x", "voxel_y", "voxel_z",mask_attribute],axis = 1)
        self.voxelized = False

    @trace
    @timeit
    def select_by_mask_old2(self,
                     vasp_mask,
                     mask_attribute = "big_int_index"):
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
            self.drop_columns += ["hash_index"]
        if vasp_mask.voxelized is False:
            vasp_mask.voxelize()
        print(self.df)
        self.compute_big_int_index()
        print(vasp_mask.df)
        vasp_mask.compute_big_int_index()
        #mask by attribute
        print("Points before filtering:",self.df.shape)
        self.df = self.df[self.df[mask_attribute].isin(vasp_mask.df[mask_attribute])]
        print("Points after filtering:",self.df.shape)
        self.df = self.df.drop(["voxel_x", "voxel_y", "voxel_z","big_int_index"],axis = 1)

        

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
        self.df["hash_index"] = ((self.df["voxel_x"] * p1) ^ (self.df["voxel_y"] * p2) ^ (self.df["voxel_z"] * p3)) % n
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
        print("%s percent of the voxel space is occupied"%self.percentage_occupied)

   
    def compute_distance_to_center_of_gravity(self):
        x_diff = self.df["X"]-self.df["cog_x"]
        y_diff = self.df["Y"]-self.df["cog_y"]
        z_diff = self.df["Z"]-self.df["cog_z"]
        distances = np.sqrt(x_diff**2+y_diff**2+z_diff**2)
        self.df["distance"] = distances
        self.distance_to_center_of_gravity = True

    @trace
    @timeit
    def compute_closest_to_center_of_gravity(self):
        if not self.center_of_gravity:
            self.compute_center_of_gravity()
            self.drop_columns += ["cog_x", "cog_y", "cog_z"]
        if not self.distance_to_center_of_gravity:
            self.compute_distance_to_center_of_gravity()
            self.drop_columns += ["distance"]
        grouped = self.df.groupby(["cog_x", "cog_y", "cog_z"])
        self.voxel_cls2cog = grouped[["distance"]].min().reset_index()
        self.voxel_cls2cog.rename(columns={"distance": "min_distance"}, inplace=True)
        self.df = self.df.merge(self.voxel_cls2cog, how="left", on=["cog_x", "cog_y", "cog_z"])
        self.closest_to_center_of_gravity = True

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
    def compute_std_of_cog(self):
        """
        Computes the center of gravity for all occupied voxels.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        self.voxel_cog = grouped[["X", "Y", "Z"]].std().reset_index()
        self.voxel_cog.rename(columns={"X": "std_x", "Y": "std_y", "Z": "std_z"}, inplace=True)
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
        covariance_df = pd.DataFrame(cov_df.values.tolist(), index=cov_df.index, columns=col_names).reset_index()
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
        col_names = ["Eigenvalue_1", "Eigenvalue_2","Eigenvalue_3"]
        eigenvalue_df = pd.DataFrame(eig_df.values.tolist(), index=eig_df.index, columns=col_names).reset_index()
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
            self.drop_columns += ["Eigenvalue_1", "Eigenvalue_2","Eigenvalue_3"]
        self.df["Sum_of_Eigenvalues"] = self.df["Eigenvalue_1"]+self.df["Eigenvalue_2"]+self.df["Eigenvalue_3"]
        self.df["Omnivariance"] = (self.df["Eigenvalue_1"]*self.df["Eigenvalue_2"]*self.df["Eigenvalue_3"])**(1/3)
        try:
            self.df["Eigentropy"] = -1*(self.df["Eigenvalue_1"]*math.log(self.df["Eigenvalue_1"])+self.df["Eigenvalue_2"]*math.log(self.df["Eigenvalue_2"])+self.df["Eigenvalue_3"]*math.log(self.df["Eigenvalue_3"]))
        except:
            self.df["Eigentropy"] = np.nan
        self.df["Anisotropy"] = (self.df["Eigenvalue_1"]-self.df["Eigenvalue_3"])/self.df["Eigenvalue_1"]
        self.df["Planarity"] = (self.df["Eigenvalue_2"]-self.df["Eigenvalue_3"])/self.df["Eigenvalue_1"]
        self.df["Linearity"] = (self.df["Eigenvalue_1"]-self.df["Eigenvalue_2"])/self.df["Eigenvalue_1"]
        self.df["Surface_Variation"] = self.df["Eigenvalue_3"]/self.df["Sum_of_Eigenvalues"]
        self.df["Sphericity"] = self.df["Eigenvalue_3"]/self.df["Eigenvalue_1"]
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
            if self.center_of_voxel is False or not hasattr(self.df,"center_x"):
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

        elif self.return_at == "closest_to_center_of_gravity":
            if not self.center_of_gravity:
                self.compute_center_of_gravity()
                self.drop_columns+=["cog_x", "cog_y", "cog_z"]
            if not self.closest_to_center_of_gravity:
                self.compute_closest_to_center_of_gravity()
                self.drop_columns+=["min_distance"]
            self.df = self.df[self.df["distance"]==self.df["min_distance"]]
        else:
            print(f"Voxels cannot be reduced to {self.return_at},try 'center_of_gravity', 'center_of_voxel', 'closest_to_center_of_gravity', or 'corner_of_voxel'")
            return

        #Update columns with their required values
        for col_name in self.new_column_names.keys():
            self.df[col_name] = self.df[self.new_column_names[col_name]]
        self.df = self.df.drop(set(self.drop_columns),axis = 1)
        self.df = self.df.drop_duplicates()
        self.df = self.df.groupby(['X', 'Y', 'Z'], as_index=False).median()
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
            self.df = self.df[self.df[filter_attribute]>=filter_value]
        elif min_max_eq == "max":
            self.df = self.df[self.df[filter_attribute]<filter_value]
        elif min_max_eq == "max_eq":
            self.df = self.df[self.df[filter_attribute]<=filter_value]
        else:
            print("Filter invalid, use eq, min, min_eq, max, and max_eq only.")



    @trace
    @timeit
    def compute_clusters(self,
                         cluster_distance,
                         cluster_by = ["voxel_x","voxel_y","voxel_z"]):
        def _update_existing_objects(O, indices, relevantPoints):
            obj, counts = np.unique(O[indices[relevantPoints]], return_counts=True)
            existingObjects = obj[obj > 0]
            for existingObject in existingObjects:
                mask = np.where(O == existingObject)
                O[mask] = existingObjects.min()
            O[indices[relevantPoints]] = existingObjects.min()

        nr_of_neighbours = 50
        pts = np.array(self.df[cluster_by])
        tree = KDTree(pts)
        O = np.zeros_like(self.df[cluster_by[0]])
        self.objectCounter = 1.

        for i, point in enumerate(pts):
            distances, indices = tree.query(point, nr_of_neighbours)
            relevantPoints = np.where(distances <= cluster_distance)
            if O[indices[relevantPoints]].max() < 1:
                O[indices[relevantPoints]] = self.objectCounter
                self.objectCounter += 1
            else:
                _update_existing_objects(O,indices, relevantPoints)

        oids, cts = np.unique(O, return_counts=True)
        ct_df = pd.DataFrame(data = np.array((oids,cts)).T,columns = ["cluster_id","cluster_size"])
        self.df["cluster_id"] = O
        self.df = self.df.merge(ct_df,on = "cluster_id",how = "left",validate="many_to_one")


