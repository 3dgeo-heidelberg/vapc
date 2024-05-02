from vasp import VASP
from data_handler import DATA_HANDLER
import os

base_folder = os.path.abspath(os.curdir)
test_folder = os.path.join(base_folder,"test_data")
test_file = os.path.join(test_folder,"vasp_in.laz")

test_out = os.path.join(test_folder,"vasp_out.laz")

test_ref = os.path.join(test_folder,"vasp_reference.laz")

vasp_config = {
			"voxel_size":0.25,
			"origin":[0,0,0],
			"attributes":{
				"red":"mean",
				"green":"mean",
				"blue":"mean"
			},
			"filter":{
				"filter_attribute":"point_count",
				"min_max_eq":"min",
				"filter_value":2
			},
			"compute":[ "big_int_index",
                        "hash_index",
                        "point_density",
                        "percentage_occupied",
                        "covariance_matrix",
                        "eigenvalues",
                        "geometric_features",
                        "center_of_gravity",
                        "distance_to_center_of_gravity",
                        "std_of_cog",
                        "closest_to_center_of_gravity",
                        "center_of_voxel",
                        "corner_of_voxel",
                                    ],
			"return_at":"closest_to_center_of_gravity"
		}

dh = DATA_HANDLER([test_file],
                  vasp_config["attributes"])
dh.load_las_files()
vasp = VASP(vasp_config["voxel_size"],
            vasp_config["origin"],
            vasp_config["attributes"],
            vasp_config["compute"],
            vasp_config["return_at"])
vasp.get_data_from_data_handler(dh)
vasp.voxelize()
vasp.compute_point_count()
vasp.filter_attributes( filter_attribute = "point_count",
                        min_max_eq = "min",
                        filter_value = 2,
                        )
vasp.compute_requested_statistics_per_attributes()
vasp.compute_requested_attributes()
vasp.reduce_to_voxels()
dh.df = vasp.df
dh.save_as_las(test_out)
