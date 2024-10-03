from data_handler import DATA_HANDLER
from vasp import VASP
from vasp_tools import *
from las_split_append_merge import *
import datetime
import json


VASP_VERSION = "0.0.0.1"

def do_vasp_on_files(file, 
                out_dir,
                voxel_size, 
                vasp_command,
                tile = False, 
                reduce_to = "closest_to_center_of_gravity",
                save_as = ".laz"):
    #get_timestamp_for_output
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H-%M-%S")

    #document settings
    config = vasp_command
    config["vasp_version"]= VASP_VERSION
    config["file"] = file
    config["out_dir"] = out_dir
    config["voxel_size"] = voxel_size
    config["reduce_to"] = reduce_to
    config["tile"] = tile
    config["save_as"] = save_as
    #generate filepaths
    outfile = os.path.join(out_dir,vasp_command["tool"]+"_%s.las"%timestamp)
    outconfig = os.path.join(out_dir,vasp_command["tool"]+"_%s_cfg.json"%timestamp)
    #Use temporary tiling for big datasets
    if tile:
        tile_dir = os.path.join(out_dir,"temp_tiles")
        if not os.path.isdir(tile_dir):
            os.mkdir(tile_dir)
        all_tiles = las_create_3DTiles(lazfile = file,
                            outDir=tile_dir,
                            tilesize= tile,
                            tilename="temptile",
                            buffer= (voxel_size/2))

        for tile in all_tiles:
            if tile == False:
                continue
            use_tool(vasp_command["tool"],
                     tile,
                     tile,
                     voxel_size=voxel_size,
                     args=vasp_command["args"],
                     reduce_to = vasp_command["reduce_to"])

        tiles_without_buffer =  las_remove_buffer(tile_dir)

        las_merge(filepaths=    tiles_without_buffer,
                  outfile=      outfile)
        
        for file in tiles_without_buffer:
            os.remove(file)
        os.rmdir(tile_dir)        
    #work without tiles
    else:
        pass

    #save configuration for documentation
    with open(outconfig, 'w') as cfg:
        json.dump(config, 
                  cfg,
                  indent=4)


    if save_as == ".laz":
        laSZ_to_laSZ(outfile,outfile[:-4]+".laz")
        os.remove(outfile)
        return True
    if save_as == ".las":
        return True
    if save_as == ".ply":
        ply_out = outfile[:-4]+".ply"
        #XXX implement converter here
        os.remove(outfile)
        return True
    


# ##subsampling:
# config = {  "infile":r"C:\Users\ronny\repos\vasp\tests\test_data\vasp_in.laz",
#             "outdir":r"C:\Users\ronny\repos\vasp\tests\test_data",
#             "voxel_size":1,
#             "vasp_command":{
#                 "tool":"subsample",
#                 "args":{"sub_sample_method":"closest_to_center_of_gravity"}
#                 },
#             "tile":5,
#             "reduce_to":False
#           }



# ##masking:
# config = {  "infile":r"C:\Users\ronny\repos\vasp\tests\test_data\vasp_in.laz",
#             "outdir":r"C:\Users\ronny\repos\vasp\tests\test_data",
#             "voxel_size":0.2,
#             "vasp_command":{
#                 "tool":"mask",
#                 "args":{
#                     "maskfile":r"C:\Users\ronny\repos\vasp\tests\test_data\subsample_2024_10_03_22-27-27.laz",
#                     "segment_in_or_out":"in",
#                     "buffer_size":0}
#                 },
#             "tile":5,
#             "reduce_to":False
#           }

##compute voxel attributes:
config = {  "infile":r"C:\Users\ronny\repos\vasp\tests\test_data\vasp_in.laz",
            "outdir":r"C:\Users\ronny\repos\vasp\tests\test_data",
            "voxel_size":0.2,
            "vasp_command":{
                "tool":"compute",
                "args":{
                    "compute":["geometric_features",
                                  "point_count"]}
                },
            "tile":2,
            "reduce_to":"center_of_voxel"
          }

do_vasp_on_files(file=config["infile"],
                 out_dir=config["outdir"],
                 voxel_size=config["voxel_size"],
                 vasp_command=config["vasp_command"],
                 tile = config["tile"],
                 reduce_to=config["reduce_to"]
                 )


# do_vasp_on_files(files = [tree.laz], 
#                 out_dir = "./outfolder",
#                 voxel_size = 1,
#                 vasp_commands = [
#                         {"tool":"vasp_mask.py", 
#                         "mask_file": "mask.laz",
#                         "buffer_size":0.5},
#                         {"tool":"vasp_geom_feature.py"}
#                         ],
#                 tile = 10)



# test_in = r"C:\Users\ronny\repos\vasp\tests\test_data\vasp_in.laz"
# test_out = r"C:\Users\ronny\repos\vasp\tests\test_data\tiles"

# tilesize = 2.1
# voxel_size = 0.5
# las_create_3DTiles(test_in,
#     test_out,
#     tilesize,
#     tilename = "3DTile",
#     buffer = voxel_size*.5)

# las_remove_buffer(test_out)