from data_handler import DATA_HANDLER
from vasp import VASP
from vasp_tools import *
from las_split_append_merge import *
import datetime
import json
import argparse
from utilities import *

VASP_VERSION = "0.0.0.1"

@trace
@timeit
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

        for sub_tile in all_tiles:
            if sub_tile == False:
                continue
            use_tool(vasp_command["tool"],
                     sub_tile,
                     sub_tile,
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
        use_tool(vasp_command["tool"],
                 file,
                 outfile,
                 voxel_size=voxel_size,
                 args=vasp_command["args"],
                 reduce_to = vasp_command["reduce_to"])

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
        laSZ_to_ply(infile=outfile,
                    outfile=ply_out,
                    voxel_size=voxel_size,
                    shift_to_center=False)
        os.remove(outfile)
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Use VASP.")
    parser.add_argument("config_file", help="Configuration of the VASP command")
    return parser.parse_args()

def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    #Example usage:
    #python src/vasp/main.py config_file.json
     
    # Get input 
    args = parse_args()

    # Load json
    config = load_config(args.config_file)

    # Apply command
    do_vasp_on_files(file=config["infile"],
                    out_dir=config["outdir"],
                    voxel_size=config["voxel_size"],
                    vasp_command=config["vasp_command"],
                    tile = config["tile"],
                    reduce_to=config["reduce_to"],
                    save_as=config["save_as"]
                    )
