from vasp.vasp_tools import *
from vasp.las_split_append_merge import *
import datetime
import json
import argparse
from vasp.utilities import *
# from vasp import __version__



@trace
@timeit
def do_vasp_on_files(file, 
                out_dir,
                voxel_size, 
                vasp_command,
                tile = False, 
                reduce_to = "closest_to_center_of_gravity",
                save_as = ".laz",
                doc = True):
    """
    Executes VASP operations on LAS/LAZ files based on the provided configuration.

    This function handles the voxelization process, optional tiling for large datasets,
    applies VASP tools, and saves the processed data in the desired format. It also
    manages temporary files and directories used during processing.

    Parameters
    ----------
    file : str or list of str
        Path to a single LAS/LAZ file or a list of paths to LAS/LAZ files to be processed.
    out_dir : str
        Directory where output files and configurations will be saved.
    voxel_size : float
        Defines the size of each voxel for processing.
    vasp_command : dict
        Dictionary containing VASP command configurations, including the tool to use
        and any additional arguments.
    tile : bool, optional
        If True, enables tiling for processing large datasets by partitioning into smaller tiles.
        Defaults to False.
    reduce_to : str, optional
        Specifies the method to reduce the DataFrame to one value per voxel. Options include:
        "closest_to_center_of_gravity", "center_of_voxel", "center_of_gravity","corner_of_voxel".
        Defaults to "closest_to_center_of_gravity".
    save_as : str, optional
        Specifies the output file format. Options include ".laz", ".las", ".ply".
        Defaults to ".laz".
    doc : bool, optional
        If True, saves the processing configuration as a JSON file for documentation purposes.
        Defaults to True.

    Returns
    -------
    bool
        Returns True if processing is successful.

    Raises
    ------
    FileNotFoundError
        If input files are not found.
    ValueError
        If invalid parameters are provided.
    KeyError
        If required keys are missing in the vasp_command dictionary.

    Notes
    -----
    - When tiling is enabled, temporary tiles are created in a subdirectory within out_dir.
    - After processing, temporary files and directories are cleaned up to save space.
    - The configuration used for processing is saved as a JSON file if `doc` is True.
    """
    #get_timestamp_for_output
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H-%M-%S")

    #document settings
    config = vasp_command
    config["vasp_version"] = get_version()
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
        del_inlas = False
        if type(file) == list:
            inlas = os.path.join(out_dir,"merged_las.las")
            las_merge(file,
                        inlas)
            del_inlas = True
        else:
            inlas = file
            
        all_tiles = las_create_3DTiles(lazfile = inlas,
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
        
        for two in tiles_without_buffer:
            os.remove(two)
        if del_inlas:
            os.remove(inlas)
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
    if doc:
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
    """
    Parses command-line arguments for the VASP command.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Use VASP.")
    parser.add_argument("config_file", help="Configuration of the VASP command")
    return parser.parse_args()

def load_config(config_file):
    """
    Loads the VASP command configuration from a JSON file.

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        A dictionary containing the VASP command configurations.
    """
    with open(config_file, 'r') as file:
        return json.load(file)

def main():
    """
    The main function that orchestrates the VASP processing workflow.

    This function parses command-line arguments, loads the configuration,
    and executes the VASP operations on the specified files.

    Raises
    ------
    Exception
        If any step in the processing workflow fails.

    Examples
    --------
    To run VASP with a configuration file:
    
    >>> python run.py config.json
    """
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


if __name__ == "__main__":
    main()
