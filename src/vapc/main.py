import os
import datetime
import json
import argparse
from vapc.vapc_tools import use_tool, lasz_to_ply
from vapc.las_split_append_merge import (
    las_merge,
    las_create_3dtiles,
    las_remove_buffer,
    lasz_to_lasz,
)
from vapc.utilities import trace, timeit, get_version


@trace
@timeit
def do_vapc_on_files(
    file,
    out_dir,
    voxel_size,
    vapc_command,
    tile=False,
    reduce_to="closest_to_center_of_gravity",
    save_as=".laz",
    doc=True,
):
    """
    Executes Vapc operations on LAS/LAZ files based on the provided configuration.

    This function handles the voxelization process, optional tiling for large datasets,
    applies Vapc tools, and saves the processed data in the desired format. It also
    manages temporary files and directories used during processing.

    Parameters
    ----------
    file : str or list of str
        Path to a single LAS/LAZ file or a list of paths to LAS/LAZ files to be processed.
        If a list is provided, the LAS/LAZ files will be merged and then voxelized.
    out_dir : str
        Directory where output files and configurations will be saved.
    voxel_size : float
        Defines the size of each voxel for processing.
    vapc_command : dict
        Dictionary containing Vapc command configurations, including the tool to use
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
        If required keys are missing in the vapc_command dictionary.

    Notes
    -----
    - When tiling is enabled, temporary tiles are created in a subdirectory within out_dir.
    - After processing, temporary files and directories are cleaned up to save space.
    - The configuration used for processing is saved as a JSON file if `doc` is True.
    """
    # get_timestamp_for_output
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H-%M-%S")

    # document settings
    config = vapc_command
    config["vapc_version"] = get_version()
    config["file"] = str(file)
    config["out_dir"] = str(out_dir)
    config["voxel_size"] = voxel_size
    config["reduce_to"] = reduce_to
    config["tile"] = tile
    config["save_as"] = save_as
    # generate filepaths
    outfile = os.path.join(out_dir, vapc_command["tool"] + f"_{timestamp}.las")
    outconfig = os.path.join(out_dir, vapc_command["tool"] + f"_{timestamp}_cfg.json")
    # Use temporary tiling for big datasets
    if tile:
        tile_dir = os.path.join(out_dir, "temp_tiles")
        if not os.path.isdir(tile_dir):
            os.mkdir(tile_dir)
        del_inlas = False
        if isinstance(file, list):
            inlas = os.path.join(out_dir, "merged_las.las")
            las_merge(file, inlas)
            del_inlas = True
        else:
            inlas = file

        all_tiles = las_create_3dtiles(
            lazfile=inlas,
            out_dir=tile_dir,
            tilesize=tile,
            tilename="temptile",
            buffer=(voxel_size / 2),
        )

        for sub_tile in all_tiles:
            if not sub_tile:
                continue
            use_tool(
                vapc_command["tool"],
                sub_tile,
                sub_tile,
                voxel_size=voxel_size,
                args=vapc_command["args"],
                reduce_to=vapc_command["reduce_to"],
            )

        tiles_without_buffer = las_remove_buffer(tile_dir)

        las_merge(filepaths=tiles_without_buffer, outfile=outfile)

        for two in tiles_without_buffer:
            os.remove(two)
        if del_inlas:
            os.remove(inlas)
        os.rmdir(tile_dir)
    # work without tiles
    else:
        use_tool(
            vapc_command["tool"],
            file,
            outfile,
            voxel_size=voxel_size,
            args=vapc_command["args"],
            reduce_to=vapc_command["reduce_to"],
        )

    # save configuration for documentation
    if doc:
        with open(outconfig, "w") as cfg:
            json.dump(config, cfg, indent=4)

    if save_as == ".laz":
        laz_out = outfile.replace(".las", ".laz")
        lasz_to_lasz(outfile, laz_out)
        os.remove(outfile)
        return laz_out
    if save_as == ".las":
        return outfile
    if save_as == ".ply":
        ply_out = outfile.replace(".las", ".ply")
        lasz_to_ply(infile=outfile,
                    outfile=ply_out,
                    voxel_size=voxel_size,
                    shift_to_center=False)
        os.remove(outfile)
        return ply_out


def parse_args():
    """
    Parses command-line arguments for the Vapc command.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Use Vapc.")
    parser.add_argument("config_file", help="Configuration of the Vapc command")
    return parser.parse_args()


def load_config(config_file):
    """
    Loads the Vapc command configuration from a JSON file.

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        A dictionary containing the Vapc command configurations.
    """
    with open(config_file, "r") as file:
        return json.load(file)


def main():
    """
    The main function that orchestrates the Vapc processing workflow.

    This function parses command-line arguments, loads the configuration,
    and executes the Vapc operations on the specified files.

    Raises
    ------
    Exception
        If any step in the processing workflow fails.

    Examples
    --------
    To run Vapc with a configuration file:

    >>> python run.py config.json
    """
    # Get input
    args = parse_args()

    # Load json
    config = load_config(args.config_file)

    # Apply command
    do_vapc_on_files(
        file=config["infile"],
        out_dir=config["outdir"],
        voxel_size=config["voxel_size"],
        vapc_command=config["vapc_command"],
        tile=config["tile"],
        reduce_to=config["reduce_to"],
        save_as=config["save_as"],
    )


if __name__ == "__main__":
    main()
