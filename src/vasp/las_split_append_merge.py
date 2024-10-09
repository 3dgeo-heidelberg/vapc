import laspy
import numpy as np
import os
from functools import partial
from multiprocessing import Pool

def las_create_or_append(fh, las, mask, tile_name):
    if os.path.exists(tile_name):
        # If the outfile exists, append points to it
        with laspy.open(tile_name, "a") as lf:
            app_p = las.points[mask]
            lf.append_points(app_p)
            print("Appended points to:\t", tile_name)
    else:
        header = laspy.LasHeader(
            point_format=fh.header.point_format, version=fh.header.version
        )
        header.offsets = fh.header.offsets
        header.scales = fh.header.scales
        lasTile = laspy.LasData(header)
        if las.points.x[mask].shape[0] == 0:
            return False
        lasTile.points = las.points[mask]
        lasTile.write(tile_name)
        print("Created file:\t\t", tile_name)
    return tile_name

def las_merge(filepaths,
              outfile):
    print(filepaths)
    with laspy.open(filepaths[0],"r") as lf_0:
        las = lf_0.read()
        las.write(outfile)
    with laspy.open(outfile,"a") as lf:
        scales = lf.header.scales
        offsets = lf.header.offsets
        for file in filepaths[1:]:
            with laspy.open(file) as lf_a:
                lf_aa = lf_a.read()
                lf_aa.X = (lf_aa.x - offsets[0]) / scales[0]
                lf_aa.Y = (lf_aa.y - offsets[1]) / scales[1]
                lf_aa.Z = (lf_aa.z - offsets[2]) / scales[2]
                lf.append_points(lf_aa.points)


def laSZ_to_laSZ(lasfile,lazfile):
    with laspy.open(lasfile) as lasf:
        las = lasf.read()
        las.write(lazfile)

def las_create_3DTiles(lazfile,
    outDir,
    tilesize,
    tilename = "",
    buffer = 0,
    chunk_wise = False):
    """ 
    reads a laz File. 
    calculates the boundaries of the tiles that are defined by the tilesize. Tiles start from the lowest x and y value.
    Buffer can be applied.
    tilename is an additional prefix for reference.
    """
    with laspy.open(lazfile,"r") as fh:
        las = fh.read()
        extent = [*fh.header.min, *fh.header.max]
        x_min = np.floor(extent[0])
        x_max = np.ceil(extent[3])
        y_min = np.floor(extent[1])
        y_max = np.ceil(extent[4])
        z_min = np.floor(extent[2])
        z_max = np.ceil(extent[5])
        if x_min == 0 and x_max == 0 and y_min == 0 and y_max == 0 and z_min == 0 and z_max == 0:
            assert False, "extent not defined in header of las/laz file"
    #Find number of Tiles for x and y direction
    diffX = x_max - x_min
    diffY = y_max - y_min
    diffZ = z_max - z_min
    xTiles = int(np.ceil(diffX / tilesize))
    yTiles = int(np.ceil(diffY / tilesize))
    zTiles = int(np.ceil(diffZ / tilesize))

    # Define X,Y Boundaries for the tiles
    xVerts = np.arange(x_min, x_min + xTiles * tilesize + 1, tilesize)
    yVerts = np.arange(y_min, y_min + yTiles * tilesize + 1, tilesize)
    zVerts = np.arange(z_min, z_min + zTiles * tilesize + 1, tilesize)
    tile_names = []
    masks = []
    for xTile in range(xTiles):
        tileMinX = round(xVerts[xTile],3)
        tileMaxX = round(xVerts[xTile + 1],3)
        for yTile in range(yTiles):
            tileMinY = round(yVerts[yTile],3)
            tileMaxY = round(yVerts[yTile + 1],3)
            for zTile in range(zTiles):
                tileMinZ = round(zVerts[zTile],3)
                tileMaxZ = round(zVerts[zTile + 1],3)
                outTile = os.path.join(outDir, f"{tilename}_{tileMinX}_{tileMinY}_{tileMinZ}_{tilesize}_tileSize_{buffer}_buff.las")
                mMinX = las.points.x >= (tileMinX - buffer)
                mMaxX = las.points.x < (tileMaxX + buffer)
                mMinY = las.points.y >= (tileMinY - buffer)
                mMaxY = las.points.y < (tileMaxY + buffer)
                mMinZ = las.points.z >= (tileMinZ - buffer)
                mMaxZ = las.points.z < (tileMaxZ + buffer)
                mask_all = mMinX & mMaxX & mMinY & mMaxY & mMinZ & mMaxZ
                # outfile_mask_dir[outTile] = mask_all

                tile_names.append(outTile)
                masks.append(mask_all)
    print("Created masks")
    write_tile_for_lasfile = partial(las_create_or_append, fh, las)

    #Clip point clouds
    existing_tiles = []
    for mask, tile_name in zip(masks,tile_names):
        existing_tiles.append(write_tile_for_lasfile(mask,tile_name))
    return np.unique(list(filter((False).__ne__, existing_tiles))) #removing false entries

def clip_to_bbox(laz_in,
                 laz_out,
                 bbox):
    with laspy.open(laz_in, "r") as fh:
        las = fh.read()

    xmi = las.points.x >= bbox[0]
    xma = las.points.x < bbox[3]

    ymi = las.points.y >= bbox[1]
    yma = las.points.y < bbox[4]

    zmi = las.points.z >= bbox[2]
    zma = las.points.z < bbox[5]
    points_masked = las.points[xmi & xma & ymi & yma & zmi & zma]

    if points_masked.X.shape[0] == 0:
        print("File empty after removing buffer: %s"%laz_in)
        os.remove(laz_in)
        return 0
    ct = points_masked.X.shape[0]
    header = laspy.LasHeader(point_format=fh.header.point_format, version=fh.header.version)
    header.offsets = fh.header.offsets
    header.scales = fh.header.scales

    lasTile = laspy.LasData(header)
    lasTile.points = points_masked
    lasTile.write(laz_out)
    return ct

def las_remove_buffer(folder):
    ofs = []
    for file in os.listdir(folder):
        bbox_str = file.split("_")[1:4]
        tilesize = float(file.split("_")[4])
        x_min, y_min, z_min, x_max, y_max, z_max = float(bbox_str[0]),float(bbox_str[1]),float(bbox_str[2]),float(bbox_str[0])+tilesize,float(bbox_str[1])+tilesize,float(bbox_str[2])+tilesize
        bbox = np.round([x_min, y_min, z_min, x_max, y_max, z_max],4)
        in_file = os.path.join(folder,file)
        out_file = in_file
        clip_to_bbox(in_file,out_file,bbox)
        if os.path.isfile(out_file):
            ofs.append(out_file)
    return ofs