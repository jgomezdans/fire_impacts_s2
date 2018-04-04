#!/usr/bin/env python
"""Some useful functions
"""

import logging

import gdal

from numba import jit

import numpy as np

import osr

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(module)s." +
                    "%(funcName)s - " +
                    "- %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Needs to
# 1. Record angles
GDAL2NUMPY = {gdal.GDT_Byte:   np.uint8,
              gdal.GDT_UInt16:   np.uint16,
              gdal.GDT_Int16:   np.int16,
              gdal.GDT_UInt32:   np.uint32,
              gdal.GDT_Int32:   np.int32,
              gdal.GDT_Float32:   np.float32,
              gdal.GDT_Float64:   np.float64,
              gdal.GDT_CInt16:   np.complex64,
              gdal.GDT_CInt32:   np.complex64,
              gdal.GDT_CFloat32:   np.complex64,
              gdal.GDT_CFloat64:   np.complex128
              }


def extract_chunks(the_files, the_bands=None):
    """A function that extracts chunks from datafiles"""
    ds_config = {}
    gdal_ptrs = []
    datatypes = []
    for the_file in the_files:
        g = gdal.Open(the_file)
        gdal_ptrs.append(gdal.Open(the_file))
        datatypes.append(GDAL2NUMPY[g.GetRasterBand(1).DataType])

    block_size = g.GetRasterBand(1).GetBlockSize()
    nx = g.RasterXSize
    ny = g.RasterYSize
    if the_bands is None:
        the_bands = np.arange(g.RasterCount) + 1
    proj = g.GetProjectionRef()
    geoT = g.GetGeoTransform()
    ds_config['nx'] = nx
    ds_config['ny'] = ny
    ds_config['nb'] = g.RasterCount
    ds_config['geoT'] = geoT
    ds_config['proj'] = proj
    block_size = [block_size[0]*10, block_size[1]*10]
    log.info("Blocksize is (%d,%d)" % (block_size[0], block_size[1]))
    #  block_size = [ 256, 256 ]
    #  store these numbers in variables that may change later
    nx_valid = block_size[0]
    ny_valid = block_size[1]
    # find total x and y blocks to be read
    nx_blocks = (int)((nx + block_size[0] - 1) / block_size[0])
    ny_blocks = (int)((ny + block_size[1] - 1) / block_size[1])
    buf_size = block_size[0] * block_size[1]
    ################################################################
    # start looping through blocks of data
    ################################################################
    # loop through X-lines
    for X in range(nx_blocks):
        # change the block size of the final piece
        if X == nx_blocks - 1:
            nx_valid = nx - X * block_size[0]
            buf_size = nx_valid * ny_valid

        # find X offset
        this_X = X * block_size[0]

        # reset buffer size for start of Y loop
        ny_valid = block_size[1]
        buf_size = nx_valid * ny_valid

        # loop through Y lines
        for Y in range(ny_blocks):
            # change the block size of the final piece
            if Y == ny_blocks - 1:
                ny_valid = ny - Y * block_size[1]
                buf_size = nx_valid * ny_valid

            # find Y offset
            this_Y = Y * block_size[1]
            data_in = []
            for ig, ptr in enumerate(gdal_ptrs):
                buf = ptr.ReadRaster(this_X, this_Y, nx_valid, ny_valid,
                                     buf_xsize=nx_valid, buf_ysize=ny_valid,
                                     band_list=the_bands)
                a = np.frombuffer(buf, dtype=datatypes[ig])
                data_in.append(a.reshape((
                    len(the_bands), ny_valid, nx_valid)).squeeze())

            yield (ds_config, this_X, this_Y, nx_valid, ny_valid,
                   data_in)


@jit(nopython=True)
def invert_spectral_mixture_model(rho_pre, rho_post, mask, w):
    """A method to invert the spectral mixture model using pre and
    post fire reflectances with the same acquisition geometry.

    """
    n_bands, ny, nx = rho_pre.shape
    fcc = np.zeros((ny, nx), dtype=np.float32)
    a0 = np.zeros((ny, nx), dtype=np.float32)
    a1 = np.zeros((ny, nx), dtype=np.float32)
    rmse = np.zeros((ny, nx), dtype=np.float32)

    k = np.ones((n_bands, 3))
    k[:, 1] = w
    for i in range(ny):
        for j in range(nx):
            if mask[i, j]:
                y = rho_post[:, i, j] - rho_pre[:, i, j]
                k[:, 2] = rho_pre[:, i, j]
                # system of equations (K*x = y)
                if np.linalg.cond(k) < 1e6:
                    sP, residual, rank, singular_vals = np.linalg.lstsq(
                        k, y, rcond=-1)
                    fcc[i, j] = -sP[2]
                    a0[i, j] = sP[0] / fcc[i, j]
                    a1[i, j] = sP[1] / fcc[i, j]
                    s_burn = a0[i, j] + W * a1[i, j]
                    sFWD = rho_pre[:, i, j] * (1. - fcc[i, j]) + \
                        fcc[i, j] * s_burn
                    rmse[i, j] = (sFWD - rho_post[:, i, j]).std()
                else:
                    fcc[i, j] = -999
                    a0[i, j] = -999
                    a1[i, j] = -999
                    rmse[i, j] = -999
            else:
                    fcc[i, j] = -900
                    a0[i, j] = -900
                    a1[i, j] = -900
                    rmse[i, j] = -900
    return fcc, a0, a1, rmse


def reproject_image(source_img, target_img, dstSRSs=None, fmt="VRT"):
    """Reprojects/Warps an image to fit exactly another image.
    Additionally, you can set the destination SRS if you want
    to or if it isn't defined in the source image."""
    g = gdal.Open(target_img)
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterXSize, g.RasterYSize
    xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
    xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
    ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
    ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])
    xRes, yRes = abs(geo_t[1]), abs(geo_t[5])
    if dstSRSs is None:
        dstSRS = osr.SpatialReference()
        raster_wkt = g.GetProjection()
        dstSRS.ImportFromWkt(raster_wkt)
    else:
        dstSRS = dstSRSs
    output_fname = source_img.replace(".tif", "_slave.vrt")
    g = gdal.Warp(output_fname, source_img, format=fmt,
                  outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                  dstSRS=dstSRS)
    return output_fname
