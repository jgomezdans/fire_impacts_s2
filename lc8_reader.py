#!/usr/bin/env python
"""Classes to interpret and pre-process data files for fcc calculations.
"""
import datetime
import logging
import shutil
import tempfile

import numpy as np
import gdal
import osr

from pathlib import Path 

from collections import namedtuple
# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s " +
                    "- %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Needs to 
# 1. Record angles

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


class LC8File(object):
    """A class to process the LC8 files from the USGS. Basically,
    just unpacks the tarball into a temporary folder, and interprets
    the QA flags. I have used recommended values, but they might not
    be very good and may need revision.
    
    The class provides a `mask`, as well as a set of `lc8_files`."""
    def __init__(self, lc8_file, temp_folder=".", master_file=None):
        """The creator takes a LC8 archive filename (a tarball!), and
        additionally, a folder to create the temporary files (default is
        current folder, but might be e.g. `/tmp/` or '/scratch/`. This
        folder gets deleted at the end!"""
        log.info(f"Processing LC8 file {lc8_file}")
        lc8_file = Path(lc8_file)
        if not lc8_file.exists():
            raise IOError(f"{lc8_file.name} doesn't exist")

        if temp_folder is None:
            log.info(f"Persistently uncompressing {lc8_file}")
            target_folder = (lc8_file.parent/(lc8_file.stem.split(".")[0]))
            self.temp_folder = None
            log.debug(f"Created persistent folder {target_folder.name}")
        else:
            self.temp_folder = tempfile.TemporaryDirectory(dir=temp_folder)
            target_folder = self.temp_folder
            log.debug(f"Created temporary folder {self.temp_folder.name}")
        lc8_data = namedtuple("lc8tuple", 
                                "surface_reflectance qa saturation aot")
        if "".join(lc8_file.suffixes) == ".tar.gz":
            self.lc8_files = lc8_data(*self._uncompress_to_folder(
                                          lc8_file, target_folder, 
                                          master_file))
        self.mask = self.interpret_qa()
        log.info(f"Number of valid pixels: {self.mask.sum()}" + 
                 f"({100.*self.mask.sum()/np.prod(self.mask.shape):g}%)")
            
            
    def interpret_qa(self):
        """Calculates QA mask"""
        log.debug("Calculating mask")
        g = gdal.Open(self.lc8_files.qa.as_posix())
        qa = g.ReadAsArray()
        mask1 = np.in1d(qa, np.array([322, 386, 834, 898, 1346])).reshape(
            qa.shape)
        g = gdal.Open(self.lc8_files.saturation.as_posix())
        qa = g.ReadAsArray()
        mask2 = qa == 0 # No saturation
        
        g = gdal.Open(self.lc8_files.aot.as_posix())
        qa = g.ReadAsArray()
        mask3 = np.in1d(qa, np.array(
                [2, 66, 130, 194, 32, 96, 100, 160, 164, 224, 228])).reshape(
                    qa.shape)
        return mask1 * mask2 * mask3
        
        
    def _uncompress_to_folder(self, archive, target_folder,
                              reproject_to_master):
        """Uncompresses tarball to temporary folder"""
        if target_folder.exists():
            log.info("Target folder already exists, not unpacking")
        else:
            log.info("Unpacking files...")
            target_folder.mkdir(exist_ok=True)
            shutil.unpack_archive(archive.as_posix(),
                              extract_dir=target_folder.as_posix())
        the_files = [ x for x in target_folder.glob("*.tif")]
        if reproject_to_master is not None:
            
            all_files = []
            for the_file in the_files:
                log.info(f"Reprojecting {the_file.name} to master file {reproject_to_master}")
                fname = reproject_image(the_file.as_posix(),
                                        reproject_to_master)
                all_files.append(Path(fname))
            the_files = all_files     
            
        if len(the_files) == 0:
            raise IOError(f"Something weird with {target_folder}")
        surf_reflectance = []
        for the_file in the_files:
            if the_file.name.find("sr_band") >= 0:
                surf_reflectance.append(the_file)
            elif the_file.name.find("pixel_qa") >= 0:
                pixel_qa = the_file
            elif the_file.name.find("radsat_qa") >= 0:
                radsat_qa = the_file
            elif the_file.name.find("sr_aerosol") >= 0:
                aot_qa = the_file
        log.debug("All files found!")
        
        self.pathrow, acq_time = the_file.stem.split("_")[2:4]
        self.acq_time = datetime.datetime.strptime(acq_time, "%Y%m%d")
        log.debug(f"Path/Row:{self.pathrow}")
        log.debug(f"Acqusition time:{self.acq_time.strftime('%Y-%m-%d')}")
        return surf_reflectance, pixel_qa, radsat_qa, aot_qa
    
    def __del__ (self):
        """Remove temporary folder"""
        self.temp_folder = None
        

if __name__ == "__main__":
    lc8_pre = LC8File("./test_data/" + 
                  "LC082040322017061501T1-SC20180328085351.tar.gz", temp_folder=None)
    lc8_post = LC8File("./test_data/" + 
                  "LC082040322017070101T1-SC20180328085355.tar.gz", temp_folder=None,
                  master_file=lc8_pre.lc8_files.qa.as_posix())

