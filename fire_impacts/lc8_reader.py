#!/usr/bin/env python
"""Classes to interpret and pre-process data files for fcc calculations."""

# fire_impacts_s2 A fcc model calculator for S2 & L8
# Copyright (c) 2019 J Gomez-Dans. All rights reserved.
#
# This file is part of fire_impacts.
#
# fire_impacts is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fire_impacts is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with fire_impacts.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "J Gomez-Dans"

import datetime
import logging
import shutil
import tempfile

import numpy as np
from osgeo import gdal
from osgeo import osr


from pathlib import Path

from collections import namedtuple

from .utils import reproject_image

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(module)s."
    + "%(funcName)s - "
    + "- %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


class NoLC8File(IOError):
    def __init__(self, arg):
        self.args = arg


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
        folder gets deleted at the end!
        """
        log.info(f"Processing LC8 file {lc8_file}")
        lc8_file = Path(lc8_file)
        if not lc8_file.exists():
            raise IOError(f"{lc8_file.name} doesn't exist")
        if not lc8_file.match("*LC08*"):
            raise NoLC8File(lc8_file.name)
        if temp_folder is None:
            log.info(f"Persistently uncompressing {lc8_file}")
            target_folder = lc8_file.parent / (lc8_file.stem.split(".")[0])
            self.temp_folder = None
            log.debug(f"Created persistent folder {target_folder.name}")
        else:
            self.temp_folder = tempfile.TemporaryDirectory(dir=temp_folder)
            target_folder = Path(self.temp_folder.name)
            log.debug(f"Created temporary folder {self.temp_folder.name}")
        lc8_data = namedtuple("lc8tuple", "surface_reflectance qa saturation aot")
        if "".join(lc8_file.suffixes) == ".tar.gz":
            self.lc8_files = lc8_data(
                *self._uncompress_to_folder(lc8_file, target_folder, master_file)
            )
        self.surface_reflectance = self.lc8_files.surface_reflectance
        self.mask = self.interpret_qa()
        log.info(
            f"Number of valid pixels: {self.mask.sum()}"
            + f"({100.*self.mask.sum()/np.prod(self.mask.shape):g}%)"
        )

    def interpret_qa(self):
        """Calculates QA mask"""
        log.debug("Calculating mask")
        g = gdal.Open(self.lc8_files.qa.as_posix())
        qa = g.ReadAsArray()
        mask1 = np.in1d(qa, np.array([322, 386, 834, 898, 1346])).reshape(qa.shape)
        g = gdal.Open(self.lc8_files.saturation.as_posix())
        qa = g.ReadAsArray()
        mask2 = qa == 0  # No saturation

        g = gdal.Open(self.lc8_files.aot.as_posix())
        qa = g.ReadAsArray()
        mask3 = np.in1d(
            qa, np.array([2, 66, 130, 194, 32, 96, 100, 160, 164, 224, 228])
        ).reshape(qa.shape)
        return mask1 * mask2  # * mask3

    def _uncompress_to_folder(self, archive, target_folder, reproject_to_master):
        """Uncompresses tarball to temporary folder"""
        if target_folder.exists() and not target_folder.match("tmp*"):
            log.info("Target folder already exists, not unpacking")
        else:
            log.info("Unpacking files...")
            target_folder.mkdir(exist_ok=True)
            shutil.unpack_archive(
                archive.as_posix(), extract_dir=target_folder.as_posix()
            )
        the_files = [x for x in target_folder.glob("*.tif")]
        if reproject_to_master is not None:
            all_files = []
            for the_file in the_files:
                log.info(
                    f"Reprojecting {the_file.name} to"
                    + f" master file {reproject_to_master}"
                )
                fname = reproject_image(the_file.as_posix(), reproject_to_master)
                all_files.append(Path(fname))
            the_files = all_files

        if len(the_files) == 0:
            raise IOError(f"Something weird with {target_folder}")
        surf_reflectance = []
        for the_file in the_files:
            # Ignore the deep blue band...
            if (
                the_file.name.find("sr_band") >= 0
                and the_file.name.find("sr_band1") == -1
            ):
                surf_reflectance.append(the_file.as_posix())
            elif the_file.name.find("pixel_qa") >= 0:
                pixel_qa = the_file
            elif the_file.name.find("radsat_qa") >= 0:
                radsat_qa = the_file
            elif the_file.name.find("sr_aerosol") >= 0:
                aot_qa = the_file
        log.debug("All files found!")
        surf_reflectance_output = target_folder / "LC8_surf_refl.vrt"
        print(sorted(surf_reflectance))
        gdal.BuildVRT(
            surf_reflectance_output.as_posix(),
            sorted(surf_reflectance),
            resolution="highest",
            resampleAlg="near",
            separate=True,
        )
        self.pathrow, acq_time = the_file.stem.split("_")[2:4]
        self.acq_time = datetime.datetime.strptime(acq_time, "%Y%m%d")
        log.debug(f"Path/Row:{self.pathrow}")
        log.debug(f"Acqusition time:{self.acq_time.strftime('%Y-%m-%d')}")
        return surf_reflectance_output, pixel_qa, radsat_qa, aot_qa

    def __del__(self):
        """Remove temporary folder"""
        self.temp_folder = None


if __name__ == "__main__":
    lc8_pref = "../test_data/" + "LC082040322017061501T1-SC20180328085351.tar.gz"
    lc8_postf = "../test_data/" + "LC082040322017070101T1-SC20180328085355.tar.gz"

    # lc8_pref = "/home/ucfajlg/temp/" + \
    #            "LC082040322017061501T1-SC20180328085351.tar.gz"
    # lc8_postf = "/home/ucfajlg/temp/" + \
    #            "LC082040322017070101T1-SC20180328085355.tar.gz"

    lc8_pre = LC8File(lc8_pref, temp_folder=None)
    lc8_post = LC8File(
        lc8_postf, temp_folder=None, master_file=lc8_pre.lc8_files.qa.as_posix()
    )

    # fcc = LC8Fire(lc8_pre, lc8_post)
    # fcc.launch_processor()
