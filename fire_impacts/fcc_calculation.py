#!/usr/bin/env python

"""
A module for fcc calcuations. Two classes: `Observsations` and `FireImpacts`.
`Observations` deals with matching and splicing the S2 or Landsat8 files, 
whereas `FireImpacts` implements the actual per pixel calcluations.
"""

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
import gdal
import osr


from pathlib import Path

from collections import namedtuple

from .lc8_reader import LC8File, NoLC8File

from .s2_reader import S2File, NoS2File, search_s2_tiles

from .utils import reproject_image, extract_chunks
from .utils import invert_spectral_mixture_model
from .utils import invert_spectral_mixture_model_burn_signal


# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(module)s." +
                    "%(funcName)s - " +
                    "- %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class Observations(object):
    def __init__(self, pre_fire, post_fire, temp_folder=None):

        try:
            self.pre_fire = LC8File(pre_fire, temp_folder=temp_folder)
            self.post_fire = LC8File(post_fire,
                                     master_file=
                                     self.pre_fire.surface_reflectance.as_posix(),
                                     temp_folder=temp_folder)
            log.info("Spectral setup for Landsat8")
            self.sensor = "LC8"
            self.wavelengths = np.array([480., 560., 655., 865., 1610., 2200])
            self.bu = np.array([8.5, 5.4, 4.0, 2.6, 1.1, 3.6])/1000.
            self.bu = np.sqrt(self.bu)
            self.n_bands = len(self.wavelengths)
            self.lk, self.K = self._setup_spectral_mixture_model()

        except NoLC8File:
            try:
                self.pre_fire = S2File.get_file_format_version(pre_fire)
                self.post_fire = S2File.get_file_format_version(post_fire)

                log.info("Spectral setup for Sentinel2")
                self.sensor = "S2"
                self.wavelengths = np.array([490., 560., 665., 705, 740., 783,
                                             865., 1610., 2190])
                self.bu = np.ones_like(self.wavelengths)*0.02
                self.bu = np.sqrt(self.bu)
                self.n_bands = len(self.wavelengths)
                self.lk, self.K = self._setup_spectral_mixture_model()
            except NoS2File:
                raise IOError("File wasn't either LC8 or S2/Sen2Cor")

        self.rho_pre_prefix = self.pre_fire.acq_time.strftime("%Y%m%d")
        self.rho_post_prefix = self.post_fire.acq_time.strftime("%Y%m%d")
        assert self.post_fire.acq_time > self.pre_fire.acq_time

    def _setup_spectral_mixture_model(self):
        """This method sets up the wavelength array for the quadratic soil
        function, for a given sensor selected in the class creator

        .. note::
            Needs self.bu, self.n_bands and self.wavelengths defined!

        :return: The (normalised) wavelength values for each band,
        :math:`\\lambda_{i}` and the :math:`\\mathbf{K}` matrix,
        required for inversion.
        """
        # These numbers have been fitted elsewhere, and should better
        # be left floating or as an option. However, I can't be bothered
        # doing that
        loff = 400.
        lmax = 2000.
        ll = self.wavelengths - loff
        llmax = lmax - loff
        lk = (2.0 / llmax) * (ll - ll * ll / (2.0 * llmax))
        ####lk = 2.0*ll - (ll*ll)/(llmax)
        K = np.array(np.ones([self.n_bands, 3]))
        K[:, 1] = lk.transpose() / np.sqrt(self.bu)
        K[:, 0] = K[:, 0] / np.sqrt(self.bu)

        return lk, K




class FireImpacts(object):
    def __init__(self, observations, output_dir=".", quantise=False,
                user_a0=None, user_a1=None):
        if (user_a0 is not None  and user_a1 is None) or (user_a0 is None and
            user_a1 is not None):
            raise ValueError("Either you provide a0 and a1, or I solve for both")
        if user_a0 is not None and user_a1 is not None:
            log.info(f"Fixing a0={float(user_a0):f} and a1={float(user_a1):f}")
        self.user_a0 = user_a0
        self.user_a1 = user_a1
        self.output_dir = output_dir
        self.observations = observations
        self.save_quantised = quantise

    def launch_processor(self):
        """A method that process the pre- and post image pairs. All this
        method does is to loop over the input files in a chunk-way manner,
        and return the reflectances for the two images, which are then
        processed in a pixel per pixel way, with the resulting chunk then
        saved out. While this is reasonable when you have a burned area
        product to select the pixels, it might take forever to do on a
        single Landsat tile where all pixels are considered. Efficiency
        gains are very possible, but I haven't explored them here."""

        mask = self.observations.pre_fire.mask * \
              self.observations.post_fire.mask
        pre_file = self.observations.pre_fire.surface_reflectance.as_posix()
        post_file = self.observations.post_fire.surface_reflectance.as_posix()
        the_fnames = [pre_file, post_file]
        first = True
        chunk = 0

        for (ds_config, this_X, this_Y, nx_valid, ny_valid, data) in \
                extract_chunks(the_fnames):
            if first:
                self.create_output(ds_config['proj'], ds_config['geoT'],
                                   ds_config['nx'], ds_config['ny'])
                first = False
            chunk += 1
            log.info("Doing Chunk %d..." % chunk)
            M = mask[this_Y:(this_Y + ny_valid),
                     this_X:(this_X + nx_valid)]
            if M.sum() == 0:
                log.info("Chunk %d:  All pixels masked..." % chunk)
                continue

            rho_pre = data[0]*0.0001
            rho_post = data[1]*0.0001
            if self.user_a0 is None or self.user_a1 is None:
                xfcc, a0, a1, rmse = invert_spectral_mixture_model(rho_pre,
                                                               rho_post,
                                                               M,
                                                               self.observations.bu,
                                                               self.observations.lk)
            else:
                rho_burn = self.user_a0 + self.user_a1*self.observations.lk 
                
                xfcc, a0, a1, rmse = invert_spectral_mixture_model_burn_signal(
                    rho_burn, rho_pre, rho_post,
                                                               M,
                                                               self.observations.bu,
                                                               self.observations.lk)

            
            # Some information
            fcc_pcntiles = np.percentile(xfcc[M], [5, 95])
            a0_pcntiles = np.percentile(a0[M], [5, 95])
            a1_pcntiles = np.percentile(a1[M], [5, 95])
            log.info(f"\t->Total pixels:{M.sum()}")
            log.info(f"\t->fcc_mean:{xfcc[M].mean():+.2f}, " +
                     f"fcc_sigma:{xfcc[M].std():+.2f}, " +
                     f"5%pcntile:{fcc_pcntiles[0]:+.2f}, " +
                     f"95%pcntile:{fcc_pcntiles[1]:+.2f}")

            log.info(f"\t->a0_mean:{a0[M].mean():+.2f}, " +
                     f"a0_sigma:{a0[M].std():+.2f}, " +
                     f"5%pcntile:{a0_pcntiles[0]:+.2f}, " +
                     f"95%pcntile:{a0_pcntiles[1]:+.2f}")

            log.info(f"\t->a1_mean:{a1[M].mean():+.2f}, " +
                     f"a1_sigma:{a1[M].std():+.2f}, " +
                     f"5%pcntile:{a1_pcntiles[0]:+.2f}, " +
                     f"95%pcntile:{a1_pcntiles[1]:+.2f}")

            # Block has been processed, dump data to files
            log.info(f"Chunk {chunk} done! Now saving to disk...")
            if self.save_quantised:
                xfcc = np.where(np.logical_and(xfcc > 1e-3, xfcc <= 1.2),
                               xfcc * 100, 255).astype(np.uint8)
                a0 = np.where(np.logical_and(a0 > 1e-3, a1 <= 0.2),
                              a0 * 500, 255).astype(np.uint8)
                a1 = np.where(np.logical_and(a1 > 1e-3, a1 <= 0.5),
                              a1 * 400, 255).astype(np.uint8)

            self.ds_params.GetRasterBand(1).WriteArray(
                xfcc, xoff=this_X, yoff=this_Y)
            self.ds_params.GetRasterBand(2).WriteArray(
                a0, xoff=this_X, yoff=this_Y)
            self.ds_params.GetRasterBand(3).WriteArray(
                a1, xoff=this_X, yoff=this_Y)
            self.ds_rmse.GetRasterBand(1).WriteArray(
                rmse, xoff=this_X, yoff=this_Y)
            log.info("Burp!")
        self.ds_burn = None
        self.ds_fwd = None
        self.ds_params = None

    def create_output(self, projection, geotransform, Nx, Ny, fmt="GTiff",
                      suffix="tif",
                      gdal_opts=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES",
                                 "INTERLEAVE=BAND",  "BIGTIFF=YES"]):
        """A method to create the output from the inversion code. By default,
        uses the GeoTIFF format, although this can be changed by changing
        the value of ``fmt`` to another GDAL-friendly format (remember to
        change the file ``suffix`` too!). There are a number of options that
        you can pass to GDAL when creating the dataset. The ones chosen will
        compress the data quite a lot.

        The method then makes available the ``self.ds_params``,
        ``self.ds_burn`` and ``self.ds_fwd`` opened GDAL datasets."""

        drv = gdal.GetDriverByName(fmt)
        output_fname = f"{self.output_dir}/" + \
            f"{self.observations.sensor}_" + \
            f"{self.observations.pre_fire.pathrow}_" + \
            f"{self.observations.rho_pre_prefix}_" + \
            f"{self.observations.rho_post_prefix}_fcc.{suffix}"
        log.debug(f"Creating output parameters file {output_fname}")
        if self.save_quantised:
            self.ds_params = drv.Create(output_fname, Nx, Ny,
                                        3, gdal.GDT_Byte, options=gdal_opts)
        else:
            self.ds_params = drv.Create(output_fname, Nx, Ny,
                                        3, gdal.GDT_Float32, options=gdal_opts)
        self.ds_params.SetGeoTransform(geotransform)
        self.ds_params.SetProjection(projection)
        log.debug("Success!")

        output_fname = f"{self.output_dir}/" + \
            f"{self.observations.sensor}_" + \
            f"{self.observations.pre_fire.pathrow}_" + \
            f"{self.observations.rho_pre_prefix}_" + \
            f"{self.observations.rho_post_prefix}_rmse.{suffix}"

        log.debug("Creating output RMSE signal file %s " % output_fname)
        self.ds_rmse = drv.Create(output_fname, Nx, Ny,
                                  1, gdal.GDT_Float32,
                                  options=gdal_opts)
        self.ds_rmse.SetGeoTransform(geotransform)
        self.ds_rmse.SetProjection(projection)
        log.debug("Success!")
        log.info("Output files successfully created")

if __name__ == "__main__":
    granules = search_s2_tiles("/data/selene/ucfajlg/fcc_sentinel2/Alberta/",
                                   "T12VVH")
    #granules = search_s2_tiles("/data/selene/ucfajlg/fcc_sentinel2/Australia/",
    #                               "T52LFK")
    #granules = search_s2_tiles("/data/selene/ucfajlg/fcc_sentinel2/Colombia/",
    #                               "T18NZL")

    files = []
    for k, v in granules.items():
        files.append(v.as_posix())
        print(k,v)
    #observations_s2 = Observations(files[0], files[1])
    lc8_pref = "../test_data/" + \
                  "LC082040322017061501T1-SC20180328085351.tar.gz"
    lc8_postf = "../test_data/" + \
                    "LC082040322017070101T1-SC20180328085355.tar.gz"
    #observations_lc8 = Observations(lc8_pref, lc8_postf)
    #s2_pref = "/data/selene/ucfajlg/fcc_sentinel2/Portugal/T29TNE/" + \
    #    "S2A_MSIL2A_20170614T112111_N0205_R037_T29TNE_20170614T112422.SAFE/"
    #s2_postf = "/data/selene/ucfajlg/fcc_sentinel2/Portugal/T29TNE/" + \
    #    "S2A_MSIL2A_20170704T112111_N0205_R037_T29TNE_20170704T112431.SAFE"
    s2_pref=files[0]
    s2_postf=files[1]
    fire_imp = FireImpacts(Observations(s2_pref, s2_postf))
    fire_imp.launch_processor()

    #fire_imp = FireImpacts(Observations(lc8_pref, lc8_postf))
    #fire_imp.launch_processor()
