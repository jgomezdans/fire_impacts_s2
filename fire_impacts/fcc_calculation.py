import datetime
import logging
import shutil
import tempfile

import numpy as np
import gdal
import osr


from pathlib import Path

from collections import namedtuple

from lc8_reader import LC8File, NoLC8File

from s2_reader import S2File, NoS2File, search_s2_tiles

from utils import reproject_image


# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(module)s." +
                    "%(funcName)s - " +
                    "- %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class Observations(object):
    def __init__(self, pre_fire, post_fire, temp_folder=None):
        try:
            self.pre_fire = LC8File(pre_fire, temp_folder=temp_folder)
            self.post_fire = LC8File(post_fire, master_file=pre_fire,
                                     temp_folder=temp_folder)
            self.rho_pre_prefix = Path(pre_fire).name.split("-")[0]
            self.rho_post_prefix = Path(post_fire).name.split("-")[0]
            log.info("Spectral setup for Landsat8")
            self.bu = np.ones(6)
            self.wavelengths = np.array([480., 560., 655., 865., 1610., 2200])
            self.n_bands = len(self.wavelengths)
            self.lk, self.K = self._setup_spectral_mixture_model()

        except NoLC8File:
            try:
                self.pre_fire = S2File.get_file_format_version(pre_fire)
                self.post_fire = S2File.get_file_format_version(post_fire)
                log.info("Spectral setup for Sentinel2")
                self.bu = np.ones(6)
                self.wavelengths = np.array([490., 560., 665., 705, 740., 783,
                                             865., 1610., 2190])
                self.n_bands = len(self.wavelengths)
                self.lk, self.K = self._setup_spectral_mixture_model()
            except NoS2File:
                raise IOError("File wasn't either LC8 or S2/Sen2Cor")

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
        #lk = (2.0 / llmax) * (ll - ll * ll / (2.0 * llmax))
        lk = 2.0*ll - (ll*ll)/(llmax)
        K = np.array(np.ones([self.n_bands, 3]))
        K[:, 1] = lk.transpose() / np.sqrt(self.bu)
        K[:, 0] = K[:, 0] / np.sqrt(self.bu)

        return lk, K




class FireImpacts(object):
    def __init__(self, observations, temp_folder=None, quantise=False):
        self.pre_fire = LC8File(pre_fire, temp_folder=temp_folder)
        self.post_fire = LC8File(post_fire, temp_folder=temp_folder)

        self.rho_pre_prefix = Path(pre_fire).name.split("-")[0]
        self.rho_post_prefix = Path(post_fire).name.split("-")[0]

        assert self.post_fire.acq_time > self.pre_fire.acq_time

        self.save_quantised = quantise

        log.info("Spectral setup for Landsat8")
        self.bu = np.ones(6)
        self.wavelengths = np.array([480., 560., 655., 865., 1610., 2200])
        self.n_bands = len(self.wavelengths)

        self.lk, self.K = self._setup_spectral_mixture_model()

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
        #lk = (2.0 / llmax) * (ll - ll * ll / (2.0 * llmax))
        lk = 2.0*ll - (ll*ll)/(llmax)
        K = np.array(np.ones([self.n_bands, 3]))
        K[:, 1] = lk.transpose() / np.sqrt(self.bu)
        K[:, 0] = K[:, 0] / np.sqrt(self.bu)

        return lk, K


    def launch_processor(self):
        """A method that process the pre- and post image pairs. All this
        method does is to loop over the input files in a chunk-way manner,
        and return the reflectances for the two images, which are then
        processed in a pixel per pixel way, with the resulting chunk then
        saved out. While this is reasonable when you have a burned area
        product to select the pixels, it might take forever to do on a
        single Landsat tile where all pixels are considered. Efficiency
        gains are very possible, but I haven't explored them here."""


        mask = self.pre_fire.mask * self.post_fire.mask

        the_fnames = [self.pre_fire.lc8_files.surface_reflectance.as_posix(),
                      self.post_fire.lc8_files.surface_reflectance.as_posix()]
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
            if np.sum( mask[this_Y:(this_Y + ny_valid),
                            this_X:(this_X + nx_valid)] ) == 0:
                log.info("Chunk %d:  All pixels masked..." % chunk)
                continue
            M = mask[this_Y:(this_Y + ny_valid),
                     this_X:(this_X + nx_valid)]
            rho_pre = data[0]*0.0001
            rho_post = data[1]*0.0001
            xfcc, a0, a1, rmse = invert_spectral_mixture_model(rho_pre,
                                                                rho_post,
                                                                M, self.lk)



            # Block has been processed, dump data to files
            log.info("Chunk %d done! Now saving to disk..." % chunk)
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

            #####self.ds_unc.GetRasterBand(1).WriteArray(
                #####fccunc, xoff=this_X,
                #####yoff=this_Y)
            #####self.ds_unc.GetRasterBand(2).WriteArray(a0unc, xoff=this_X,
                                                    #####yoff=this_Y)
            #####self.ds_unc.GetRasterBand(3).WriteArray(a1unc, xoff=this_X,
                                                    #####yoff=this_Y)
            #####self.ds_unc.GetRasterBand(4).WriteArray(rmse, xoff=this_X,
                                                    #####yoff=this_Y)
            #####[self.ds_burn.GetRasterBand(i + 1).WriteArray(burn[i, :, :],
                                                          #####xoff=this_X,
                                                          #####yoff=this_Y)
             #####for i in xrange(self.n_bands)]
            #####[self.ds_fwd.GetRasterBand(i + 1).WriteArray(fwd[i, :, :],
                                                         #####xoff=this_X,
                                                         #####yoff=this_Y)
             #####for i in xrange(self.n_bands)]
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
        output_fname = "%s_%s_fcc.%s" % (self.rho_pre_prefix,
                                         self.rho_post_prefix, suffix)
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
        log.debug("Creatingoutput RMSE signal file %s " % output_fname)
        output_fname = "%s_%s_rmse.%s" % (self.rho_pre_prefix,
                                          self.rho_post_prefix, suffix)
        self.ds_rmse = drv.Create(output_fname, Nx, Ny,
                                  1, gdal.GDT_Float32,
                                  options=gdal_opts)
        self.ds_rmse.SetGeoTransform(geotransform)
        self.ds_rmse.SetProjection(projection)
        log.debug("Success!")
        log.info("Output files successfully created")
        ####logger.info("Created output burn signal file %s " % output_fname)
        ####output_fname = "%s_%s_burn.%s" % (self.rho_pre_prefix,
                                            ####self.rho_post_prefix, suffix)
        ####self.ds_burn = drv.Create(output_fname, Nx, Ny,
                                    ####self.n_bands, gdal.GDT_Float32,
                                    ####options=gdal_opts)
        ####self.ds_burn.SetGeoTransform(geotransform)
        ####self.ds_burn.SetProjection(projection)

        ####logger.info("Created output uncertainties file %s " % output_fname)
        ####output_fname = "%s_%s_uncertainties.%s" % (self.rho_pre_prefix,
                                                    ####self.rho_post_prefix,
                                                    ####suffix)
        ####self.ds_unc = drv.Create(output_fname, Nx, Ny,
                                    ####4, gdal.GDT_Float32, options=gdal_opts)
        ####self.ds_unc.SetGeoTransform(geotransform)
        ####self.ds_unc.SetProjection(projection)

        ####output_fname = "%s_%s_fwd.%s" % (self.rho_pre_prefix,
                                            ####self.rho_post_prefix, suffix)
        ####logger.info("Created output fwd model file %s " % output_fname)
        ####self.ds_fwd = drv.Create(output_fname, Nx, Ny,
                                    ####self.n_bands, gdal.GDT_Float32,
                                    ####options=gdal_opts)
        ####self.ds_fwd.SetGeoTransform(geotransform)
        ####self.ds_fwd.SetProjection(projection)

if __name__ == "__main__":
    granules = search_s2_tiles("/data/selene/ucfajlg/fcc_sentinel2/Alberta/",
                                   "T12VVH")
    files = []
    for k, v in granules.items():
        files.append(v.as_posix())
    observations =
