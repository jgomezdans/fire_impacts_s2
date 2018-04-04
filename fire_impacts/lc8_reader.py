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

from utils import GDAL2NUMPY, invert_spectral_mixture_model
from utils import extract_chunks, reproject_image
# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(module)s." +
                    "%(funcName)s - " +
                    "- %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

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

        if temp_folder is None:
            log.info(f"Persistently uncompressing {lc8_file}")
            target_folder = (lc8_file.parent/(lc8_file.stem.split(".")[0]))
            self.temp_folder = None
            log.debug(f"Created persistent folder {target_folder.name}")
        else:
            self.temp_folder = tempfile.TemporaryDirectory(dir=temp_folder)
            target_folder = Path(self.temp_folder.name)
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
        mask2 = qa == 0  # No saturation

        g = gdal.Open(self.lc8_files.aot.as_posix())
        qa = g.ReadAsArray()
        mask3 = np.in1d(qa, np.array(
                [2, 66, 130, 194, 32, 96, 100, 160, 164, 224, 228])).reshape(
                    qa.shape)
        return mask1 * mask2 * mask3

    def _uncompress_to_folder(self, archive, target_folder,
                              reproject_to_master):
        """Uncompresses tarball to temporary folder"""
        if target_folder.exists() and not target_folder.match("tmp*"):
            log.info("Target folder already exists, not unpacking")
        else:
            log.info("Unpacking files...")
            target_folder.mkdir(exist_ok=True)
            shutil.unpack_archive(archive.as_posix(),
                                  extract_dir=target_folder.as_posix())
        the_files = [x for x in target_folder.glob("*.tif")]
        if reproject_to_master is not None:
            all_files = []
            for the_file in the_files:
                log.info(f"Reprojecting {the_file.name} to" +
                         f" master file {reproject_to_master}")
                fname = reproject_image(the_file.as_posix(),
                                        reproject_to_master)
                all_files.append(Path(fname))
            the_files = all_files

        if len(the_files) == 0:
            raise IOError(f"Something weird with {target_folder}")
        surf_reflectance = []
        for the_file in the_files:
            # Ignore the deep blue band...
            if the_file.name.find("sr_band") >= 0 and \
                    the_file.name.find("sr_band1") == -1:
                surf_reflectance.append(the_file.as_posix())
            elif the_file.name.find("pixel_qa") >= 0:
                pixel_qa = the_file
            elif the_file.name.find("radsat_qa") >= 0:
                radsat_qa = the_file
            elif the_file.name.find("sr_aerosol") >= 0:
                aot_qa = the_file
        log.debug("All files found!")
        surf_reflectance_output = target_folder/"LC8_surf_refl.vrt"
        gdal.BuildVRT(surf_reflectance_output.as_posix(),
                      sorted(surf_reflectance),
                      resolution="highest", resampleAlg="near",
                      separate=True)
        self.pathrow, acq_time = the_file.stem.split("_")[2:4]
        self.acq_time = datetime.datetime.strptime(acq_time, "%Y%m%d")
        log.debug(f"Path/Row:{self.pathrow}")
        log.debug(f"Acqusition time:{self.acq_time.strftime('%Y-%m-%d')}")
        return surf_reflectance_output, pixel_qa, radsat_qa, aot_qa

    def __del__(self):
        """Remove temporary folder"""
        self.temp_folder = None

class LC8Fire(object):
    def __init__(self, pre_fire, post_fire, temp_folder=None, quantise=False):
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
    #lc8_pref = "./test_data/" + \
                  #"LC082040322017061501T1-SC20180328085351.tar.gz"
    #lc8_postf = "./test_data/" + \
                    #"LC082040322017070101T1-SC20180328085355.tar.gz"

    lc8_pref = "/home/ucfajlg/temp/" + \
                "LC082040322017061501T1-SC20180328085351.tar.gz"
    lc8_postf = "/home/ucfajlg/temp/" + \
                "LC082040322017070101T1-SC20180328085355.tar.gz"

    lc8_pre = LC8File(lc8_pref, temp_folder=None)
    lc8_post = LC8File(lc8_postf, temp_folder=None,
                  master_file=lc8_pre.lc8_files.qa.as_posix())

    #fcc = LC8Fire(lc8_pre, lc8_post)
    #fcc.launch_processor()
