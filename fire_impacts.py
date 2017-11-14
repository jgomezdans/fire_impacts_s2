#!/usr/bin/env python

"""
Fire impacts processor

This code takes two Landsat images from the same location and calculates
spectral measures of fire impact due to the change in reflectance.
The code assumes Landsat family of sensors (in particular LDCM), although
it is designed to be adapted to other sensors (e.g. Sentinel2, ETM+/TM5...)

The code makes use of GDAL to pre-process, read and write files (mostly in
GeoTIFF format), so you must have a working version of the Python GDAL
library (as well as the GDAL binary tools) available.

For more information on the code, get in touch with
Jose Gomez-Dans <j.gomez-dans@ucl.ac.uk>

"""
import argparse
import datetime
import fnmatch
import logging
import os
import subprocess
import sys

import gdal

import numpy as np

__author__ = "J Gomez-Dans (NCEO & UCL)"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "J Gomez-Dans"
__email__ = "j.gomez-dans@ucl.ac.uk"

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

logger = logging.getLogger(__name__)
hdlr = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)


def get_SPOT_mask(self, fname):
    """Calculates the mask from its different components (again, SPOT4/Take5 data
    assumed). Will probably need to be modified for other sensors, and clearly
    we are assuming we have a mask already, in TOA processing, this might be a
    good place to apply a simple cloud/cloud shadow mask or something
    like that."""

    logger.info("Start of process for the mask file associated with %s" %
                fname)
    the_dir = os.path.dirname(fname)
    the_fname = os.path.basename(fname)
    mask_out = os.path.join(
        the_dir, "MASK", the_fname.replace(".xml", "_MASK.TIF"))

    the_mask = os.path.join(
        the_dir, "MASK", the_fname.replace(".xml", "_SAT.tiff"))
    g = gdal.Open(the_mask)
    sat = g.ReadAsArray()
    m3 = sat == 0

    the_mask = mask.replace("SAT", "DIV")

    g = gdal.Open(the_mask)
    div = g.ReadAsArray()
    m1 = div == 0

    the_mask = mask.replace("SAT", "NUA")

    g = gdal.Open(the_mask)
    nua = g.ReadAsArray()
    m2 = np.logical_not(np.bitwise_and(nua, 1).astype(np.bool))

    drv = gdal.GetDriverByName("GTiff")
    dst_ds = drv.Create(mask_out, g.RasterXSize, g.RasterYSize, 1,
                        gdal.GDT_Int16, gdal_options=["COMPRESS=DEFLATE",
                                                      "PREDICTOR=2", "INTERLEAVE=BAND", "TILED=YES",
                                                      "BIGTIFF=YES"])
    dst_ds.SetGeoTransform(g.GetGeoTransform())
    dst_ds.SetProjection(g.GetProjectionRef())
    M = np.array(m1 * m2 * m3).astype(np.int16)
    dst_ds.GetRasterBand(1).WriteArray(M)

    return output_mask


def reproject_image_to_master(master, slave, res=None):

    g = gdal.Open(master)
    geoT = g.GetGeoTransform()
    xmin = geoT[0]
    ymax = geoT[3]
    xmax = geoT[0] + g.RasterXSize * geoT[1]
    ymin = geoT[3] + g.RasterYSize * geoT[-1]
    dst_filename = slave.replace(".tiff", "_crop.vrt")
    subprocess.call(["gdalwarp", "-of", "VRT", "-ot", "Int16",
                     "-r", "near",
                     "-te", "%.3f" % xmin, "%.3f" % ymin, "%.3f" % xmax,
                     "%.3f" % ymax,  slave, dst_filename])
    return dst_filename


def locate_fich(pattern, root=os.getcwd()):
    """Find a file pattern recursively from ``root``"""
    for path, dirs, files in os.walk(root):
        for filename in [os.path.abspath(os.path.join(path, filename))
                         for filename in files
                         if fnmatch.fnmatch(filename, pattern)]:
            yield filename


def check_dates(rho_pre_prefix, rho_post_prefix):
    """A simple method that ensures that the post fire image actually
    was after the pre fire image, as it's easy to get it wrong. It works
    on the prefixes, and assumes that characters 9 to 16 of the prefix
    are actually the data."""

    predate = datetime.datetime.strptime(rho_pre_prefix[9:16], "%Y%j")
    postdate = datetime.datetime.strptime(rho_post_prefix[9:16], "%Y%j")
    logger.info("The prefire file is from %s" %
                predate.strftime("%Y-%B-%d"))
    logger.info("The postfire file is from %s" %
                predate.strftime("%Y-%B-%d"))
    if postdate < predate:
        raise ValueError, "The pre-fire image (%s) is older than the " % \
            rho_pre_prefix + "post-fire image (%s)" % rho_post_prefix


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
    block_size = [block_size[0], block_size[1]]
    logger.info("Blocksize is (%d,%d)" % (block_size[0], block_size[1]))
    #block_size = [ 256, 256 ]
    # store these numbers in variables that may change later
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
    for X in xrange(nx_blocks):
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
        for Y in xrange(ny_blocks):
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


def produce_virtualdataset(prefix, directory, tmpdir="/tmp/"):
    """This function takes the GeoTIFF files from ESPA surface
    reflectance, and creates a virtual dataset with the reflectance
    and the CFmask (other masks are possible, but I've only considered
    CFmask here, as it appears to be the best).
    NOTE that you will beed to have gdalbuildvrt in your path!
    Parameters
    ------------
    prefix: str
        The Landsat scene identifier something that typically looks like
        this: ``LC82040312015257LGN00``. It is used to find related
        GeoTIFF files in the system
    directory: str
        The (parent) directory where the files that start by ``prefix``
        will be searched for.
    tmpdir: str
        A temporary directory. By default, this is ``/tmp/`` in Linux. This
        directory requires the user to have r/w access, and is where the VRTs
        are created

    Returns
    ---------
    The filename of the virtual dataset
    """
    logger.info("Creating VRT with the bands in the right order")
    files = [f for f in locate_fich(
        "%s*.tif" % prefix, root=directory)]
    selected_bands = ["cfmask", "band1", "band2", "band3", "band4", "band5", "band6",
                      "band7"]
    sel_files = []
    for f in files:
        for band in selected_bands:
            if f.find(band) >= 0:
                sel_files.append(f)
                break
    sel_files.sort()
    sel_files.pop(1)  # Remove the cfmask conf band ;-)
    with open(os.path.join(tmpdir, "%s_files.txt" % prefix), 'w') as fp:
        for f in sel_files:
            fp.write("%s\n" % f)
    subprocess.call(["gdalbuildvrt", "-separate", "-input_file_list",
                     os.path.join(tmpdir, "%s_files.txt" % prefix),
                     os.path.join(tmpdir, "%s.vrt" % prefix)])
    subprocess.call(["gdal_translate", "-of", "GTiff", "-ot", "Int16",
                     "-co", "COMPRESS=PACKBITS", "-co", "BIGTIFF=YES",
                     os.path.join(tmpdir, "%s.vrt" % prefix),
                     os.path.join(tmpdir, "%swork.tiff" % prefix)])
    subprocess.call(["gdal_translate", "-of", "VRT", "-ot", "Int16",
                     os.path.join(tmpdir, "%s.vrt" % prefix),
                     os.path.join(tmpdir, "%swork.vrt" % prefix)])

    logger.info("Created output in %s" % os.path.join(tmpdir,
                                                      "%swork.tiff" % prefix))
    #os.remove ( os.path.join ( tmpdir, "%s.vrt" % prefix) )
    #os.remove ( os.path.join ( tmpdir, "%s_files.txt" % prefix) )

    logger.info("Removed temporary VRT %s " % os.path.join(tmpdir,
                                                           "%s.vrt" % prefix))
    return os.path.join(tmpdir, "%swork.tiff" % prefix)


def produce_virtualdataset_SPOT(prefix, directory, tmpdir="/tmp/"):
    """
    What am I doing here? One is to get the mask, using ``get_SPOT_mask``.
    Then, attaching that to the reflectance as band 0. Easiest thing to
    do is to create a VRT from the REFL GeoTIFF, & add the mask.
    """

    fname_mask = get_SPOT_mask(fname)
    g = gdal.Open()
    dst_ds = drv.CreateCopy(os.path.join(tmpdir, "copy.vrt"), g, 0)
    dst_ds.AddBand(gdal.GDT_Int16, ['subClass: VRTRasterBand'])
    md = {}
    xml = '''    <SimpleSource>
      <SourceFilename>%s</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>''' % fname_mask
    md['source_0'] = xml
    dst_ds.GetRasterBand(5).SetMetadata(md, 'vrt_sources')

    drv_out = gdal.GetDriverByName("GTiff")
    dst_ds2 = drv_out.CreateCopy(os.path.join(tmpdir, "copy.tif"), dst_ds, 0)
    del dst_ds2


class FireImpacts (object):

    """A (parent) class to calculate the spectral fcc/a0/a1 model from
    surface reflectance data.
    """

    def __init__(self, rho_pre_prefix, rho_post_prefix, datadir,
                 tmpdir="/home/ucfajlg/test_landsat/tmp/", quantise=False):

        self.save_quantised = quantise
        self.rho_pre_prefix = rho_pre_prefix
        self.rho_post_prefix = rho_post_prefix
        logger.info("Preprocessing pre file %s" % rho_pre_prefix)
        self.rho_pre_file = produce_virtualdataset(rho_pre_prefix,
                                                   datadir, tmpdir=tmpdir)
        logger.info("Preprocessing post file %s" % rho_pre_prefix)

        self.rho_post_file = produce_virtualdataset(rho_post_prefix,
                                                    datadir, tmpdir=tmpdir)
        self.rho_post_file = reproject_image_to_master(self.rho_pre_file,
                                                       self.rho_post_file)

        self._spectral_setup()

    def _spectral_setup(self):
        """A method that sets up the spectral properties of the data. In
        this case, we need to select the centre wavelengths (the example
        here is for LDCM/Landsat8), and the associated per band uncertainties.
        These are hard to get hold of, and you might want to set them to
        ones, in which case the uncertainty quantification in the model
        parameters will make no sense. The idea is that if you want to use
        another sensor, you just derive a class and change this method (and
        maybe methods that allow you to access data).        """
        logger.info("Spectral setup for Landsat8")
        self.bu = np.array([0.004, 0.015, 0.003, 0.004, 0.013,
                            0.010, 0.006])
        self.wavelengths = np.array([440, 480., 560., 655., 865., 1610., 2200])
        self.n_bands = len(self.wavelengths)

        self.lk, self.K = self._setup_spectral_mixture_model()

    def launch_processor(self):
        """A method that process the pre- and post image pairs. All this
        method does is to loop over the input files in a chunk-way manner,
        and return the reflectances for the two images, which are then
        processed in a pixel per pixel way, with the resulting chunk then
        saved out. While this is reasonable when you have a burned area
        product to select the pixels, it might take forever to do on a
        single Landsat tile where all pixels are considered. Efficiency
        gains are very possible, but I haven't explored them here."""

        the_fnames = [self.rho_pre_file, self.rho_post_file]
        first = True
        chunk = 0
        for (ds_config, this_X, this_Y, nx_valid, ny_valid, data) in \
                extract_chunks(the_fnames):
            if first:
                self.create_output(ds_config['proj'], ds_config['geoT'],
                                   ds_config['nx'], ds_config['ny'])
                first = False
            chunk += 1
            logger.info("Doing Chunk %d..." % chunk)
            fcc = -999. * np.ones((ny_valid, nx_valid), dtype=np.float32)
            a0 = -9999. * np.ones((ny_valid, nx_valid), dtype=np.float32)
            a1 = -9999. * np.ones((ny_valid, nx_valid), dtype=np.float32)
            fccunc = -9999. * np.ones((ny_valid, nx_valid), dtype=np.float32)
            a0unc = -9999. * np.ones((ny_valid, nx_valid), dtype=np.float32)
            a1unc = -9999. * np.ones((ny_valid, nx_valid), dtype=np.float32)
            rmse = -1. * np.ones((ny_valid, nx_valid), dtype=np.float32)
            burn = -9999. * np.ones((self.n_bands, ny_valid, nx_valid),
                                    dtype=np.float32)
            fwd = np.zeros((self.n_bands, ny_valid, nx_valid),
                           dtype=np.float32)

            for (j, i) in np.ndindex((nx_valid, ny_valid)):
                # if ( data[0][0, i,j] == 0 ) and ( data[1][0, i,j] == 0 ):
                if (data[0][0, i, j] != 0) or (data[1][0, i, j] != 0):
                    continue

                if np.all(data[0][1:, i, j] > 0) \
                        and np.all(data[1][1:, i, j] > 0) \
                        and np.all(data[0][1:, i, j] < 10000) \
                        and np.all(data[1][1:, i, j] < 10000):
                    try:
                        (xfcc, xa0, xa1, xsBurn, xsFWD, xfccUnc, xa0Unc,
                            xa1Unc, xrmse) = self.invert_spectral_mixture_model(
                            data[0][1:, i, j], data[1][1:, i, j])
                        fcc[i, j] = xfcc
                        a0[i, j] = xa0
                        a1[i, j] = xa1
                        fccunc[i, j] = xfccUnc
                        a0unc[i, j] = xa0Unc
                        a1unc[i, j] = xa1Unc
                        rmse[i, j] = xrmse
                        burn[:, i, j] = xsBurn
                        fwd[:, i, j] = xsFWD
                        rmse[i, j] = xrmse
                    except np.linalg.LinAlgError:
                        fcc[i, j] = -9999
                        a0[i, j] = -9999
                        a1[i, j] = -9999
                        fccunc[i, j] = -9999
                        a0unc[i, j] = -9999
                        a1unc[i, j] = -9999
                        rmse[i, j] = -9999
                        burn[:, i, j] = -9999
                        fwd[:, i, j] = -9999
                        rmse[i, j] = -9999
            # Block has been processed, dump data to files
            logger.info("Chunk %d done! Now saving to disk..." % chunk)
            if self.save_quantised:
                fcc = np.where(np.logical_and(fcc > 1e-3, fcc <= 1.2),
                               fcc * 100, 255).astype(np.uint8)
                a0 = np.where(np.logical_and(a0 > 1e-3, a1 <= 0.2),
                              a0 * 500, 255).astype(np.uint8)
                a1 = np.where(np.logical_and(a1 > 1e-3, a1 <= 0.5),
                              a1 * 400, 255).astype(np.uint8)

            self.ds_params.GetRasterBand(1).WriteArray(
                                                fcc, xoff=this_X, yoff=this_Y)

            self.ds_params.GetRasterBand(2).WriteArray(
                                                a0, xoff=this_X, yoff=this_Y)
            self.ds_params.GetRasterBand(3).WriteArray(
                                                a1, xoff=this_X, yoff=this_Y)
            self.ds_unc.GetRasterBand(1).WriteArray(
                                                fccunc, xoff=this_X,
                                                yoff=this_Y)
            self.ds_unc.GetRasterBand(2).WriteArray(a0unc, xoff=this_X,
                                                    yoff=this_Y)
            self.ds_unc.GetRasterBand(3).WriteArray(a1unc, xoff=this_X,
                                                    yoff=this_Y)
            self.ds_unc.GetRasterBand(4).WriteArray(rmse, xoff=this_X,
                                                    yoff=this_Y)
            [self.ds_burn.GetRasterBand(i + 1).WriteArray(burn[i, :, :],
                                                          xoff=this_X,
                                                          yoff=this_Y)
             for i in xrange(self.n_bands)]
            [self.ds_fwd.GetRasterBand(i + 1).WriteArray(fwd[i, :, :],
                                                         xoff=this_X, yoff=this_Y)
             for i in xrange(self.n_bands)]
            logger.info("Burp!")
        self.ds_burn = None
        self.ds_fwd = None
        self.ds_params = None
        os.remove(self.rho_post_file)
        os.remove(self.rho_pre_file)

    def create_output(self, projection, geotransform, Nx, Ny, fmt="GTiff", suffix="tif",
                      gdal_opts=["COMPRESS=DEFLATE", "PREDICTOR=2",
                                 "INTERLEAVE=BAND", "TILED=YES", "BIGTIFF=YES"]):
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
        logger.info("Created output parameters file %s " % output_fname)
        if self.save_quantised:
            self.ds_params = drv.Create(output_fname, Nx, Ny,
                                        3, gdal.GDT_Byte, options=gdal_opts)
        else:
            self.ds_params = drv.Create(output_fname, Nx, Ny,
                                        3, gdal.GDT_Float32, options=gdal_opts)
        self.ds_params.SetGeoTransform(geotransform)
        self.ds_params.SetProjection(projection)
        logger.info("Created output burn signal file %s " % output_fname)
        output_fname = "%s_%s_burn.%s" % (self.rho_pre_prefix,
                                          self.rho_post_prefix, suffix)
        self.ds_burn = drv.Create(output_fname, Nx, Ny,
                                  self.n_bands, gdal.GDT_Float32, options=gdal_opts)
        self.ds_burn.SetGeoTransform(geotransform)
        self.ds_burn.SetProjection(projection)

        logger.info("Created output uncertainties file %s " % output_fname)
        output_fname = "%s_%s_uncertainties.%s" % (self.rho_pre_prefix,
                                                   self.rho_post_prefix, suffix)
        self.ds_unc = drv.Create(output_fname, Nx, Ny,
                                 4, gdal.GDT_Float32, options=gdal_opts)
        self.ds_unc.SetGeoTransform(geotransform)
        self.ds_unc.SetProjection(projection)

        output_fname = "%s_%s_fwd.%s" % (self.rho_pre_prefix,
                                         self.rho_post_prefix, suffix)
        logger.info("Created output fwd model file %s " % output_fname)
        self.ds_fwd = drv.Create(output_fname, Nx, Ny,
                                 self.n_bands, gdal.GDT_Float32, options=gdal_opts)
        self.ds_fwd.SetGeoTransform(geotransform)
        self.ds_fwd.SetProjection(projection)

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

        K = np.array(np.ones([self.n_bands, 3]))
        # print K[:, 1].shape
        K[:, 1] = lk.transpose() / np.sqrt(self.bu)
        K[:, 0] = K[:, 0] / np.sqrt(self.bu)
        return lk, K

    def invert_spectral_mixture_model(self, rho_pre, rho_post):
        """A method to invert the spectral mixture model using pre and
        post fire reflectances with the same acquisition geometry.
        First, we call :meth:`InvertBurn.SetUpSpectraMixtureModel` to
        set up the spectral domain and the
        main matrix. Then, we calculate the difference between pre and
        post fire reflectance, and solve the linear system of
        equations.
        Additionally, calculate the burn signal, and model post fire
        reflectance, uncertainty etc.

        :returns: :math:`fcc, a_{0}, a_{1}`, burn signal and forward-
        modelled post-burn reflectance.

        """

        rho_post = rho_post * 0.0001
        rho_pre = rho_pre * 0.0001
        K = self.K * 1.
        y = np.array(rho_post - rho_pre) / np.sqrt(self.bu)
        # Difference Post
        # and pre fire rhos
        # K is the matrix with our linear
        K[:, 2] = rho_pre.squeeze() / np.sqrt(self.bu)
        # system of equations (K*x = y)

        sP, residual, rank, singular_vals = np.linalg.lstsq(K, y)
        # Uncertainty
        (fccUnc, a0Unc, a1Unc) = \
            np.linalg.inv(np.dot(K.T, K)).diagonal().squeeze()

        fcc = -sP[2]
        a0 = sP[0] / fcc
        a1 = sP[1] / fcc
        sBurn = a0 + self.lk * a1
        sFWD = rho_pre * (1. - fcc) + fcc * sBurn
        rmse = (sFWD - rho_post).std()
        return (fcc, a0, a1, sBurn, sFWD, fccUnc, a0Unc, a1Unc, rmse)


class FireImpactsTM5 (FireImpacts):

    def __init__(self, rho_pre_prefix, rho_post_prefix, datadir,
                 tmpdir="/home/ucfajlg/test_landsat/tmp/"):

        FireImpacts.__init__(self, rho_pre_prefix,
                             rho_post_prefix, datadir, tmpdir=tmpdir)

    def _spectral_setup(self):
        """A method that sets up the spectral properties of the data. In
        this case, we need to select the centre wavelengths (the example
        here is for LDCM/Landsat8), and the associated per band uncertainties.
        These are hard to get hold of, and you might want to set them to
        ones, in which case the uncertainty quantification in the model
        parameters will make no sense. The idea is that if you want to use
        another sensor, you just derive a class and change this method (and
        maybe methods that allow you to access data).        """
        logger.info("Spectral setup for ETM+")
        self.bu = np.array([0.004, 0.015, 0.003, 0.004, 0.013,
                            0.010, 0.006])
        self.wavelengths = np.array(
            [485.,   560.,   660.,   830.,  1650.,  2215.])

        self.n_bands = len(self.wavelengths)

        self.lk, self.K = self._setup_spectral_mixture_model()


class FireImpactsETM (FireImpacts):

    def __init__(self, rho_pre_prefix, rho_post_prefix, datadir,
                 tmpdir="/home/ucfajlg/test_landsat/tmp/"):

        FireImpacts.__init__(self, rho_pre_prefix,
                             rho_post_prefix, datadir, tmpdir=tmpdir)

    def _spectral_setup(self):
        """A method that sets up the spectral properties of the data. In
        this case, we need to select the centre wavelengths (the example
        here is for LDCM/Landsat8), and the associated per band uncertainties.
        These are hard to get hold of, and you might want to set them to
        ones, in which case the uncertainty quantification in the model
        parameters will make no sense. The idea is that if you want to use
        another sensor, you just derive a class and change this method (and
        maybe methods that allow you to access data).        """
        logger.info("Spectral setup for ETM+")
        self.bu = np.array([0.004, 0.015, 0.003, 0.004, 0.013,
                            0.010, 0.006])
        self.wavelengths = np.array(
            [485.,   560.,   660.,   835.,  1650.,  2220.])

        self.n_bands = len(self.wavelengths)

        self.lk, self.K = self._setup_spectral_mixture_model()


class FireImpactsSPOT (FireImpacts):
    """################################################################################
       #     NOTES
       ################################################################################

        It probably makes sense to stack all the SPOT masks, reflectance and what not
        in a multilayer raster, reproject using VRTs if required, and change the
        chunking loop to interpret the mask there.

        The other option is to take the masks, interpret them and create a Byte dataset
        that can then be added to the reflectance bands.
        """

    def __init__(self, rho_pre_prefix, rho_post_prefix, datadir,
                 tmpdir="/home/ucfajlg/test_landsat/tmp/", quantise=False):

        self.save_quantised = quantise
        self.rho_pre_prefix = rho_pre_prefix
        self.rho_post_prefix = rho_post_prefix
        logger.info("Preprocessing pre file %s" % rho_pre_prefix)
        self.rho_pre_file = produce_virtualdataset_SPOT(rho_pre_prefix,
                                                        datadir, tmpdir=tmpdir)
        logger.info("Preprocessing post file %s" % rho_pre_prefix)

        self.rho_post_file = produce_virtualdataset_SPOT(rho_post_prefix,
                                                         datadir, tmpdir=tmpdir)
        self.rho_post_file = reproject_image_to_master(self.rho_pre_file,
                                                       self.rho_post_file)

        self._spectral_setup()

    def _spectral_setup(self):
        """A method that sets up the spectral properties of the data. In
        this case, we need to select the centre wavelengths (the example
        here is for LDCM/Landsat8), and the associated per band uncertainties.
        These are hard to get hold of, and you might want to set them to
        ones, in which case the uncertainty quantification in the model
        parameters will make no sense. The idea is that if you want to use
        another sensor, you just derive a class and change this method (and
        maybe methods that allow you to access data).        """
        logger.info("Spectral setup for SPOT4")
        self.bu = np.array([0.004, 0.015, 0.003, 0.004])
        self.wavelengths = np.array([570., 645., 835., 1665])

        self.n_bands = len(self.wavelengths)

        self.lk, self.K = self._setup_spectral_mixture_model()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pre_fire_scene", help="The Landsat8 SceneID for the "
                        "pre-fire scene, e.g. LC81050702015188LGN00", type=str)
    parser.add_argument("post_fire_scene", help="The Landsat8 Scene ID for"
                        " the post-fire scene", type=str)
    parser.add_argument("--data", "-d", help="Where to search for the data",
                        default=".", type=str)
    parser.add_argument("--temp", "-t", help="Temporary work directory",
                        default="/tmp/", type=str)
    parser.add_argument("--quantise",  dest="quantise",
                        action="store_true", help="Save parameters quantised")
    parser.add_argument("--no-quantise", dest="quantise",
                        action="store_false",
                        help="Don't save  quantised parameters")
    parser.set_defaults(quantise=False)
    args = parser.parse_args()

    fcc_processor = FireImpacts(args.pre_fire_scene, args.post_fire_scene,
                                args.data, tmpdir=args.temp,
                                quantise=args.quantise)
    fcc_processor.launch_processor()