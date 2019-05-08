#!/usr/bin/env python
"""Some functionality to read in Sen2Cor files. In order to deal with the
older format,
"""
import datetime
import logging

import numpy as np
import gdal

from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(module)s." +
                    "%(funcName)s - " +
                    "- %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class NoS2File(IOError):
    def __init__(self, arg):
        self.args = arg


S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]

    

def search_s2_tiles(s2_path, tile):
    s2_path = Path(s2_path)
    granules={}
    for fich in s2_path.glob(f"**/*{tile}*.SAFE"):
        if fich.is_dir():
            granules[datetime.datetime.strptime(
                                                fich.name.split("_")[2],
                                                "%Y%m%dT%H%M%S")] = fich
    return granules


class S2File(object):
    @staticmethod
    def get_file_format_version(granule):
        granuler = Path(granule)
        clouds = [ f for f in granuler.glob("**/cloud.tif")]
        if len(clouds) != 1:
            if granuler.match("*USER*"):
                return S2FileVintageFormat(granule)
            else:
                return S2FileNewFormat(granule)
        else:
            return S2FileSIAC(granule)

class S2FileVintageFormat(object):
    def __init__(self, granule_path):
        if granule_path.find("MSIL2A") < 0:
            raise NoS2File
        log.debug("Dealing with S2 Vintage file")

        self.granule_path = Path(granule_path)
        if not self.granule_path.exists():
            raise IOError(f"{self.granule_path.name} does not exist!")
        self.pathrow = granule_path.split("_")[-2]
        log.info(f"S2 tile code is {self.pathrow}")
        self.acq_time = datetime.datetime.strptime(
                                        self.granule_path.name.split("_")[-4],
                                        "%Y%m%dT%H%M%S")
        log.info(f"S2 tile acquisition date: {self.acq_time}")
        for fich in self.granule_path.glob("**/*_SCL*_20m.jp2"):
            scene_class = fich
        log.debug(f"Found scene class file {scene_class.name}")
        self.mask = self.interpret_qa(scene_class)
        log.info(f"Number of valid pixels: {self.mask.sum()}" +
                 f"({100.*self.mask.sum()/np.prod(self.mask.shape):g}%)")
        img_path = self.granule_path/"IMG_DATA"/"R20m"
        surf_refl = [None for i in range(len(S2_BANDS))]
        for fich in img_path.glob("*_20m.jp2"):
            if not ((fich.as_posix().find("AOT") >= 0) or
                    (fich.as_posix().find("WVP") >= 0) or
                    (fich.as_posix().find("VIS") >= 0)):
                band = S2_BANDS.index(fich.name.split("_")[-2])
                log.debug(f"Band #{S2_BANDS[band]} -> {fich.name}")
                surf_refl[band] = fich.as_posix()

        if any(v is None for v in surf_refl):
            raise IOError("Not all bands were found!")
        else:
            log.debug("Found all surface reflectance files")

        surf_reflectance_output = self.granule_path/"S2_surf_refl.vrt"
        gdal.BuildVRT(surf_reflectance_output.as_posix(),
                      sorted(surf_refl),
                      resolution="highest", resampleAlg="near",
                      separate=True)
        log.debug("Created VRT with surface reflectance files")
        log.info(f"Pre-processed files for {self.granule_path.as_posix()}")
        self.surface_reflectance = surf_reflectance_output

    def interpret_qa(self, scene_class):
        g = gdal.Open(scene_class.as_posix())
        scl = g.ReadAsArray()
        mask = np.in1d(scl, np.array([4, 5, 6])).reshape(scl.shape)
        return mask


class S2FileNewFormat(object):
    def __init__(self, granule_path):
        if granule_path.find("MSIL2A") < 0:
            raise NoS2File
        log.debug("Dealing with S2 New format file")

        self.granule_path = Path(granule_path)
        if not self.granule_path.exists():
            raise IOError(f"{self.granule_path.name} does not exist!")
        self.acq_time = datetime.datetime.strptime(
            granule_path[:granule_path.find(".SAFE")].split(
                "/")[-1].split("_")[2], "%Y%m%dT%H%M%S")
        self.pathrow = granule_path[:granule_path.find(".SAFE")].split(
                        "/")[-1].split("_")[-2]

        log.info(f"S2 tile code is {self.pathrow}")
        log.info(f"S2 tile acquisition date: {self.acq_time}")
        for fich in self.granule_path.glob("**/*_SCL_20m.jp2"):
            scene_class = fich

        log.debug(f"Found scene class file {scene_class.name}")
        self.mask = self.interpret_qa(scene_class)
        log.info(f"Number of valid pixels: {self.mask.sum()}" +
                 f"({100.*self.mask.sum()/np.prod(self.mask.shape):g}%)")
        img_path = scene_class.parent
        surf_refl = [None for i in range(len(S2_BANDS))]
        for fich in img_path.glob("*_20m.jp2"):
            if not ((fich.as_posix().find("AOT") >= 0) or
                    (fich.as_posix().find("WVP") >= 0) or
                    (fich.as_posix().find("VIS") >= 0) or
                    (fich.as_posix().find("SCL") >= 0) or
                    (fich.as_posix().find("TCI") >= 0)):
                if (fich.as_posix().find("B8A") >= 0):
                    band = S2_BANDS.index("B08")
                    log.debug(f"Band #{S2_BANDS[band]} -> {fich.name} HACK!")
                    surf_refl[band] = fich.as_posix()
                else:
                    band = S2_BANDS.index(fich.name.split("_")[-2])
                    log.debug(f"Band #{S2_BANDS[band]} -> {fich.name}")
                    surf_refl[band] = fich.as_posix()

        if any(v is None for v in surf_refl):
            raise IOError("Not all bands were found!")
        else:
            log.debug("Found all surface reflectance files")

        surf_reflectance_output = self.granule_path/"S2_surf_refl.vrt"
        gdal.BuildVRT(surf_reflectance_output.as_posix(),
                      sorted(surf_refl),
                      resolution="highest", resampleAlg="near",
                      separate=True)
        log.debug("Created VRT with surface reflectance files")
        log.info(f"Pre-processed files for {self.granule_path.as_posix()}")
        self.surface_reflectance = surf_reflectance_output

    def interpret_qa(self, scene_class):
        g = gdal.Open(scene_class.as_posix())
        scl = g.ReadAsArray()
        mask = np.in1d(scl, np.array([2, 4, 5, 6, 7, 11])).reshape(scl.shape)
        return mask


class S2FileSIAC(object):
    def __init__(self, granule_path):
        granule_path = Path(granule_path) 
        self.granule_path = granule_path
        if not self.granule_path.exists():
            raise IOError(f"{self.granule_path.name} does not exist!")

        clouds = [ f for f in granule_path.glob("**/cloud.tif")]
        if len(clouds) != 1:
            raise IOError("No SIAC file!")
        log.debug("Dealing with S2 SIAC file")
        cloud_mask = clouds[0] # This should always work
        self.granule_path = Path(granule_path)
        #self.acq_time = datetime.datetime.strptime(
        #    granule_path[:granule_path.find(".SAFE")].split(
        #        "/")[-1].split("_")[2], "%Y%m%dT%H%M%S")
        self.acq_time = datetime.datetime.strptime(
            self.granule_path.name.split("_")[2], "%Y%m%dT%H%M%S")
        self.pathrow = self.granule_path.name.split("_")[5]

        log.info(f"S2 tile code is {self.pathrow}")
        log.info(f"S2 tile acquisition date: {self.acq_time}")
#        for fich in self.granule_path.glob("**/*_SCL_20m.jp2"):
#            scene_class = fich

#        log.debug(f"Found scene class file {scene_class.name}")
        img_path = cloud_mask.parent/Path("IMG_DATA")
        surf_refl = [None for i in range(len(S2_BANDS))]
        for fich in img_path.glob("*_sur.tif"):
            print(fich.name)
            the_band = fich.name.split("_")[-2]
            if the_band in S2_BANDS:
                band = S2_BANDS.index(the_band)
                log.debug(f"Band #{S2_BANDS[band]} -> {fich.name}")
                surf_refl[band] = fich.as_posix()
        if any(v is None for v in surf_refl):
            raise IOError("Not all bands were found!")
        else:
            log.debug("Found all surface reflectance files")

        surf_reflectance_output = self.granule_path/"S2_surf_refl.vrt"
        g = gdal.BuildVRT(surf_reflectance_output.as_posix(),
                      sorted(surf_refl),
                      resolution="highest", resampleAlg="near",
                      separate=True)
        R = gdal.Info(g)
        ul_text = [r for r in R.split("\n") if r.find("Upper Left") >=0]
        lr_text = [r for r in R.split("\n") if r.find("Lower Right") >=0]
        lr_x, lr_y = [float(x.strip(",)")) for x in lr_text[0].split()[3:5]]
        ul_x, ul_y = [float(x.strip(",)")) for x in ul_text[0].split()[3:5]]

        g = gdal.Warp("", cloud_mask.as_posix(),
                      format="MEM", xRes=10, yRes=10,
                      outputBounds=[ul_x, lr_y, lr_x, ul_y])
        
        self.mask = self.interpret_qa(g)
        log.info(f"Number of valid pixels: {self.mask.sum()}" +
                 f"({100.*self.mask.sum()/np.prod(self.mask.shape):g}%)")

        
        log.debug("Created VRT with surface reflectance files")
        log.info(f"Pre-processed files for {self.granule_path.as_posix()}")
        self.surface_reflectance = surf_reflectance_output

    def interpret_qa(self, cld, threshold=10):
        #g = gdal.Open(scene_class.as_posix())
        #cld = g.ReadAsArray()
        
        mask = cld.ReadAsArray() <= threshold
        return mask




if __name__ == "__main__":
    ###granules = search_s2_tiles("/data/selene/ucfajlg/fcc_sentinel2/Alberta/",
                               ###"T11VNC")
    ###files = []
    ###for k, v in granules.items():
        ###files.append(v.as_posix())
    ####    s2_file0 = S2FileVintageFormat(files[0])
    ####   s2_file1 = S2FileVintageFormat(files[1])

    ###granule = "/data/selene/ucfajlg/fcc_sentinel2/Portugal/T29TPE/" + \
        ###"S2A_MSIL2A_20170704T112111_N0205_R037_T29TPE_20170704T112431.SAFE/" +\
        ###"GRANULE/L2A_T29TPE_A010616_20170704T112431"
    ####   s2_new = S2FileNewFormat(granule)

    ###s2_file0 = S2File.get_file_format_version(files[0])
    ###s2_file1 = S2File.get_file_format_version(files[1])
    ###s2_file2 = S2File.get_file_format_version(granule)
    granules = search_s2_tiles("/data/selene/ucfajlg/Chile/", "T18HYF")
    ff = S2FileSIAC(Path("/data/selene/ucfajlg/Chile/S2A_MSIL1C_20170218T143751_N0204_R096_T18HYF_20170218T145150.SAFE"))
