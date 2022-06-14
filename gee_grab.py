import datetime
import ee
import json
import logging
import multiprocessing
import requests
import shutil
from retry import retry
from pathlib import Path

import matplotlib.pyplot as plt
from osgeo import gdal

"""
This tool demonstrates extracting data from Earth Engine using parallel
request and getThumbURL.
"""

ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")


CLOUD_FILTER = 40
CLD_PRB_THRESH = 30
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50
YEAR = 2019


def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER))
    )

    # Import and filter s2cloudless.
    s2_cloudless_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            **{
                "primary": s2_sr_col,
                "secondary": s2_cloudless_col,
                "condition": ee.Filter.equals(
                    **{
                        "leftField": "system:index",
                        "rightField": "system:index",
                    }
                ),
            }
        )
    )


def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get("s2cloudless")).select("probability")

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename("clouds")

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select("SCL").neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = (
        img.select("B8")
        .lt(NIR_DRK_THRESH * SR_BAND_SCALE)
        .multiply(not_water)
        .rename("dark_pixels")
    )

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(
        ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (
        img.select("clouds")
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
        .reproject(**{"crs": img.select(0).projection(), "scale": 100})
        .select("distance")
        .mask()
        .rename("cloud_transform")
    )

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename("shadows")

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select("cloudmask").Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select("B.*").updateMask(not_cld_shdw)


def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = (
        img_cloud_shadow.select("clouds").add(img_cloud_shadow.select("shadows")).gt(0)
    )

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (
        is_cld_shdw.focal_min(2)
        .focal_max(BUFFER * 2 / 20)
        .reproject(**{"crs": img.select([0]).projection(), "scale": 20})
        .rename("cloudmask")
    )

    # Add the final cloud-shadow mask to the image.
    # return img_cloud_shadow.addBands(is_cld_shdw)
    return img.addBands(is_cld_shdw)


# @retry(tries=10, delay=1, backoff=2)
def getResult(fire_name, geometry, burn_date, buffer=250, folder="s2_data/"):
    """Handle the HTTP requests to download an image."""
    sel_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]
    one_day = datetime.timedelta(days=1)
    start_date = burn_date - one_day - datetime.timedelta(days=15)
    end_date = burn_date + datetime.timedelta(days=15)

    # Generate the desired image from the given point.
    geometry = ee.Geometry(geometry)
    # Buffer a couple of kms around the burnscar
    region = geometry.buffer(2000).bounds()

    # Build collection of S2 SR images.
    s2_sr_cld_col = get_s2_sr_cld_col(region, start_date, end_date)
    # we will loop the pre and post-fire periods next
    files = []
    for label, (s, e) in zip(
        ["prefire", "postfire"],
        [[start_date, burn_date], [burn_date + one_day, end_date]],
    ):
        # filter the collection to the pre and post fire period
        collection = s2_sr_cld_col.filterDate(s, e)
        # Do a median reflectance composite before/after fire
        s2_sr_median = (
            collection.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median()
        )
        url = s2_sr_median.getDownloadURL(
            {
                "bands": sel_bands,
                "region": region,
                "scale": 20,
                "filePerBand": False,
                "format": "GEO_TIFF",
            }
        )

        # Handle downloading the actual pixels.
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            r.raise_for_status()

        filename = f"{fire_name}_{label}.tif"
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / filename, "wb") as out_file:
            shutil.copyfileobj(r.raw, out_file)
        files.append(folder / filename)
    return files


if __name__ == "__main__":

    # Let's open the geojson file with some fires
    with open("data/Historic_Perimeters_2019.geojson", "r") as fp:
        db = json.load(fp)

    # Loop through the fires and download the images
    print(f"I have a total of {len(db['features'])} fires")

    # Note that I only download the first 10 fires [:10]
    fires = [x for x in db["features"][:10]]
    geometries = [x["geometry"] for x in db["features"][:10]]
    # I am not sure whether the datecurrent field relates to the fire data, so
    # subract 10 days? Needs checking
    burn_dates = [
        datetime.datetime.strptime(x["properties"]["datecurrent"], "%Y-%m-%dT%H:%M:%SZ")
        - datetime.timedelta(days=10)
        for x in db["features"][:10]
    ]
    fire_names = [x["properties"]["uniquefireidentifier"] for x in db["features"][:10]]

    # getResult(label, geometry, burn_date, buffer=250)
    pool = multiprocessing.Pool(25)
    filenames = pool.starmap(getResult, zip(fire_names, geometries, burn_dates))
    print("Done downloading!")
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10, 10))

    for i, (fire_name, filename) in enumerate(zip(fire_names, filenames)):
        pre, post = filename
        g = gdal.Open(pre.as_posix())
        data_pre = g.ReadAsArray()
        g = gdal.Open(post.as_posix())
        data_post = g.ReadAsArray()
        axs[i, 0].imshow(3 * data_pre[[2, 1, 0], :, :].transpose((1, 2, 0)) / 10000)
        axs[i, 1].imshow(3 * data_post[[2, 1, 0], :, :].transpose((1, 2, 0)) / 10000)
        axs[i, 0].set_title(f"{fire_name} prefire", fontsize=8)
        axs[i, 1].set_title(f"{fire_name} postfire", fontsize=8)
