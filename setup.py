#!/usr/bin/env python
"""Setup script for building prosail's python bindings"""
import os
import codecs
import re
from os import path
from setuptools import setup

# Global variables for this extension:
name = "fire_impacts"  # name of the generated python extension (.so)
description = "The fcc spectral fire impacts model"
long_description = "The fcc spectral fire impacts model explains the change" + \
    " in reflectance due to fire as a linear combination of an unburned " + \
        "spectrum and a burned spectrum. This code will fit this model to " + \
        "data from Sentinel 2 and Landsat 8. Differnet versions of the S2 " + \
        "Level 2 product are considered: the Sen2Cor version and the SIAC " + \
        "one."

this_directory = path.abspath(path.dirname(__file__))                                                              

def read(filename):
    with open(os.path.join(this_directory, filename), "rb") as f:
        return f.read().decode("utf-8")


if os.path.exists("README.md"):
    long_description = read("README.md")

def read(*parts):                                                                                                  
    with codecs.open(os.path.join(this_directory, *parts), 'r') as fp:                                             
        return fp.read()                                                                                           
                                                                                                                   
def find_version(*file_paths):                                                                                     
    version_file = read(*file_paths)                                                                               
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",                                               
                              version_file, re.M)                                                                  
    if version_match:                                                                                              
        return version_match.group(1)                                                                              
    raise RuntimeError("Unable to find version string.")

author = "J Gomez-Dans/NCEO & University College London"
author_email = "j.gomez-dans@ucl.ac.uk"
url = "http://github.com/jgomezdans/fire_impacts_s2"
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries :: Python Modules',
    "Topic :: Scientific/Engineering :: Atmospheric Science",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: GIS",
    'Intended Audience :: Science/Research',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Developers',
    'Environment :: Console'
]




setup(
    name=name,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=author,
    url=url,
    author_email=author_email,
    classifiers=classifiers,
    install_requires=[
        "numpy",
        "gdal",
        "numba",
        "scipy",
        "pytest",
    ],
    version=find_version("fire_impacts", "__init__.py"),
    packages=["fire_impacts"],
    entry_points = {
        'console_scripts': ['fire_impacts=fire_impacts.command_line:main'],
    },
    zip_safe=False # Apparently needed for conda
)
