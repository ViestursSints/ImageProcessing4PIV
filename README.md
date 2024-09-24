# ImageProcessing4PIV
Python code for processing of microscope images for Particle Image Velocimetry

Context for the code is given in: https://arxiv.org/abs/arXiv:2409.13650

The code contains functions that can be used to produce images suited for PIV analysis, from images obtained with PIV software.

The raw images are expected to be in a multi image .tif format. The naming convention used here is:

run[RunN].[ID].[ImageN].tif

where:

RunN: number of the run (experiment) where images were taken, this will be used for subfolders generated by the code

ID: machine generated identifier that is automatically identified; irrelevant for other naming conventions

ImageN: six digit identifier for an image file

Note that Linux file naming conventions are used in the code. To run the code in Windows, all "/" strings need to changed to "\\"
