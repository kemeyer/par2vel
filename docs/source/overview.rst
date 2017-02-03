==============
Overview
==============

PIV starts with images - usually grayscale images of particles. You
have probably taken the images with one or more PIV cameras.  You will
need to handle how to read your images into the format that is used in
par2vel: a matrix (numpy array) with real numbers between 0 and 1. The
general library PIL is useful here (use ``import Image``). A simple
function called ``readimage`` is provided, but it may not handle
images from your camera.

Par2vel partly uses object-oriented programming. We use two
important objects:

Camera
    A camera handles the relation between an image and the physical
    space. Basically, this handles the calibration needed for PIV.

Field 
    A field contains the result - usually vectors in a grid. A field
    will basically operates using positions and displacements in
    pixels.  A field also handles common operations like
    validation and replacement of spurious vectors. A field has a
    Camera (calibration) associated and uses this to present results
    in physical space.

The interrogation of images are handled by diffent functions like for
example ``fftdx`` that takes two images and a field as input. The
field will define which areas on the images that are process to find
displacements. The function will return an updated field with the
results. 

A module for generating artificial images is also include. This is
called ``artimage``.
