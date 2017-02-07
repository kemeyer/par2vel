==============
Field
==============

The result of an analysis is a velocity field. This is stored and manupilated
in the field class. This class actually operates on displacements in pixels.
This is the raw data from the image analysis. The class will also be able
to return the field in as velocities in m/s. However, this is not implemented
in the current version at the moment. The current version only works 
with inplane vectores in planar fields, also known as 2C2D fields. This is 
handled with the Field2D class. New classes will be developed later to
hold planer fields with three velocity components (stereoscopic PIV) and 
volumetric fields.

---------------------
Field2D
---------------------
The internal data in this objects consist of several elements:

Camera:
    A field has a camera and thereby know the relation between physical
    (object) koordinates and image coordinates. 
Coordinates:
    The interrogation grid is stored in the object together with 
    interrogation areas (IA). 
Outlier detection:
    The object contains functions used for outlier detection and replacement. 
    It also keeps track of which vectors that were replaced.


