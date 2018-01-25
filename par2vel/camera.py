""" Camera setup and camera models used in par2vel """
# Copyright Knud Erik Meyer 2017
# Open source software under the terms of the GNU General Public License ver 3

import numpy
import scipy
import re
import numbers
from PIL import Image

class Camera(object):
    """Base class for camera models"""

    # valid keywords in camera file - used as variables in the class
    keywords = {'pixels': 'Image size in pixel (two integers: nrow,ncolumn)',
                'pixel_pitch': 'Pixel pitch in meters (two floats, follow x)',
                'fill_ratio': 'Fraction of active pixel area in x ' + \
                              'and y (two floats)',
                'noise_mean': 'Mean noiselevel - fraction (float)',
                'noise_rms':  'RMS noiselevel - fraction (float)',
                'focal_length': 'Focal length in meters (float)',
                'f_number':   'F-number on lens (float)'}

    def __init__(self, data=None):
        """Define a camera, possibly based on optional argument
           Options are:
              - List or array with two integers giving numer of pixels
              - Another Camera object - keyword values are copied
              - String with filename of camera file
        """
        # define camera calibration model (= class)
        self.model = 'base'
        # set some default values (may be overwritten by camera file)
        self.pixel_pitch = (1e-5, 1e-5)  # 10 micron
        self.pixels = (512, 512) 
        self.fill_ratio = (1.0, 1.0)
        self.noise_mean = 0.0
        self.noise_rms = 0.0
        self.focal_length = 0.06
        # modify using optional argument data
        if data==None:
            pass
        elif type(data)==str:
            self.read_camera(data)
        elif isinstance(data, (tuple, list, numpy.ndarray)):
            assert isinstance(data[0], numbers.Integral)
            assert isinstance(data[1], numbers.Integral)
            assert len(data) == 2
            self.pixels = data
        elif isinstance(data, Camera):
            for keyword in self.keywords:
                try:
                    exec('self.' + keyword + '=data.' + keyword)
                except:
                    pass
        else:
            print('Warning: unknown input to camera definition')
        # set shape to pixels
        self.shape = self.pixels
        
        

    def set_physical_size(self):
        """Set a guess on dimensions in physical space"""
        from numpy import array
        # For this base class we just use pixel coordinates directly
        # size of a 1 pixel displacement in physical space
        self.dXpixel = 1.0   
        # width and height of physical region                                       
        # roughly correponding to image
        self.Xsize = array([self.pixels[1], self.pixels[0]])
        # intersection (roughtly) optical axis and physical plane
        self.Xopticalaxis = (self.Xsize - 1) * 0.5 

    def set_keyword(self, line):
        """Set a keyword using a line from the camera file"""
        # comment or blank line - do nothing
        if re.search('^\\s*#',line) or re.search('^\\s*$',line): 
            return
        # remove any trailing \r (files from windows used in linux)
        if line[-1]=='\r':
            line = line[0:-1]
        # find first word, allow leading whitespace
        firstword = re.findall('^\\s*\\w+',line) 
        if firstword:
            if firstword[0] in self.keywords:
                try:
                    exec('self.'+line) # assign as variable to camera
                except:
                    print('Error at the line:')
                    print(line)
                    print('The keyword',firstword[0],'should be:', end=' ')
                    print(self.keywords[firstword[0]])
                    raise Exception('Camera file error')
            elif firstword[0] == 'model':
                pass
            else:
                print('Error at the line:')
                print(line)
                print('The keyword:',firstword[0],'is not known')
                raise Exception('Camera file error')
        else: 
            print('Error at the line:')
            print(line)
            print('Line does not start with keyword or camera model')
            raise Exception('Camera file error')

    def read_camera(self, filename):
        """Read camera definition and/or calibration data"""
        # basic camera only reads camera keywords
        lines = open(filename).readlines()
        for line in lines:
            self.set_keyword(line)
        self.shape = self.pixels
        

    def save_keywords(self, f):
        """Saves defined keywords to an already open camera file f"""
        names = dir(self)
        for keyword in self.keywords:
            if keyword in names:
                #exec('value = self.' + keyword) # get the value
                value = eval('self.' + keyword)
                print(keyword, '=', value, file=f)

    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        # basic camera only saves defined camera keywords
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        self.save_keywords(f)
      
    def set_calibration(self, calib):
        """Set calibration parameters"""
        # actually not needed in base class
        self.calib = calib

    def X2x(self, X):
        """Use camera model to get camera coordinates x
           from physical cooardinates X.
        """
        # base Camera class do not have a camera model
        raise Exception('Base class has no camera model and cannot do X2x')

    def x2X(self, x, z=0):
        """Solve to finde physical coordinate (with X[2]=z) from 
           image coordinate x. 
           In this version z is the same float value for all points.
           Note: This is the default version that is expensive to evaluate.
        """
        from numpy import zeros, vstack
        from scipy.optimize import minimize
        X = zeros((3,x.shape[1]))
        X_guess = zeros((2, 1))
        for i in range(x.shape[1]):
            def func(X2):
                #print('X2.shape',X2.shape)
                Xin = zeros((3,1))
                Xin[0:2,:] = X2.reshape((2,1))
                # Xin = vstack((X2, z))
                return ((self.X2x(Xin)[:,0] - x[:,i])**2).sum()
            res = minimize(func, X_guess)
            X[0 : 2, i] = res.x
        return X

    def dx2dX(self, x, dx, z=0):
        """Transform displacement in pixel to physical displacement.
           The displacement is assumed to be at the z=0 plane in
           physical space.
        """
        # very simple model setting physical coordinates to image coord.
        from numpy import zeros, vstack
        dX = self.x2X(x + 0.5 * dx, z) - self.x2X(x - 0.5 * dx, z)
        return dX      

    def record_image(self, image, ijcenter, pitch):
        """Record an image given in physical space"""
        # ijcenter is image index where optical axis intersects image
        # pitch is pixel pitch of image in meters
        from scipy.interpolate import interp2d
        from numpy import arange, array, indices
        # make x for image sensor
        xtmp = indices(self.shape)
        x = array([xtmp[1,:],xtmp[0,:]], dtype=float)
        # get corresponding coordinates in object space
        X = self.x2X(x.reshape(2,-1))
        # find coordinates corresponding to supplied image
        xim = X[0:2,:] / pitch + [ [ijcenter[1]], [ijcenter[0]] ]
        # find nearest center
        i = xim[1,:].round().astype(int)
        j = xim[0,:].round().astype(int)
        newimage = image[i,j]
        newimage = newimage.reshape(self.shape)
        return newimage  

class One2One(Camera):
    """Camera model that assumes object coordinates = image coordinates"""
    # same functions as Camera, but adds inverse functions x2X and dx2dX
    def __init__(self, data=None):
        Camera.__init__(self, data)
        # define camera calibration model (= class)
        self.model = 'One2One'

    def X2x(self, X):
        """Use camera model to get camera coordinates x
           from physical cooardinates X.
        """
        # this is simple model that assume samme coordinate system
        return X[0:2,:]

    def x2X(self, x, z=0):
        """Find physical coordinates from image coordinates
           We assume that third physical coordinate z=0, but a different
           value can be given as a float.
        """
        # this is simple model that simply assume same coordinate system
        from numpy import ones, vstack
        dummy, n = x.shape
        X = vstack((x, z*ones((1,n))))
        return X

    def dx2dX(self, x, dx, z=0):
        """Transform displacement in pixel to physical displacement.
           The displacement is assumed to be at the z=0 plane in
           physical space.
        """
        # very simple model setting physical coordinates to image coord.
        from numpy import zeros, vstack
        dummy, n = dx.shape
        dX = vstack((dx, zeros((1,n))))
        return dX    

class Linear2d(Camera):
    """Camera model for simple 2D PIV using Z=0 always"""
    def __init__(self, newshape=None):
        Camera.__init__(self,newshape)
        # define camera calibration model (= class)
        self.model = 'Linear2d'

    def set_physical_size(self):
        """Set a guess on dimensions in physical space"""
        from numpy import array, sqrt 
        # intersection (roughtly) optical axis and physical plane
        x_center = (array([self.shape[1], self.shape[0]]) - 1) * 0.5
        x_center.shape = (2, 1)
        self.Xopticalaxis = self.x2X(x_center)
        # width and height of physical region roughly correponding to image
        x0 = array([-0.5, -0.5]).reshape((2,1))
        xmax = array([self.shape[1], self.shape[0]]) + x0
        self.Xsize = abs(self.x2X(xmax) - self.x2X(x0))  #FIX: not right shape
        # size of a 1 pixel displacement in physical space
        self.dXpixel = sqrt(((self.x2X(x_center) -
                              self.x2X(x_center + [[1],[0]]))**2).sum())

    def set_calibration(self, calib):
        """Set calibration parameters"""
        # Calibration is a 2x3 matrix
        assert calib.shape == (2,3)
        self.calib = calib

    def read_camera(self, filename):
        """Read camera definition and/or calibration data"""
        lines = open(filename).readlines()
        nlines = len(lines)
        n = 0
        while n < nlines:
            line = lines[n]
            # check for calibration data
            if line.lower().find('calibration') == 0:
                if line.lower().find('linear2d') > 0:
                    calib = numpy.array([
                        [float(x) for x in lines[n+1].split()],
                        [float(x) for x in lines[n+2].split()] ])
                    self.set_calibration(calib)
                    n += 2                
            else:
                self.set_keyword(line)
            n += 1
        self.shape = self.pixels

    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        # first save defined keywords
        self.save_keywords(f)
        # save calibration
        print('Calibration Linear2d', file=f)
        for row in self.calib:
            for number in row:
                print(number, end=' ', file=f)
            print(file=f)
        f.close()

    def X2x(self, X):
        """Use camera model to get camera coordinates x
           from physical cooardinates X.
        """
        from numpy import vstack, dot, ones
        # append row of ones below first two rows of X
        ni, nj = X.shape
        Xone = vstack((X[0:2,:],ones(nj)))
        # transformation is found by multiplying with calib matrix
        x = dot(self.calib,Xone)
        return x

    def calibrate(self, X, x, print_residual=False):
        """Calibrate using at least 3 data point with points in 
           physical space X linked to corresponding points in 
           camera space x. All points cannot be on the same line.
           
           If X has two components it is assumed to be points in a plane
           If X has three compontents the last components is ignored
        """
        # expand this to take only two points
        # this requires guessing 
        from numpy.linalg import lstsq
        from numpy import ones, vstack, sqrt
        assert X.shape[1] > 2
        assert x.shape[1] > 2
        myX = vstack((X[0:2,:], ones((1, X.shape[1]))))
        res = lstsq(myX.T, x.T)
        calib = res[0].T
        residual = sqrt((res[1]**2).sum())
        if print_residual:
            print('Residual from calibration fit is:', residual)
        self.set_calibration(calib)
        

    def x2X(self, x, z=0):
        """Find physical coordinates from image coordinates
           We assume that third physical coordinate z=0, providing another
           value will have not effect in Linear2D.
        """
        from numpy import vstack, dot, zeros
        from numpy.linalg import inv
        ni, nj = x.shape
        calibinv = inv(self.calib[:,0:2])
        X = dot(calibinv,(x - self.calib[:,2].reshape((-1,1))))
        X = vstack((X, zeros(nj)))
        return X

    def dx2dX(self, x, dx, z=0):
        """Transform displacement in pixel to physical displacement.
           The displacement is assumed to be at the z=0 plane in
           physical space.
        """
        # very simple model setting physical coordinates to image coord.
        from numpy import dot
        from numpy.linalg import inv
        calibinv = inv(self.calib[:,0:2])
        dX = dot(calibinv,dx)
        return dX

class Linear3d(Camera):
    """Camera model using Direct Linear Transform (DFT)"""
    def __init__(self, data=None):
        Camera.__init__(self, data)
        # define camera calibration model (= class)
        self.model = 'Linear3d'

    def set_calibration(self, calib):
        """Set calibration parameters"""
        # Calibration is a 3x4 matrix
        assert calib.shape == (3,4)
        self.calib = calib

    def read_camera(self, filename):
        """Read camera definition and/or calibration data"""
        lines = open(filename).readlines()
        nlines = len(lines)
        n = 0
        while n < nlines:
            line = lines[n]
            # check for calibration data
            if line.lower().find('calibration') == 0:
                if line.lower().find('linear3d') > 0:
                    calib = numpy.array([
                        [float(x) for x in lines[n+1].split()],
                        [float(x) for x in lines[n+2].split()],
                        [float(x) for x in lines[n+3].split()] ])
                    self.set_calibration(calib)
                    n += 3                
            else:
                self.set_keyword(line)
            n += 1
        self.shape = self.pixels

    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        # first save defined keywords
        self.save_keywords(f)
        # save calibration
        print('Calibration Linear3d', file=f)
        for row in self.calib:
            for number in row:
                print(repr(number), end=' ', file=f)
            print(file=f)
        f.close()

    def X2x(self, X):
        """Use camera model to get camera coordinates x
           from physical cooardinates X.
        """
        from numpy import vstack, dot, ones
        # append row of ones below first two rows of X
        ni, nj = X.shape
        Xone = vstack((X,ones(nj)))
        # transformation 
        k = dot(self.calib,Xone)
        # apply perspective correction
        x = k[0:2,:] / k[2,:]
        return x
    
    def calibrate(self, X, x, print_residual=False):
        """Calibrate using at least 12 data point with points in 
           physical space X linked to corresponding points in 
           camera space x. There should be variation in all three
           coordinate directions in X.
        """
        from scipy.optimize import minimize
        from numpy import array
        # guess corresponding to One2One camera
        calib = array([[1.0,   0, 0, 0],
                       [  0, 1.0, 0, 0],
                       [  0,   0, 0, 1]])
        a0 = calib.flatten()
        # make function to be minimized
        def func(a):
            calib = a.reshape((3,4))
            self.set_calibration(calib)
            return ((self.X2x(X) - x)**2).sum()
        # do optimization
        res = minimize(func, a0)
        calib = res.x.reshape((3,4))
        self.set_calibration(calib)
            

class Scheimpflug(Camera):
    """Camera model for simple Scheimpflug camera. 
       This camera is for different tests, but is not meant for use
       with real experiments.
    """
    # this camera assumes the coordinatesystem to have origin on optical axis
    def __init__(self, data=None):
        Camera.__init__(self, data)

    def set_physical_size(self):
        """Set a guess on dimensions in physical space"""
        from numpy import array 
        # intersection (roughtly) optical axis and physical plane
        # - in this case coordinate systems origin in on optical axis
        self.Xopticalaxis = array([[0.0], [0.0], [0.0]])
        # width and height of physical region roughly correponding to image
        size_camerachip = p * array([self.pixels[1], self.pixels[0]])
        self.Xsize = size_camerachip / self.M
        # size of a 1 pixel displacement in physical space
        self.dXpixel = p / M

    def set_calibration(self, theta, M):
        """Set calibration parameters (theta in radians)"""
        # Calibration uses the following parameters
        self.M = M           # magnification
        self.theta = theta   # angle between optical axis and object plane
        # note that focal_length and pixel_pitch must be defined in camera
                             
    def read_camera(self, filename):
        """Read camera definition and/or calibration data"""
        lines = open(filename).readlines()
        nlines = len(lines)
        n = 0
        while n < nlines:
            line = lines[n]
            # check for calibration data
            if line.lower().find('calibration') == 0:
                if line.lower().find('scheimpflug') > 0:
                    fields = lines[n+1].split()
                    self.set_calibration(float(fields[0]), float(fields[1]))
                    n += 1                
            else:
                self.set_keyword(line)
            n += 1
        self.shape = self.pixels
        
    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        # first save defined keywords
        self.save_keywords(f)
        # save calibration
        f.write('Calibration Scheimpflug\n')
        f.write('{:} {:}\n'.format(self.theta, self.M))
        f.close()

    def X2x(self, X):
        """Use camera model to get camera coordinates x
           from physical cooardinates X.
        """
        from numpy import cos, sin, tan, arctan, sqrt, vstack
        # find angles and distances
        theta = self.theta
        alpha = arctan(self.M * tan(theta))
        p = self.pixel_pitch[0] # use pitch in x0 direction
        a = self.focal_length * (self.M + 1) / self.M
        b = self.M * a
        xcenter0 = 0.5 * self.pixels[1] - 0.5   # note model for camera coord. 
        xcenter1 = 0.5 * self.pixels[0] - 0.5
        # find unitvector elements
        uox = sin(theta)
        uoy = cos(theta)
        uix = p * sin(alpha)
        uiy = -p * cos(alpha)
        # arrays with coordinates in physical space
        r = X[0,:]
        s = X[2,:]
        # find corresponding image coordinate
        t = - b * (r * uoy + s * uox) / ( uiy * (a + r * uox - s * uoy)
                                         - uix * (r * uoy + s * uox) )
        # local magnification
        m = sqrt( ((-b + t * uix)**2 + (t * uiy)**2)   /
                  ((a + r * uox - s * uoy)**2 + (r * uoy + s * uox)**2) )
        # make x matrix
        x0 = t + xcenter0
        x1 = -(m * X[1,:] / self.pixel_pitch[1]) + xcenter1
        x = vstack((x0, x1))
        return x

    def x2X(self, x, z=0):
        """Use camera model to get physcial coordinates X from c
           camera coordinates x (assuming X[2]=0, i.e. z must be 0.
        """
        # using solution from PIV book by Raffel et al (2007), page 215
        from numpy import cos, sin, tan, arctan, vstack, zeros, sqrt
        assert z == 0
        ni, nj = x.shape
        # find angles and distances
        theta = self.theta
        alpha = arctan(self.M * tan(theta))
        p = self.pixel_pitch[0] # use pitch in x0 direction
        a = self.focal_length * (self.M + 1) / self.M
        b = self.M * a
        xcenter0 = 0.5 * self.pixels[1] - 0.5   # note model for camera coord. 
        xcenter1 = 0.5 * self.pixels[0] - 0.5
        # find unitvector elements
        uox = sin(theta)
        uoy = cos(theta)
        uix = p * sin(alpha)
        uiy = -p * cos(alpha)
        # arrays with image coordinates (relative to center)
        t = x[0,:] - xcenter0
        # find corresponding object coordinate
        r = t * uiy * a / (-uoy * b + t * uoy * uix - t * uox * uiy)
        # local magnification
        m = sqrt( ((-b + t * uix)**2 + (t * uiy)**2)   /
                  ((a + r * uox)**2 + (r * uoy)**2) )
        # make physical coordinates
        X0 = r
        X1 = -self.pixel_pitch[1] * (x[1,:] - xcenter1) / m
        X2 = zeros((1,nj))
        X = vstack((X0, X1, X2))
        return X

    def dx2dX(self, x, dx, z=0):
        """Transform displacement in pixel to physical displacement.
           The displacement is assumed to be at the z=0 plane in
           physical space (other values cannot be used).
        """
        # very simple model setting physical coordinates to image coord.
        from numpy import zeros, vstack
        assert z == 0
        dX = self.x2X(x + 0.5 * dx) - self.x2X(x - 0.5 * dx)
        return dX      


class Pinhole(Camera):
    """Pinhole model with lens distortion"""
    
    def __init__(self, data=None):
        Camera.__init__(self, data)
        # define camera calibration model (= class)
        self.model = 'Pinhole'

    def calculate_R(self):
        """Calculate rotation matrix from Euler angles"""
        from numpy import array, sin, cos
        # equation from Tsai (1987)
        theta, phi, psi = self.angle
        self.R = array([
            [cos(psi) * cos(theta), sin(psi) * cos(theta), -sin(theta)],
            [-sin(psi) * cos(phi) + cos(psi) * sin(theta) * cos(phi), 
             cos(psi) * cos(phi) + sin(psi) * sin(theta) * sin(phi),
             cos(theta) * sin(phi)],
            [sin(psi) * sin(phi) + cos(psi) * sin(theta) * cos(phi),
             -cos(psi) * sin(phi) + sin(psi) * sin(theta) * cos(phi),
             cos(theta) * cos(phi)]
            ]) 

    def set_calibration(self, angle, T, f, k, x0):
        from numpy import array
        assert len(angle) == 3;
        Tarray = array(T).reshape((3,1))
        assert type(f) == float or type(f) == int
        if type(k) == float or type(k) == int: k = array([k])
        assert len(k) > 0
        assert len(x0) == 2
        self.angle = angle # Euler angles for coordinate transformation
        self.T = Tarray    # Translation vector for coordinate transformation
        self.f = f         # Effective focal length
        self.k = k         # lens distortion coefficients (one or more)
        self.x0 = x0       # pixel coordinates of optical axis
        self.calculate_R()

    def save_camera(self, filename):
        """Save camera definition and/or calibration data"""
        f = open(filename,'w')
        f.write('# par2vel camera file\n')
        f.write("model = '{:}'\n".format(self.model))
        # first save defined keywords
        self.save_keywords(f)
        # save calibration
        print('Calibration Pinhole model', file=f)
        for number in self.angle:
            print(repr(number), end=' ', file=f)
        print(file=f)
        for number in self.T.flatten():
            print(repr(number), end=' ', file=f)
        print(file=f)
        print(repr(self.f), file=f)
        for number in self.k:
            print(repr(number), end=' ', file=f)
        print(file=f)
        for number in self.x0:
            print(repr(number), end=' ', file=f)
        print(file=f)
        f.close()

    
    def read_camera(self, filename):
        """Read camera definition and/or calibration data"""
        from numpy import array
        lines = open(filename).readlines()
        nlines = len(lines)
        n = 0
        while n < nlines:
            line = lines[n]
            # check for calibration data
            if line.lower().find('calibration') == 0:
                if line.lower().find('pinhole') > 0:
                    angle = array([float(x) for x in lines[n+1].split()])
                    T = array([float(x) for x in lines[n+2].split()])
                    f = float(lines[n+3])
                    k = array([float(x) for x in lines[n+4].split()])
                    x0 = array([float(x) for x in lines[n+5].split()])
                    self.set_calibration(angle, T, f, k, x0)
                    n += 5                
            else:
                self.set_keyword(line)
            n += 1
        self.shape = self.pixels

        
    def X2x(self, X):
        """Use camera model to get camera coordinates x
           from physical cooardinates X.
        """
        from numpy import dot, sqrt, array
        # append row of ones below first two rows of X
        Xc = dot(self.R, X) + self.T      # transform to pinhole coordinates
        xd = self.f * Xc[0:2,:] / Xc[2,:] # pinhole model
        r = sqrt(xd[0,:]**2 + xd[1,:]**2)    # radial distance to optical axis
        xu = xd + self.k[0] * r                # first order radial distortion
        for i in range(len(self.k)-1):
            xu += self.k[i + 1] * r**(i+2)      # optional higher order terms
        pixelpitch = array(self.pixel_pitch).reshape((2,1))
        x0 = array(self.x0).reshape((2,1))
        x = xu / pixelpitch + x0
        return x


               
def readimage(filename):
    """ Read grayscale image from file 
        Probably only works with tiff and bmp files 
    """
    im=Image.open(filename)
    s=im.tobytes()
    if im.mode=='L':        # 8 bit image
        gray=numpy.fromstring(s,numpy.uint8)/255.0
    elif im.mode=='I;16':   # 16 bit image (assume 12 bit grayscale)
        gray=numpy.fromstring(s,numpy.uint16)/4095.0
    else:
        raise ImageFormatNotSupported
    gray=numpy.reshape(gray,(im.size[1],im.size[0]))        
    return gray

def saveimage(image,filename):
    """ Save float array (values from 0 to 1) as 8 bit grayscale image """
    imwork=image.copy()
    imwork[imwork<0]=0
    imwork[imwork>1]=1
    im8bit=(255*imwork).astype(numpy.uint8)
    im=Image.frombytes('L',(image.shape[1],image.shape[0]),im8bit.tostring())
    im.save(filename)
