"""Generate artifical PIV images for par2vel"""
# Copyright Knud Erik Meyer 2017
# Open source software under the terms of the GNU General Public License ver 3

import numpy
import scipy

class ArtImage(object):
    """Generate an artificial PIV image"""
    def __init__(self, camera, particle_image_size=2.0, sheet_thickness=0):
        """ Make defaults for articificial PIV image  """
        from numpy import array, sqrt
        self.cam = camera  # ArtImage has a camera associated
        self.particle_image_size = particle_image_size # diameter in pixels
        if sheet_thickness:
            self.sheet_thickness = sheet_thickness
        else:
            # use thickness correponding to 32 pixel displacement as default
            ni, nj = camera.shape
            x = array([[0.5 * nj], [0.5 * ni]])
            dx = array([[32],[0.0]]) 
            dX = self.cam.dx2dX(x, dx)
            self.sheet_thickness = sqrt((dX**2).sum())
        # make empty lists of particle position array and of images
        self.X = []
        self.Im = []

    def random_particles(self, particle_density):
        """ Create array with random particles at initial position

            particle_density is particle concentration as particles/pixel
        """
        from numpy import empty, array, cross, sqrt
        from numpy.random import rand
        # get physical coordinates of camera image corners
        ni, nj = self.cam.shape
        Xc = self.cam.x2X(array([[0.0, nj,  0, nj],
                                 [0.0,  0, ni, ni]])) # ni is x1 and nj is x0
        # find area in physical space (z=0 plane) as sum of two triangles
        Xc2 = Xc[:2, :]
        area = 0.5 * ( abs(cross(Xc2[:,1] - Xc2[:,0], Xc2[:,2] - Xc2[:,0])) +
                       abs(cross(Xc2[:,1] - Xc2[:,3], Xc2[:,2] - Xc2[:,3])) )
        # increase physcial area to account for particle displacements
        factor = 0.1   # increase by 10% at each side
        deltaX = Xc[:,3] - Xc[:,0]
        Xc0 = Xc[:,0] - factor * deltaX
        Xc0[2] = -self.sheet_thickness  # use 2 * sheet_thickness
        deltaX = (1 + 2 * factor) * deltaX
        deltaX[2] = 2 * self.sheet_thickness
        # get number of particles in this space
        n_particles = ni * nj * particle_density
        npar2 = int(2 * n_particles * (deltaX[0] * deltaX[1] / area))
        # generate random particle positions (origo at center of space)
        # particle array is stored in list as first element
        self.X = [empty((3,npar2),float)]
        self.X[0][0,:] = deltaX[0] * rand(npar2) + Xc0[0]
        self.X[0][1,:] = deltaX[1] * rand(npar2) + Xc0[1]
        self.X[0][2,:] = deltaX[2] * rand(npar2) + Xc0[2]

    def particle_positions(self, X):
        """ Add array of particle positions manually """
        self.X.append(X)

    def displace_particles(self, func, delta_time):
        """ Create array with particles at final position

            func is a function u(t,X) giving the 3D velocity field,
            New particles position are found by integration using delta_time
        """
        from scipy.integrate import ode
        # wrap func in new function that works with flat arrays
        def funcflat(t, X):
            X3 = X.reshape(3, -1)
            U3 = func(t, X3)
            return U3.reshape(1, -1)
        # do integration
        r = ode(funcflat).set_integrator('vode')
        r.set_initial_value(self.X[0].flatten())
        Xresult = r.integrate(delta_time)
        # append result to X list, reshaped to original shape
        self.X.append(numpy.reshape(Xresult,self.X[0].shape))


    def generate_images(self):
        """ Generate images from particle positions """
        from numpy import zeros, ceil, arange, pi, sqrt, dot
        from scipy.special import erf
        ni, nj = self.cam.shape
        for X in self.X:
            # create image and append to list
            Im = zeros(self.cam.shape,float)
            self.Im.append(Im)
            # find images coordinates for particles
            x = self.cam.X2x(X)
            # indices to pixels with particle centers
            xcp = numpy.round_(x).astype(int)
            xrel = x - xcp
            # window index for integration windows
            di = int(ceil(self.particle_image_size))
            xpixrel = arange(-di, di + 0.1)
            # relativ position of pixel edges
            x0pixlow = xpixrel - 0.5 * self.cam.fill_ratio[0]
            x1pixlow = xpixrel - 0.5 * self.cam.fill_ratio[1]
            x0pixhigh = xpixrel + 0.5 * self.cam.fill_ratio[0]
            x1pixhigh = xpixrel + 0.5 * self.cam.fill_ratio[1]
            # constant for use in integration function erf()
            # see documentation for SIG program for details
            sigma = sqrt(2.0) / (pi * 2.44)   # sigma used in gaussian function
            # divide argument in erf with s
            s = sqrt(2.0) * self.particle_image_size * sigma 
            # c=(pi/8.0)*sigma**2             # constant in front of erf
            for npar in range(x.shape[1]):
                # get differences between erf levels at pixel edges
                Derf0=erf((x0pixhigh-xrel[0,npar])/s) - \
                      erf((x0pixlow-xrel[0,npar])/s)
                Derf1=erf((x1pixhigh-xrel[1,npar])/s) - \
                      erf((x1pixlow-xrel[1,npar])/s)
                # reshape to matrix shapes and multiply to get field
                Derf0.shape = (1,-1)
                Derf1.shape = (-1,1)
                I = dot(Derf1, Derf0)
                # add result to image - take image bounds into account
                # i is index in x[1] direction, j is x[0] direction
                # first border of overlapped region on Im
                i_min = min(ni, max(0, xcp[1,npar] - di))
                i_max = min(ni, max(0, xcp[1,npar] + di + 1))
                j_min = min(nj, max(0, xcp[0,npar] - di))
                j_max = min(nj, max(0, xcp[0,npar] + di + 1))
                # then borders on overlapped images on I 
                iI_min = i_min - (xcp[1,npar] - di)
                iI_max = i_max - (xcp[1,npar] - di)
                jI_min = j_min - (xcp[0,npar] - di)
                jI_max = j_max - (xcp[0,npar] - di)
                # add intensity field I to image Im
                Im[i_min:i_max,j_min:j_max] += I[iI_min:iI_max,jI_min:jI_max]
            # normalize picture to max value = 1.0 for a particle
            maxIntensity=(erf(0.5*self.cam.fill_ratio[0]/s)
                           -erf(-0.5*self.cam.fill_ratio[0]/s)) \
                         *(erf(0.5*self.cam.fill_ratio[1]/s)
                           -erf(-0.5*self.cam.fill_ratio[1]/s))
            Im/=maxIntensity+self.cam.noise_mean

            


def constUfunc(Uconst):
    """ Generate velocity function for particle displacement with constant velocity

        In returned Ufunc(t,X): t is time between frames, 
                                X is 3xn matrix (coordinates for n points)
    """
    def Ufunc(t,X):
        U=numpy.zeros(X.shape,float)
        U[0,:]=Uconst[0]
        U[1,:]=Uconst[1]
        if len(Uconst)==3:
            U[2,:]=Uconst[3]
        return U
    return Ufunc
        

def sineUfunc(wavelength,maxU):
    """ Generate velocity function for particle displacement with sine variations

        wavelength is obviously wavelength of variations
        maxU is maximum size of velocity
        In returned Ufunc(t,X): t is time between frames, 
                                X is 3xn matrix (coordinates for n points)
    """
    def Ufunc(t,X):
        U=scipy.numpy(X.shape,float)
        U[0,:]=maxU*scipy.sin(numpy.pi*X[1,:]/wavelength)
        return U
    return Ufunc

def OseenUfunc(size,maxU,Xcenter=[0.,0,0]):
    """ Generate velocity function for particle displacements with Oseen vortex 
    
        This version does not take time-dependence into account.
          size is distance from center (0,0) to edge of viscous core
          maxU is maximum velocity
          Xcenter is vortex center in object space (to be converted to array)
        In returned Ufunc(t,X): t is time between frames, 
                                X is 3xn matrix (coordinates for n points)
    """
    import scipy.optimize
    # find location of maximum velocity (note sign change because we use fmin)
    def f(r): return -(1/r)*(1-scipy.exp(-r**2))
    rmax=scipy.optimize.fmin(f,[1.0],disp=0)
    U_rmax=-f(rmax)
    # define Ufunc
    def Ufunc(t,X):
        U=scipy.zeros(X.shape,float)
        Xc=X-scipy.array(Xcenter).reshape((3,1))
        r=scipy.sqrt(Xc[0,:]**2+Xc[1,:]**2)
        angle=scipy.arctan2(Xc[1,:],Xc[0,:])
        # value of r=2 correspond to edge of viscous core
        Uveclen=(maxU/U_rmax)*(-f(2.0*r/size))
        U[0,:]=Uveclen*scipy.sin(angle)
        U[1,:]=-Uveclen*scipy.cos(angle)
        return U
    return Ufunc
        

