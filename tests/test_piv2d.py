# test of piv2d.py

import unittest
import numpy
from par2vel.artimage import ArtImage, constUfunc
from par2vel.camera import Camera, One2One
from par2vel.field import Field2D
from par2vel import piv2d
from par2vel.piv2d import fftdx, Optimizex0, squarewindow, FindCorr

class TestDisplacmentFFT(unittest.TestCase):

    def testdisplacement(self):
        self.cam = One2One((32,32))
        ai = ArtImage(self.cam)
        dx = 0.3
        X = numpy.array([[15.0],[15],[0]])
        ai.particle_positions(X)
        X2 = X.copy()
        X2[0] += dx
        ai.particle_positions(X2)
        ai.generate_images()
        ifrac, jfrac = piv2d.displacementFFT(ai.Im[0],ai.Im[1])
        self.assertTrue(abs(dx - jfrac) < 0.1)

class TestPIV2D(unittest.TestCase):
    
    def setUp(self):
        self.cam=One2One((64,64))
        AI=ArtImage(self.cam)
        AI.random_particles(0.02)
        self.dx=numpy.array([5.1,2.2])
        AI.displace_particles(constUfunc(self.dx), 1)
        AI.generate_images()
        self.AI=AI
        self.IAg=Field2D(self.cam)
        
    def testfftdxDisplacements(self):
        self.IAg.setx(numpy.array([[31.5],[31.5]]))
        fftdx(self.AI.Im[0],self.AI.Im[1],self.IAg)
        # expected RMS error is 0.1 pixel - so choose 0.3 as limit to be sure
        self.assertTrue(abs(self.IAg.dx.flatten()-self.dx).sum()<0.3)
        
    def testfftdxGrid(self):
        self.IAg.squarewindows(32,0.0)
        fftdx(self.AI.Im[0],self.AI.Im[1],self.IAg)
        self.assertTrue(abs(self.IAg.dx[:,0,0]-self.dx).sum()<0.3)
        fftdx(self.AI.Im[0],self.AI.Im[1],self.IAg)
        self.assertTrue(abs(self.IAg.dx[:,0,0]-self.dx).sum()<0.3)
        self.assertTrue(abs(self.IAg.dx[:,1,1]-self.dx).sum()<0.3)
        
    def testFindCorr(self):
        self.IAg.setx(numpy.array([[31.5],[31.5]]))
        x0=self.IAg.x
        dx=self.dx.reshape((2,1))
        window=squarewindow(32)
        corr0=FindCorr(self.AI.Im[0],self.AI.Im[1],x0,dx,[],window)
        corr1=FindCorr(self.AI.Im[0],self.AI.Im[1],x0,dx+0.1,[],window)
        self.assertTrue(corr0>corr1)
        
    def testOptimizedx(self):
        self.IAg.setx(numpy.array([[31.5],[31.5]]))
        x0=self.IAg.x
        dx=self.dx.reshape((2,1))
        window=squarewindow(32)
        dxnew=Optimizex0(self.AI.Im[0],self.AI.Im[1],x0,dx+0.1,[],window)
        self.assertTrue(abs(dxnew-dx).sum()<0.5)

    def testInterp_fft(self):
        self.IAg.setx(numpy.array([[31.5],[31.5]]))
        self.IAg.winsize = 32
        self.IAg.wintype = 'square'
        x0=self.IAg.x
        dx=self.dx.reshape((2,1))
        self.IAg.dx = dx
        fftdx(self.AI.Im[0], self.AI.Im[1], self.IAg)
        window=squarewindow(32)
        interp = piv2d.bicubic_image_interpolate
        piv2d.interp_fft(self.AI.Im[0], self.AI.Im[1], self.IAg, interp)
        self.assertTrue(abs(self.IAg.dx-dx).sum()<0.5)


class TestGaussinterpolation(unittest.TestCase):
    
    def testGaussianinterpolation(self):
        for peakposition in numpy.arange(-0.5,0.501,0.1):
            def gauss(x): return numpy.exp(-(x-peakposition)**2)
            x = numpy.array([-1.0,0,1])
            ifrac = piv2d.gauss_interpolate1(gauss(x))
            self.assertAlmostEqual(peakposition,ifrac)
            
class TestBicubic_image_interpolate(unittest.TestCase):

    def testBicubic_image_interpolate(self):
        import numpy as np
        # function to interpolate
        def parabolic(x):
            x0 = 3.0
            res =  1 - 0.1 *((x0 - x[0,:,:])**2 + (x0 - x[1,:,:])**2)
            return res
        # make image
        range1 = np.arange(7,dtype=float)
        x,y = np.meshgrid(range1,range1)
        x1 = np.array([x, y])
        im = parabolic(x1)
        # grid to interpolate to
        n2 =21
        range2 = np.linspace(2.0, 4.0, n2)
        x,y = np.meshgrid(range2, range2 + 0.01)
        x2 = np.array([x,y])
        im2true = parabolic(x2)
        # do interpolation
        x2flat = x2.reshape((2,-1))
        im2calc = piv2d.bicubic_image_interpolate(im,x2flat)
        im2calc = im2calc.reshape(n2,n2)
        error = im2calc - im2true
        self.assertTrue(abs(error).max()<5*numpy.finfo(float).eps)


        
if __name__=='__main__':
    numpy.set_printoptions(precision=4)
    unittest.main()
