# unittest of camera
# Copyright Knud Erik Meyer 2017
# Open source software under the terms of the GNU General Public License ver 3

import unittest
import numpy
import os
from par2vel.camera import *


class testCamera(unittest.TestCase):

    def test_X2x(self):
        cam = Camera((32,32))
        X = numpy.array([1.0, 1.0, 0]).reshape(3,-1)
        x = cam.X2x(X)
        self.assertAlmostEqual((x-[1, 1]).sum(),0)

    def test_x2X(self):
        cam = One2One((32,32))
        X = numpy.array([1.0, 1.0, 0]).reshape(3,-1)
        x = cam.X2x(X)
        X2 = cam.x2X(x)
        self.assertAlmostEqual((X-X2).sum(),0)

    def test_dx2dX(self):
        cam = One2One((32,32))
        dx = numpy.array([1.0, 1.0]).reshape(2,-1)
        x = dx
        dX = cam.dx2dX(x, dx)
        self.assertAlmostEqual((dX[0:2,:]-dx).sum(),0)

    def test_keywords(self):
        cam1 = Camera()
        cam1.focal_length = 0.06
        cam1.pixel_pitch = (0.00001, 0.00001)
        filename = 'temporary1.cam'
        cam1.save_camera(filename)
        cam2 = Camera()
        cam2.read_camera(filename)
        self.assertAlmostEqual(cam2.focal_length,0.06)
        self.assertAlmostEqual(cam2.pixel_pitch[1],0.00001)
        os.remove(filename)

##    def test_X2x(self):
##        cam = Camera((32,32))
##        cam.set_physical_size()
##        dx = numpy.array([[1],[0]])
##        dX = cam.dx2dX(dx,dx) 
##        self.assertAlmostEqual(numpy.sqrt((dX**2).sum()), 1)

class testLinear2d(unittest.TestCase):

    def test_X2x(self):
        from numpy import array
        cam = Linear2d()
        cam.set_calibration(array([[1, 0, 0.1], [0, 2, 0.2]]))
        X = array([[1, 0.5], [1, 0.6], [0.1, 0.2]])
        x_result = array([[1.1, 0.6],[2.2, 1.4]])
        x = cam.X2x(X)
        self.assertAlmostEqual((x - x_result).sum(),0)

    def test_x2X(self):
        from numpy import array
        cam = Linear2d()
        cam.set_calibration(array([[0.1, 0.01, 10], [0.02, 0.1, 11]]))
        X = array([[1, 0.5, -0.1], [1, 0.6, -0.12], [0, 0, 0]])
        x = cam.X2x(X)
        Xres = cam.x2X(x)
        self.assertAlmostEqual((X - Xres).sum(),0)        

    def test_dx2dX(self):
        from numpy import array
        cam = Linear2d()
        cam.set_calibration(array([[1, 0, 0.1], [0, 2, 0.2]]))
        x = array([[15.5, 32.5], [0, 15.5]])
        dx = array([[1, 1.2],[2,0]])
        dXres = array([[1, 1.2],[0, 1]])
        dX = cam.dx2dX(x,dx)
        self.assertAlmostEqual((dX - dXres).sum(), 0)

    def test_save_read(self):
        cam1 = Linear2d()
        cam1.focal_length = 0.06
        calib = numpy.array([[0.1, 0, 100], [0, 0.11, 101]])
        cam1.set_calibration(calib)
        filename = 'temporary2.cam'
        cam1.save_camera(filename)
        cam2 = Linear2d()
        cam2.read_camera(filename)
        self.assertAlmostEqual(cam2.focal_length,0.06)
        self.assertAlmostEqual((cam2.calib - calib).sum(), 0)
        os.remove(filename)

class testLinear3d(unittest.TestCase):

    def test_X2x(self):
        from numpy import array
        cam = Linear3d()
        matrix = array([[1.0, 0, 0, 0],
                        [0.0, 1, 0, 0],
                        [0.0, 0, 0, 1]])
        cam.set_calibration(matrix)
        X = array([[1.0], [1.0], [0]])
        x_result = array([[1.0],[1.0]])
        x = cam.X2x(X)
        self.assertAlmostEqual((x - x_result).sum(),0)

    def test_save_read(self):
        from numpy import array
        cam1 = Linear3d()
        cam1.focal_length = 0.04
        calib = array([[1.0, 0, 0, 0],
                       [0.0, 1, 0, 0],
                       [0.0, 0, 0, 1]])
        cam1.set_calibration(calib)
        filename = 'temporary4.cam'
        cam1.save_camera(filename)
        cam2 = Linear3d()
        cam2.read_camera(filename)
        self.assertAlmostEqual(cam2.focal_length,0.04)
        self.assertAlmostEqual((cam2.calib - calib).sum(), 0)
        os.remove(filename)


class testScheimpflug(unittest.TestCase):

    def test_X2x(self):
        from numpy import array, pi, sqrt
        cam = Scheimpflug((16,16))
        cam.focal_length = 0.06
        cam.pixel_pitch = (1e-5, 1e-5)
        cam.set_calibration(pi/4, 0.1)
        X = array([[0.0], [0.0], [0.0]])
        x_result = array([[7.5],[7.5]])
        x = cam.X2x(X)
        self.assertAlmostEqual((x - x_result).sum(),0)
        X = array([[-0.001], [0.0], [0.001]])
        x_result = array([[7.5],[7.5]])
        x = cam.X2x(X)
        self.assertAlmostEqual((x - x_result).sum(),0)
        X = array([[0.0001*sqrt(2)], [0.0], [0.0]])
        x_result = array([[8.5],[7.5]])
        x = cam.X2x(X)
        self.assertTrue(abs((x - x_result).sum())<0.005,0)

    def test_x2X(self):
        from numpy import array, pi
        cam = Scheimpflug()
        cam.set_calibration(pi/4, 0.1)
        X = array([[0.0, 0.01, 0.00,  0.01],
                   [0.0, 0.00, 0.01, -0.01],
                   [0,   0,    0,     0     ]])
        x = cam.X2x(X)
        Xresult = cam.x2X(x)
        self.assertAlmostEqual(abs((X - Xresult).sum()),0)
        
    def test_dx2dX(self):
        from numpy import array, pi, zeros
        cam = Scheimpflug()
        cam.set_calibration(pi/4, 0.1)
        X = array([[0.0], [0.0], [0.0]])
        dX = array([[0.001], [-0.002], [0.0]])
        x = cam.X2x(X)
        dx = cam.dX2dx(X, dX)
        dX2 = cam.dx2dX(x, dx)
        self.assertAlmostEqual(abs((dX - dX2).sum()),0)

    def test_save_read(self):
        cam1 = Scheimpflug()
        cam1.focal_length = 0.05
        cam1.set_calibration(numpy.pi/4, 0.1)
        filename = 'temporary3.cam'
        cam1.save_camera(filename)
        cam2 = Scheimpflug()
        cam2.read_camera(filename)
        self.assertAlmostEqual(cam2.focal_length,0.05)
        self.assertAlmostEqual(cam2.theta, numpy.pi/4)
        self.assertAlmostEqual(cam2.M, 0.1)
        os.remove(filename)



if __name__=='__main__':
    numpy.set_printoptions(precision=4)
    unittest.main()
