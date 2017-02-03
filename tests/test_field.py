# unittest of Field2D

import unittest
import numpy
from par2vel.field import Field2D
from par2vel.camera import Camera

class testField2D(unittest.TestCase):
    
    def testSquarewindows(self):
        fld=Field2D(Camera((32,48)))
        fld.squarewindows(32, 0.5)
        xres = numpy.array([[[15.5],[31.5]],[[15.5],[15.5]]])
        self.assertAlmostEqual(abs(fld.x-xres).sum(),0.0)
        self.assertAlmostEqual(abs(fld.xflat()-xres.reshape((2,2))).sum(),0.0)
    def testMedianfilter(self):
        fld=Field2D(Camera((128,128)))
        fld.squarewindows(32,0.0)
        dx=fld.xflat()
        dx[0,:]=[0.1,0.3,0.2,0.4,0.2,0.3,0.2,0.3,0.1,0.3,0.2,0.4,0.2,0.3,0.2,0.1]
        dx[1,:]=0.5
        fld.setdxflat(dx)
        fld.dx[:,1,1]=[10,0.5]  # set an outlier
        fld.medianfilter(2.0)           # test medianfilter
        outlierindex=numpy.array(fld.outlier.nonzero())
        resindex=numpy.array([[1],[1]])
        self.assertAlmostEqual(abs(outlierindex-resindex).sum(),0.0)
##    def testMaxdxfilter(self):
##        fld=Field2D(Camera('',(128,128)))
##     def testdx2U(self):
##        fld=Field2D(Camera('',(128,128)))
##        X1=numpy.array([0.01,0.005,0.0]).reshape((3,1))
##        fld.setX(X1)
##        self.assertAlmostEqual(abs(fld.getX()-X1).sum(),0)
##        fld.setdxflat(numpy.array([10,5.]).reshape((2,1)))
##        fld.squarewindows(32,0.0)
##        x=fld.xflat()
##        fld.setdxflat(x/10.0)  # dummy displacements
##        fld.dx[:,1,1]=[20,10]  # set an outlier
##        fld.maxdxfilter(19)    # test maxdxfilter
##        outlierindex=numpy.array(fld.outlier.nonzero())
##        resindex=numpy.array([[1],[1]])
##        self.assertAlmostEqual(abs(outlierindex-resindex).sum(),0.0)
    def testMeanfilter(self):
        fld=Field2D(Camera((128,128)))
        fld.squarewindows(32,0.0)
        x=fld.xflat()
        fld.setdxflat(x/10.0)  # dummy displacements
        fld.dx[:,1,1]=[20,10]  # set an outlier
        resindex=numpy.array([[1],[1]])
        fld.meanfilter(2.0)             # test meanfilter
        outlierindex=numpy.array(fld.outlier.nonzero())
        self.assertAlmostEqual(abs(outlierindex-resindex).sum(),0.0)
    def testLocalMean(self):
        fld=Field2D(Camera((128,128)))
        fld.squarewindows(32,0.0)
        fld.setdxflat(numpy.array([list(range(16)),list(range(16))]))
        meandx,stddx,n=fld.localmean(1,1)
        self.assertAlmostEqual(meandx[0],5.0)
        meandx,stddx,n=fld.localmean(1,0)
        self.assertAlmostEqual(meandx[0],4.0)
        meandx,stddx,n=fld.localmean(0,0)
        self.assertAlmostEqual(meandx[0],(1+4+5)/3.)
        fld.outlier[0,1]=True
        fld.dx[0,0,0]=1
        meandx,stddx,n=fld.localmean(1,1)
        self.assertAlmostEqual(meandx[0],(1+2+4+6+8+10)/6.)
    def testReplaceoutlier(self):
        fld=Field2D(Camera((128,128)))
        fld.squarewindows(32,0.0)
        x=fld.xflat()
        fld.setdxflat(x/10.0) # dummy displacements
        res1=fld.dx[:,1,1]
        fld.dx[:,1,1]=[0,0]
        res2=fld.dx[:,2,2]
        fld.outlier[1,1]=True
        fld.replaceoutlier()
        self.assertAlmostEqual(abs(fld.dx[:,1,1]-res1).sum(),0.0)
        self.assertTrue(fld.replaced[1,1])
        fld.outlier[2,2]=True
        fld.replaceoutlier()
        self.assertAlmostEqual(abs(fld.dx[:,2,2]-res2).sum(),0.0)
        x=fld.xflat()
        fld.setdxflat(x/10.0) # dummy displacements
        fld.outlier[0,0]=True
        res=numpy.mean(fld.dx[0,[0,1,1],[1,0,1]])
        fld.replaceoutlier()
        self.assertAlmostEqual(abs(fld.dx[:,0,0]-[res,res]).sum(),0)

if __name__=='__main__':
    numpy.set_printoptions(precision=4)
    unittest.main()
