# unittest of camera
# Copyright Knud Erik Meyer 2017
# Open source software under the terms of the GNU General Public License ver 3

import unittest
import numpy
from par2vel.camera import One2One
from par2vel.artimage import ArtImage

class TestArtImage(unittest.TestCase):
    def testTinySpot(self):
        AI = ArtImage(One2One((5,5)), particle_image_size=0.1)
        AI.particle_positions(numpy.array([[2.0], [2.0], [0]]))
        AI.generate_images()
        self.assertTrue(AI.Im[0][2,2]>0.0)
        Im0 = AI.Im[0].copy()
        Im0[2,2] = 0.0
        self.assertAlmostEqual(abs(Im0).sum(),0.0)
    def testTinySpotPosition(self):
        AI = ArtImage(One2One((9,9)), particle_image_size=0.1)
        AI.particle_positions(numpy.array([[3.0], [1.0], [0]]))
        AI.generate_images()
        # note x[0] is j and x[1] is i
        self.assertTrue(AI.Im[0][1,3]>0.1)
        Im0 = AI.Im[0].copy()
        Im0[1,3] = 0.0
        self.assertAlmostEqual(abs(Im0).sum(),0.0)
    def testTinySpotCorner(self):
        AI = ArtImage(One2One((5,5)), particle_image_size=0.1)
        AI.particle_positions(numpy.array([[-0.0002],[-0.0002],[0]]))
        AI.generate_images()
        self.assertTrue(AI.Im[0][0,0]>0.0)
        Im0=AI.Im[0].copy()
        Im0[0,0]=0.0
    def testSpotOnPixelEdge(self):
        AI = ArtImage(One2One((5,5)), particle_image_size=2)
        AI.particle_positions(numpy.array([[2.5], [2.0], [0]]))
        AI.generate_images()
        # note x[0] is j and x[1] is i
        self.assertAlmostEqual(AI.Im[0][2,2],AI.Im[0][2,3])
    def testPixelEdge(self):
        AI = ArtImage(One2One((8,8)), particle_image_size=1.0)
        AI.particle_positions(numpy.array([[3.5], [3.5], [0]]))
        AI.generate_images()
        Im=AI.Im[0]
        self.assertTrue(Im[3,3]>0.0)
        self.assertAlmostEqual(Im[3,3],Im[3,4])
        self.assertAlmostEqual(Im[3,3],Im[4,3])
        self.assertAlmostEqual(Im[3,3],Im[4,4])
        Im0=Im.copy()
        Im0[3:5,3:5]=0
        self.assertTrue(abs(Im0).sum()<0.001*Im[3,3])
    def test1DSpot(self):
        AI = ArtImage(One2One((5,5)), particle_image_size=2.0)
        AI.particle_positions(numpy.array([[2.0], [2.0], [0]]))
        AI.generate_images()
        Im=AI.Im[0]
        self.assertAlmostEqual(Im[1,2],Im[3,2])
        self.assertTrue(Im[0,2]/Im[2,2]<0.001)
        self.assertTrue(Im[1,2]/Im[2,2]>0.1)
        self.assertAlmostEqual(Im[2,2],1.0)
    def testOutofImageSpot(self):
        AI = ArtImage(One2One((5,5)), particle_image_size=2)
        AI.particle_positions(numpy.array([[-1.0], [2.0], [0]]))
        AI.generate_images()
        Im=AI.Im[0]
        self.assertTrue(Im[2,0]>0.01)
        self.assertTrue(abs(Im.sum())<Im[2,0]*2)


if __name__=='__main__':
    numpy.set_printoptions(precision=4)
    unittest.main()
