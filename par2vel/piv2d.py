""" 2D processing of Particle Image Velocimetry (PIV) images """
# Copyright Knud Erik Meyer 2017
# Open source software under the terms of the GNU General Public License ver 3

import numpy
import scipy

def xcorr2(a,b):
    """ 2D cross correlation """
    from scipy.fftpack import fft2, ifft2, fftshift
    from numpy import conj
    ni, nj = a.shape
    c = fftshift(ifft2( fft2(a, shape=(2*ni,2*nj))*
                        conj(fft2(b, shape=(2*ni,2*nj)))).real)
    # return standard size, i.e., not first row and first column
    return c[1:,1:]

def findpeakindex(a):
    """ return the indices to peak in correlation plane """
    from numpy import argmax
    ni, nj = a.shape
    ipeak = argmax(a.flat)
    i = ipeak // nj
    j = ipeak - i * ni
    return i, j

def gauss_interpolate1(x):
    """ Return fractional peak position assuming Guassian shape
        x is a vector with 3 elements,
        ifrac returned is relative to center element.
    """
    from numpy import log
    assert (x[1] >= x[0]) and (x[1] >= x[2]), 'Peak must be at center element'
    # avoid log(0) or divide 0 error
    try:
        if any(x <= 0):
            raise ValueError
        r = log(x)
        ifrac = (r[0] - r[2]) / (2 * r[0] - 4 * r[1] + 2 * r[2])
    except ValueError:   # use centroid instead
        print("using centroid")
        ifrac = (x[2] - x[0]) / sum(x)
    return ifrac

def gauss_interpolate2(Im3x3):
    """ Return fractional peak position in 3x3 image assuming Guassian shape """
    ifrac = gauss_interpolate1(Im3x3[:,1])
    jfrac = gauss_interpolate1(Im3x3[1,:])
    return ifrac, jfrac

def bilinear_image_interpolate(Im, x):
    """ return interpolated values from image at points x(2,n) """
    from numpy import take
    # note that i correspond to y and j to x
    ni, nj = Im.shape
    i = x[1,:].astype(int)
    j = x[0,:].astype(int)
    a = x[1,:] - i
    b = x[0,:] - j
    Imx = (1 - a) * (1 - b) * take(Im.flat,i * nj + j) + \
           a * (1 - b) * take(Im.flat,(i + 1) * nj + j) + \
          (1 - a) * b * take(Im.flat,i * nj + j + 1) + \
           a * b * take(Im.flat,(i + 1) * nj + j + 1)
    return Imx

def bicubic_image_interpolate(Im,x):
    """ return interpolated values from image at points x(2,n) """
    # uses the method from "Cubic Convolution Interpolation for
    # Digital Image Processing" by Keys (1981)
    # note that i correspond to y and j to x
    from numpy import zeros, take
    ni, nj = Im.shape
    i = x[1,:].astype(int)
    j = x[0,:].astype(int)
    # test that we are inside borders (need extra row/column at border)
    # assert i.min()>=1 and i.max()<ni-1 and j.min()>=1 and j.max()<nj-1
    # calculate subinterval values and their square/cube values
    si = x[1,:] - i
    si2 = si * si
    si3 = si2 * si
    sj = x[0,:] - j
    sj2 = sj * sj
    sj3 = sj2 * sj
    # calculate interpolation kernel
    u = zeros((4,) + x.shape)
    u[0,0,:] = 0.5 * (-si3 + 2 * si2 - si)
    u[1,0,:] = 1.5 * si3 - 2.5 * si2 + 1
    u[2,0,:] = -1.5 * si3 + 2 * si2 + 0.5*si
    u[3,0,:] = 0.5 * (si3 - si2)
    u[0,1,:] = 0.5 * (-sj3 + 2 * sj2 - sj)
    u[1,1,:] = 1.5 * sj3 - 2.5 * sj2 + 1
    u[2,1,:] = -1.5 * sj3 + 2 * sj2 + 0.5*sj
    u[3,1,:] = 0.5 * (sj3 - sj2)
    # make interpolation
    g = zeros(i.size)
    for m in range(4):
        for n in range(4):
            g += u[m,0,:] * u[n,1,:] * Im.take((i+m-1)*nj + (j+n-1))
    return g
    
def squarewindow(winsize):
    """ Return relative position of points in square window """
    from numpy import mgrid
    winhalf=0.5*winsize-0.5
    window=mgrid[0:winsize,0:winsize]-winhalf
    return window.reshape((2,-1))

def displacementFFT(win1,win2,biascorrect=None):
    """ Find displacement using FFT analysis

        win1, win2: nxn windows cut out of picture, n=2**i
        returns coordinates to peak: irel, jrel
    """
    # get size
    ni, nj = win1.shape
    assert ni == nj
    assert win1.shape == win2.shape
    winsize = ni
    # statistics on windows
    w1f=win1-win1.mean()
    w2f=win2-win2.mean()
    w1std=win1.std()
    w2std=win2.std()
    # calculate normalized cross correlation
    R=xcorr2(w1f,w2f)/(winsize*winsize*w1std*w2std)
    # do bias correction
    if biascorrect is None: 
        biascorrect=xcorr2(numpy.ones((winsize,winsize),float),
                      numpy.ones((winsize,winsize),float))/(winsize*winsize)
    R2 = R/biascorrect
    # find peak 
    ipeak,jpeak=findpeakindex(R)
    # find peak in R2 (might move sligthly)
    ipeak2, jpeak2 = findpeakindex(R2[ipeak-1:ipeak+2,jpeak-1:jpeak+2])
    ipeak = ipeak + ipeak2 - 1
    jpeak = jpeak + jpeak2 - 1
    # fractional position of peak
    try:
        ifrac,jfrac=gauss_interpolate2(R2[ipeak-1:ipeak+2,jpeak-1:jpeak+2])
    except IndexError:  # peak at edge of correlation plane
        ifrac,jfrac=0.0,0.0
    return winsize - 1 - ipeak - ifrac, winsize -1 - jpeak - jfrac    

def fftdx(Im1, Im2, field):
    """ Find particle displacment between two images

        Im1, Im2: two images (double arrays) for correlation
        field   : an Field2D instance contain detail for interrogation
                  field is updated with the result of interrogation
    """
    from numpy import array, zeros, ones, ravel, mean, std, round_
    # prepare some parameters
    assert field.wintype=='square'   # we can only use square windows
    x=field.xflat()
    winsize=field.winsize
    np=x.shape[1]
    dx=zeros([2,np],'double')
    winhalf=winsize/2-0.5  #distance from window center to start and end indices
    biascorrect=xcorr2(ones((winsize,winsize),float),
                       ones((winsize,winsize),float))/(winsize*winsize)
    try:      # try if a guess exists
        dxguess=field.getdxflat()
    except:   # if not, use zero guess
        dxguess=zeros((2,np),float)
    # limits for window centers in order to stay in image
    ni,nj=Im1.shape
    lowxc=array([winhalf,winhalf])
    highxc=array([nj-winhalf-1,ni-winhalf-1])
    # go through all points
    for n in range(np):
        # find window centers (located at a pixel corner)
        xc1=round_(x[:,n]-0.5*dxguess[:,n]+0.5)-0.5
        xc1=xc1.clip(lowxc,highxc)         # don't move outside image
        xc2=round_(xc1+dxguess[:,n]+0.5)-0.5
        xc2=xc2.clip(lowxc,highxc)         # don't move outside image
        # find window indices and extract windows
        # note: i is x[1] direction and j is x[0]-direction,
        #       index valid at pixel center
        i1=int(xc1[1]-winhalf)
        i2=int(xc1[1]+winhalf+1)
        j1=int(xc1[0]-winhalf)
        j2=int(xc1[0]+winhalf+1)
        w1=Im1[i1:i2,j1:j2]
        i1=int(xc2[1]-winhalf)
        i2=int(xc2[1]+winhalf+1)
        j1=int(xc2[0]-winhalf)
        j2=int(xc2[0]+winhalf+1)
        w2=Im2[i1:i2,j1:j2]
        irel, jrel = displacementFFT(w1,w2,biascorrect)
##        # statistics on windows
##        w1f=w1-mean(ravel(w1))
##        w2f=w2-mean(ravel(w2))
##        w1std=std(ravel(w1f))
##        w2std=std(ravel(w2f))
##        # calculate normalized cross correlation
##        R=xcorr2(w1f,w2f)/(winsize*winsize*w1std*w2std)
##        # do bias correction
##        R2=R/BiasCorrect
##        # find peak and fractional position of peak
##        ipeak,jpeak=findpeakindex(R)
##        try:
##            ifrac,jfrac=gauss_interpolate2(R2[ipeak-1:ipeak+2,jpeak-1:jpeak+2])
##        except:  # e.g peak at edge of IA
##            ifrac,jfrac=0.0,0.0
##        # find displacement (switch from (i,j) to (x,y))
##        # note center of correlationplane R(winsize-1,winsize-1) is zero displacement
        dx[0,n] = xc2[0] - xc1[0] + jrel 
        dx[1,n] = xc2[1] - xc1[1] + irel
    field.setdxflat(dx)
        
def FindCorr(Im1,Im2,x0,dx,ddxdx,window):
    """correlation value correspond to displacement ddxdx between two images

       Displacement between two images Im1 og Im2 is defined by a matrix
       ddxdx that specifies displacement and gradients in displacement.
       The input variables are
          Im1, Im2    Two images given as double matrices
          x0          x0 center of 'center' interrogation windows
                         in image coordinates as 2x1 vector
          window      Positions of 'pixel' centers in window relativ to x0
                      given as 2xn vector
          dx          displacement of window centers as 2x1 vector
          ddxdx       ddxdx is a (2x2) matrix with the content 
                         | ddx[0]/dx[0] ddx[0]/dx[1] |
                         | ddx[1]/dx[1] ddx[1]/dx[1] |
                      The relative displacement at a position x is found as
                      |ddx[0]|         |x[0]-x0[0]|
                      |ddx[1]| = ddxdx*|x[1]-x0[1]|
    """
    from numpy import dot,mean,std
    # find displaced windows
    if ddxdx:
        ddx=dot(ddxdx,window)
        w1x=x0-0.5*dx-0.5*ddx
        w2x=x0+0.5*dx+0.5*ddx
    else:
        w1x=x0-0.5*dx+window
        w2x=x0+0.5*dx+window
    # windows values by interpolation
    w1=bilinear_image_interpolate(Im1,w1x)
    w2=bilinear_image_interpolate(Im2,w2x)
#    w1=bicubic_image_interpolate(Im1,w1x)
#    w2=bicubic_image_interpolate(Im2,w2x)
    # do statistics
    w1f=w1-mean(w1)
    w2f=w2-mean(w2)
    w1std=std(w1f)
    w2std=std(w2f)
    # calculate normalized correlation value
    corrvalue=sum(w1f*w2f)/(window.shape[1]*w1std*w2std)
    return corrvalue

def Optimizex0(Im1,Im2,x0,dx,ddxdx,window):
    """Optimize dx and ddxdx at position x0 to give max value of FindCorr

       dx must be guessed within +-1 pixel in advance
       if ddxdx is [], then it is not optimized nor returned
       Optimization is done on each element of ddxdx and dx by
       Gaussinterpolation of the peak in FindCorr.
    """
    from numpy import ones,hstack
    # reshape x0 and dx to be sure of right form
    x0=x0.reshape((2,1))
    dx=dx.reshape((2,1))
    # parameters for optimization
    maxiteration=20  # stop criteria for main loop if slow/no convergence
    stopfrac=0.01    # normal stop criteria - change relativ to start step
    # working arrays - combine to 2x3 array if ddxdx is given
    if ddxdx:
        ddxdxw=hstack((ddxdx.copy(),dx.copy()))
        startstep=hstack((0.1*ones((2,2)),1.0*ones((2,1))))
    else:
        ddxdxw=dx.copy()
        startstep=1.0*ones((2,1))
    step=startstep.copy()
    # loop controlparameters
    corrvalue=FindCorr(Im1,Im2,x0,dx,ddxdx,window)
    stepratio=1.0
    maxcorrectfrac=1.0   # maximum relativ correction
    iteration=0
    while (iteration<maxiteration)and(maxcorrectfrac>stopfrac):
        iteration+=1
        maxcorrectfrac=0.0
        # optimize elements in ddxdxw one by one:
        for i in range(len(ddxdxw.flat)):
            w=ddxdxw.copy()
            w.flat[i]-=stepratio*startstep.flat[i]
            tmp=w.flat[i]
            if ddxdx:
                corrminus=FindCorr(Im1,Im2,x0,w[:,2],w[:,0:2],window)
            else:
                corrminus=FindCorr(Im1,Im2,x0,w,[],window)
            w.flat[i]+=2*stepratio*startstep.flat[i]
            if ddxdx:
                corrplus=FindCorr(Im1,Im2,x0,w[:,2],w[:,0:2],window)
            else:
                corrplus=FindCorr(Im1,Im2,x0,w,[],window)
            # find new value
            if corrplus>corrvalue and corrplus>corrminus: # corrplus is largest
                corrvalue=corrplus
                ddxdxw.flat[i]=w.flat[i]
                maxcorrectfrac=stepratio
            elif corrminus>corrvalue:                     # corrminus is largest
                corrvalue=corrminus
                ddxdxw.flat[i]=tmp
                maxcorrectfrac=stepratio
            else:                                         # peak near center
                correctfrac=gauss_interpolate1([corrminus,corrvalue,corrplus])
                ddxdxw.flat[i]+=correctfrac*stepratio*startstep.flat[i]
                maxcorrectfrac=max(abs(correctfrac),maxcorrectfrac)
                if ddxdx:
                    corrvalue=FindCorr(Im1,Im2,x0,ddxdxw[:,2],ddxdxw[:,0:2],window)
                else:
                    corrvalue=FindCorr(Im1,Im2,x0,ddxdxw,[],window)
            # print iteration, stepratio, maxcorrectfrac, ddxdxw.flat[i], corrvalue
        if maxcorrectfrac<0.3*stepratio:
            stepratio*=0.5
    if ddxdx:
        return ddxdxw[:,2],ddxdxw[:,0:2]
    else:
        return ddxdxw

def Optimizedx(Im1,Im2,field):
    """Optimize correlation function using continious displacement dx
       i.e. using Optimizex0 with no gradients for all points in field
    """
    x=field.xflat()
    dxguess=field.getdxflat()
    dxnew=numpy.empty(dxguess.shape,float)
    np=x.shape[1]
    if field.wintype=='square':
        window=squarewindow(field.winsize)
    else:
        raise Exception('UnknownWindow')
    for n in range(np):
        try: 
            res=Optimizex0(Im1,Im2,x[:,n],dxguess[:,n],[],window)
        except IndexError:
            # if we move out of the image, use initial guess instead
            res=dxguess[:,n]
        dxnew[:,n]=res.flatten()
    field.setdxflat(dxnew)
        
def interp_fft_ia(Im1,Im2,x0,dx,window,interp,biascorrect=None):
    """Use interpolation to get subpixel accuracy for an interrogation area

       x0:     center of interrogation area
       dx:     guessed displacement
       window: relative position of pixels in IA
       interp: interpolation function
       biascorrect: precalculated biascorrection
    """
    from math import sqrt
    # reshape x0 and dx to be sure of right form
    x0=x0.reshape((2,1))
    dx=dx.reshape((2,1))
    # get window size (assume square window)
    winsize = int(sqrt(window.shape[1]))
    # parameters for iterations
    maxiteration = 5   # stop criteria for main loop if slow/no convergence
    stopfrac = 0.1     # normal stop criteria - change relativ to start step
    # interation parameters
    dxcorr = numpy.zeros((2,1))
    # iteration loop
    for n in range(maxiteration):
        w1 = interp(Im1,x0 - 0.5 * (dx + dxcorr) + window)
        w1 = w1.reshape(winsize,winsize)
        w2 = interp(Im2,x0 + 0.5 * (dx + dxcorr) + window)
        w2 = w2.reshape(winsize,winsize)
        irel, jrel = displacementFFT(w1, w2, biascorrect)
        dxcorr[:,0] += jrel, irel
        if abs(irel) < stopfrac and abs(jrel) < stopfrac: break
    return dxcorr.flatten()

def interp_fft(Im1, Im2, field, interp):
    """Improve guessed displacements by iterative fft using
       function interp to interpolate in images
    """
    from numpy import ones
    x=field.xflat()
    dxguess=field.getdxflat()
    dxcorr=numpy.zeros(dxguess.shape, float)
    np=x.shape[1]
    assert field.wintype=='square' 
    winsize = field.winsize
    window=squarewindow(winsize)
    biascorrect=xcorr2(ones((winsize,winsize),float),
                       ones((winsize,winsize),float))/(winsize*winsize)
    for n in range(np):
        try: 
            dxcorr[:,n] = interp_fft_ia(Im1, Im2, x[:,n], dxguess[:,n],
                                  window, interp, biascorrect)
        except IndexError:
            # if we move out of the image, set correction to zero
            dxcorr[:,n] = (0, 0)
    field.setdxflat(dxguess + dxcorr)
        
    
                            


