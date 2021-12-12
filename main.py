import cv2
import numpy as np
from scipy import ndimage
from scipy import signal
from scipy import stats
from osgeo import gdal
import skimage
import statistics
import math
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.util import img_as_float

#import os
import csv




    

def test():

#    f1 = gdal.Open('../Images/1.tif')
#    band_data1 = f1.GetRasterBand(1).ReadAsArray().astype(np.uint16)
#    band_data2 = f1.GetRasterBand(2).ReadAsArray().astype(np.uint16)
#    band_data3 = f1.GetRasterBand(3).ReadAsArray().astype(np.uint16)
#    shad=np.stack([band_data3,band_data2,band_data1],axis=2)
    
    shad=cv2.imread("../Images/17.jpg")
    org=shad
    tem=shad
    win_var=shad[...,0]
    

    shad=shad.astype(float)+0.0000000001
    prop=shad.shape
    for i in  range(0,prop[0]):
        for j in range(0,prop[1]):
            B=shad[i,j,0]
            G=shad[i,j,1]
            R=shad[i,j,2]
            c1=np.arctan(R/np.maximum(G,B))
            c2=np.arctan(G/np.maximum(R,B))
            c3=np.arctan(B/np.maximum(R,G))
            
            shad[i,j,0]=c1
            shad[i,j,1]=c2
            shad[i,j,2]=c3
  
   
    outVariance = ndimage.generic_filter(shad[...,2], np.var, size=3)
    
    
    win_mean = ndimage.uniform_filter(shad[...,2],(3,3))
    win_sqr_mean = ndimage.uniform_filter(shad[...,2]**2,(3,3))
    win_var = win_sqr_mean - win_mean**2
    win_var=win_var*1024
    

    cv2.imwrite('../Generated/c1.tif',shad[...,0])    
    cv2.imwrite('../Generated/c2.tif',shad[...,1])
    cv2.imwrite('../Generated/c3.tif',shad[...,2])    
    cv2.imwrite('../Generated/variance-uniform_filter.tif',win_var*255/1024) 
    cv2.imwrite('../Generated/generic-filter.tif',outVariance*255) 
    cv2.imwrite('../Generated/original.tif',org)     
    cv2.imshow('original',org)
    cv2.imshow('c1',shad[...,0])
    cv2.imshow('c2',shad[...,1])
    cv2.imshow('c3',shad[...,2])
    cv2.imshow('variance-generic_filter',outVariance*255)
    cv2.imshow('variance-uniform_filter',win_var*255/1024)
    cv2.waitKey(0) 

       
## #  Apply gamma correction.    
 
    gamma=2                                                                                                  ###################
    gamm_corrected = np.array(org)
    orgin =np.array(org)
    shadow_count=0
    nonshadow_count=0
    
    
    for i in  range(0,prop[0]):
        for j in range(0,prop[1]):
            if (win_var[i,j]>2.5):                                                                           #####################
                gamm_corrected[i,j] = np.array(255*(org[i,j] / 255) ** (1/gamma), dtype = 'uint8')
                shadow_count=shadow_count+1                
            else:
                gamm_corrected[i,j]=np.array(org[i,j], dtype = 'uint8')
                nonshadow_count=nonshadow_count+1                
            
    org =np.array(orgin)
                
                
    median = cv2.medianBlur(gamm_corrected,5)
    
    blur = cv2.bilateralFilter(gamm_corrected,9,75,75)
    
    print('shadow_count',shadow_count)
    print('nonshadow_count',nonshadow_count)    
    cv2.imshow('original',org)            
    cv2.imshow("gamma corrected", gamm_corrected)
    cv2.imshow("median",median)
    cv2.imshow("bilateral",blur)
    cv2.imwrite('../Generated/gamma corrected.tif',gamm_corrected) 
    cv2.imwrite('../Generated/median_gamma corrected.tif',median) 
    cv2.imwrite('../Generated/bilateral_gamma corrected.tif',blur) 
    cv2.waitKey(0)

   
   
 # Apply Linear-Correlation correction.     


   
    a=0
    b=0
    ms=[0,0,0]
    mn=[0,0,0]
    ss=[0,0,0]
    sn=[0,0,0]
    for i in  range(0,prop[0]):
        for j in range(0,prop[1]):
            if (win_var[i,j]>2.5):                                                                           #####################
               ms += np.array(org[i,j])
               a=a+1
            else:
               mn += np.array(org[i,j])
               b=b+1
    ms= ms/a
    mn=mn/b
    
    for i in  range(0,prop[0]):
        for j in range(0,prop[1]):
            if (win_var[i,j]>2.5):                                                                           #####################
               ss += (np.array(org[i,j])-ms)**2
            else:
               sn += (np.array(org[i,j])-mn)**2
                   
 

    ss = ss/a-1
    ss[0] = math.sqrt(ss[0]) 
    ss[1] = math.sqrt(ss[1])  
    ss[2] = math.sqrt(ss[2])                 

    sn = sn/b-1
    sn[0] = math.sqrt(sn[0])  
    sn[1] = math.sqrt(sn[1])  
    sn[2] = math.sqrt(sn[2])      
               
               
    #print(ms,mn,ss,sn)  

    linear_corrected = np.array(org)
    orgin =np.array(org)
    
    for i in  range(0,prop[0]):
        for j in range(0,prop[1]):
            if (win_var[i,j]>2.5):                                                                           #####################
                linear_corrected[i,j] = np.array(((sn/ss)*(org[i,j]-ms)+mn) , dtype = 'uint8') 
            else:
                linear_corrected[i,j]=np.array(org[i,j], dtype = 'uint8')
    org =np.array(orgin)
    median1 = cv2.medianBlur(linear_corrected,5)
    
    blur1 = cv2.bilateralFilter(linear_corrected,9,75,75)
    cv2.imshow('original',org)            
    cv2.imshow("linear corrected", linear_corrected)
    cv2.imshow("median",median1)
    cv2.imshow("bilateral",blur1)
    cv2.imwrite('../Generated/linear corrected.tif',linear_corrected) 
    cv2.imwrite('../Generated/median_linear corrected.tif',median1) 
    cv2.imwrite('../Generated/bilateral_linear corrected.tif',blur1) 
    cv2.waitKey(0)

 # HISTOGRAM MATCHING
    
    himg=np.array(org)
    def hrep(k):
        a=0
        b=0
        for i in  range(0,prop[0]):
            for j in range(0,prop[1]):
                if (win_var[i,j]>2.5):                                                                           #####################
                    a=a+1
                else:
                    b=b+1

        MatrixS = [[0 for x in range(a)] for y in range(2)]
        MatrixN = [[0 for x in range(b)] for y in range(2)]
        MatrixS=np.float32(MatrixS)
        MatrixN=np.float32(MatrixN)
        
        a=0
        b=0  
        for i in  range(0,prop[0]):
            for j in range(0,prop[1]):
                if (win_var[i,j]>2.5):
                    MatrixS[0][a]=np.array(org[i,j,k])
                    a=a+1
                else:
                    MatrixN[0][b]=np.array(org[i,j,k])
                    b=b+1                
        for i in range(0,a):
            MatrixS[1][i]=MatrixS[0][i]
        
        for j in range(0,b):
            MatrixN[1][j]=MatrixN[0][j]    
        MatrixS=np.array(MatrixS,dtype=np.uint8)
        MatrixN=np.array(MatrixN,dtype=np.uint8)
 

        def hist_match(source, template):
            

            oldshape = source.shape
            source = source.ravel()
            template = template.ravel()
            
            s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                    return_counts=True)
            t_values, t_counts = np.unique(template, return_counts=True)
            
            s_quantiles = np.cumsum(s_counts).astype(np.float64)
            s_quantiles /= s_quantiles[-1]
            t_quantiles = np.cumsum(t_counts).astype(np.float64)
            t_quantiles /= t_quantiles[-1]
           
            interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
   

            return interp_t_values[bin_idx].reshape(oldshape)





        def ecdf(x):
            
            vals, counts = np.unique(x, return_counts=True)
            ecdf = np.cumsum(counts).astype(np.float64)
            ecdf /= ecdf[-1]
            return vals, ecdf

        def hplot():
            x1, y1 = ecdf(source.ravel())
            x2, y2 = ecdf(template.ravel())
            x3, y3 = ecdf(matched.ravel())

            fig = plt.figure()
            gs = plt.GridSpec(2, 3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[1, :])
            for aa in (ax1, ax2, ax3):
                aa.set_axis_off()
     
      
            ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
            ax4.plot(x2, y2 * 100, '-k', lw=3, label='Template')
            ax4.plot(x3, y3 * 100, '--r', lw=3, label='Matched')
            ax4.set_xlim(x1[0], x1[-1])
            ax4.set_xlabel('Pixel value')
            ax4.set_ylabel('Cumulative %')
            ax4.legend(loc=5)
            return 0

        source = np.array(MatrixS)
        
        template = np.array(MatrixN)
        matched = hist_match(source, template)

        hplot()
    
        a=0
    
        for i in  range(0,prop[0]):
            for j in range(0,prop[1]):
                if (win_var[i,j]>2.5):
                    himg[i,j,k]=np.array(matched[0][a],dtype=np.uint8)
                    a=a+1
        return himg[...,k]
                
    for k in  range(0,3):
        himg[...,k]=np.array(hrep(k))        
    cv2.imshow("histogram matched",himg)
    median2 = cv2.medianBlur(himg,5)
    
    blur2 = cv2.bilateralFilter(himg,9,75,75)
    cv2.imshow('original',org)            
  
    cv2.imshow("median",median2)
    cv2.imwrite('../Generated/histogram matched.tif',himg) 
    cv2.imwrite('../Generated/median_histogram matched.tif',median2) 
    cv2.imwrite('../Generated/bilateral_histogram matched.tif',blur2) 
    cv2.imshow("bilateral",blur2)
    cv2.waitKey(0)   
    

    

test()

