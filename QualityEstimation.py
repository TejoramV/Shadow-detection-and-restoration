import cv2
import numpy as np
from scipy import ndimage
from scipy import signal
from scipy import stats
import skimage
import statistics
import math
from matplotlib import pyplot as plt
from PIL import Image
from osgeo import gdal



orginal = Image.open("../Images/1.tif")
org=np.array(orginal)
B=orginal[...,0]
print(orginal)
########################################################
##f1 = gdal.Open('../Images/highresolution_pan.tif')
##band_data1 = f1.GetRasterBand(1).ReadAsArray().astype(np.uint16)
##band_data2 = f1.GetRasterBand(2).ReadAsArray().astype(np.uint16)
##band_data3 = f1.GetRasterBand(3).ReadAsArray().astype(np.uint16)
##orginal=np.stack([band_data3,band_data2,band_data1],axis=2)
#imarray = np.array(orginal)

#prop=orginal.shape
#for i in  range(0,prop[0]):
#    for j in range(0,prop[1]):
#        B=orginal[i,j,0]
#        G=orginal[i,j,1]
#        R=orginal[i,j,2]

################################
org=B

org=cv2.imread("../Images/grass_ns.tif")
print("min:",np.amin(org))
print("max:",np.amax(org))
print("mean:",np.mean(org))
print("Standard deviation:",np.std(org))


print("min:",round(np.amin(org),2))
print("max:",round(np.amax(org),2))
print("mean:",round(np.mean(org),2))


print("median:",round(np.median(org),2))
print("mode:",org.mode)
print("range:",round(np.ptp(org),2))
print("percentile:",round(np.percentile(org,27),2))
print("Interquartile Range(IQR):", round(stats.iqr(org),2))
print("Variance:",round(np.var(org),2))
print("Standard deviation:",round(np.std(org),2))
print("Skewness:",round(stats.skew(org)[0,0],2))
print("kurtosis:",round(stats.kurtosis(org)[0,0],2))
m=np.mean(org)
sd=np.std(org)
snr=np.where(sd == 0, 0, m / sd)
snr=np.asscalar(snr)
print("SNR:",round(snr,2))
    
print("entropy:" ,round(skimage.measure.shannon_entropy(org),2))
