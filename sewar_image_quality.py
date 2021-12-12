import cv2
import sewar as sw
import numpy as np
import math
from scipy.ndimage.filters import uniform_filter
from PIL import Image
import matplotlib.pyplot as plt

GT=cv2.imread("../Images/fused_multispectral.tif")
P=cv2.imread("../Images/lowresolution_multispetral.tif")
G=cv2.imread("../Images/highresolution_pan.tif",0)


def _compute_bef(im, block_size=8):
	
	if len(im.shape) == 3:
		height, width, channels = im.shape
	elif len(im.shape) == 2:
		height, width = im.shape
		channels = 1
	else:
		raise ValueError("Not a 1-channel/3-channel grayscale image")

	if channels > 1:
		raise ValueError("Not for color images")

	h = np.array(range(0, width - 1))
	h_b = np.array(range(block_size - 1, width - 1, block_size))
	h_bc = np.array(list(set(h).symmetric_difference(h_b)))

	v = np.array(range(0, height - 1))
	v_b = np.array(range(block_size - 1, height - 1, block_size))
	v_bc = np.array(list(set(v).symmetric_difference(v_b)))

	d_b = 0
	d_bc = 0


	for i in list(h_b):
		diff = im[:, i] - im[:, i+1]
		d_b += np.sum(np.square(diff))

	for i in list(h_bc):
		diff = im[:, i] - im[:, i+1]
		d_bc += np.sum(np.square(diff))


	for j in list(v_b):
		diff = im[j, :] - im[j+1, :]
		d_b += np.sum(np.square(diff))


	for j in list(v_bc):
		diff = im[j, :] - im[j+1, :]
		d_bc += np.sum(np.square(diff))


	n_hb = height * (width/block_size) - 1
	n_hbc = (height * (width - 1)) - n_hb
	n_vb = width * (height/block_size) - 1
	n_vbc = (width * (height - 1)) - n_vb


	d_b /= (n_hb + n_vb)
	d_bc /= (n_hbc + n_vbc)

	if d_b > d_bc:
		t = math.log2(block_size)/math.log2(min(height, width))
	else:
		t = 0


	bef = t*(d_b - d_bc)

	return bef

def psnrb(GT, P):
	
	if len(GT.shape) == 3:
		GT = GT[:, :, 0]

	if len(P.shape) == 3:
		P = P[:, :, 0]

	imdff = np.double(GT) - np.double(P)

	mse = np.mean(np.square(imdff.flatten()))
	bef = _compute_bef(P)
	mse_b = mse + bef

	if np.amax(P) > 2:
		psnr_b = 10 * math.log10(255**2/mse_b)
	else:
		psnr_b = 10 * math.log10(1/mse_b)

	return psnr_b
def imresize(arr,size):
    return np.array(Image.fromarray(arr).resize(size))

def d_s (pan,ms,fused,q=1,r=4,ws=7):

	pan = pan.astype(np.float64)
	fused = fused.astype(np.float64)

	pan_degraded = uniform_filter(pan.astype(np.float64), size=ws)/(ws**2)
	
	L = ms.shape[2]

	M1 = np.zeros(L)
	M2 = np.zeros(L)

	for l in range(L):
		M1[l] = sw.uqi(fused[:,:,l],pan)
		M2[l] = sw.uqi(ms[:,:,l],pan_degraded)

	diff = np.abs(M1 - M2)**q
	return ((1./L)*(np.sum(diff)))**(1./q)

def qnr (pan,ms,fused,alpha=1,beta=1,p=1,q=1,r=4,ws=7):

	a = (1-sw.d_lambda(ms,fused,p=p))**alpha
	b = (1-d_s(pan,ms,fused,q=q,ws=ws,r=r))**beta
	return a*b

def get_gradient(image, kernel_size=3) :
    
    grad_x = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=kernel_size)
    grad_y = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=kernel_size)

    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

    return grad
g=get_gradient(G)





print("\n1.	Full reference (FR) quality metrics calculation:\n")
print("mean squared error (mse):",round(sw.mse(GT, P),2))
print("root mean squared error (rmse) :",round(sw.rmse(GT, P),2))
print("peak signal-to-noise ratio (psnr):",round(sw.psnr(GT, P, MAX=None),2))  
S=sw.ssim(GT, P, ws=11, K1=0.01, K2=0.03, MAX=None, fltr_specs=None, mode='valid')
print("structural similarity index (ssim):","(",round(S[0],2),",",round(S[1],2),")") 
print("universal image quality index (uqi) :",round(sw.uqi(GT, P, ws=8),2)) 
M=round(sw.msssim(GT, P, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, K1=0.01, K2=0.03, MAX=None),2)
N=M.real
print("multi-scale structural similarity index (ms-ssim):",N)   
print("erreur relative globale adimensionnelle de synthese (ergas):",round(sw.ergas(GT, P, r=4, ws=8),2))   
print("spatial correlation coefficient (scc) :",round(sw.scc(GT, P, win=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], ws=8),2))
print("relative average spectral error (rase):",round(sw.rase(GT, P, ws=8),2))   
print("spectral angle mapper (sam):",round(sw.sam(GT, P),2)) 
print("PSNR with Blocking Effect Factor :",round(psnrb(GT, P),2))   
print("Pixel Based Visual Information Fidelity (vif-p) :",round(sw.vifp(GT, P, sigma_nsq=2),2)) 

print("\n2.	No reference (NR) quality metrics calculation:\n")
print("Spectral Distortion Index (D_lambda):",round(sw.d_lambda(P, GT),2))
print("Spatial Distortion Index (D_S):",round(d_s(G, P, GT),2))
print("Quality with No Reference (QNR):",round(qnr(G, P, GT),2))

print("\n3.Gradient result\n")
print("1D gradient value:",np.amax(g))
plt.title("2D gradient result:")
plt.imshow(g)


    