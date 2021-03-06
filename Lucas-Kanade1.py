import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import scipy.signal as sp1

##### Reading the images ###############
Image1 = cv2.imread('C:/Users/saura/Desktop/CVassng/basketball1.png',0)
Image2 = cv2.imread('C:/Users/saura/Desktop/CVassng/basketball2.png',0)
#Image1 = cv2.imread('C:/Users/saura/Desktop/CVassng/grove1.png',0)
#Image2 = cv2.imread('C:/Users/saura/Desktop/CVassng/grove2.png',0)

x,y = Image1.shape;
print (x,y)

##### Gaussian Blurring the images ############
def filter_to_Gaussian(I1,kernel,sigma):
    G = []
    for i in kernel: # converting a 1D filter to a Gaussian filter
        Gauss = (1 / np.sqrt(2 * np.pi)) * np.exp(-1 * i ** 2 / (2 * sigma ** 2))
        G.append(Gauss)
    G = np.asarray(G)
    #print G
    return G



sigma = 1
kernal = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
G = filter_to_Gaussian(Image1,kernal,sigma)
#Igx= sp1.convolve2d(Image1,G)
#Igy= sp1.convolve2d(Image1,np.transpose(G))
Img1 = cv2.GaussianBlur(Image1,(3,3),0)
Img2 = cv2.GaussianBlur(Image2,(3,3),0)

#plt.imshow(Img1,cmap='gray')
#plt.show()
#plt.imshow(Img2,cmap='gray')
#plt.show()

######## Reducing the image shape ###########


def img_reduction(img):
	x,y = img.shape
	x = math.floor(x/2)
	y = math.floor(y/2)
	reduced_img=np.zeros((x,y))
	for i in range(0,x):
		for j in range(0,y):
			reduced_img[i,j]= img[i*2,j*2]
	return reduced_img

	
img1 = img_reduction(Img1)

img11 = img_reduction(img1)

img2 = img_reduction(Img2)

img22 = img_reduction(img2)
#plt.imshow(img11,cmap='gray')
#plt.show()
#plt.imshow(img22,cmap='gray')
#plt.show()
	
X_mask=np.array([[-1,1],[-1,1]])
Y_mask=np.array([[-1,-1],[1,1]])
t1=np.array([[-1,-1],[-1,-1]])
t2=np.array([[1,1],[1,1]])	
	
#print (X_mask)
#print (Y_mask)
#print (t1)
#print (t2)	
	
####### Now convolve this masks with reduced images #############

IX=(sp1.convolve2d(img11,X_mask)+sp1.convolve2d(img22,X_mask))/2
#print(IX)
IY=(sp1.convolve2d(img11,Y_mask)+sp1.convolve2d(img22,Y_mask))/2
#print(IY)
IXY=(sp1.convolve2d(img11,Y_mask)+sp1.convolve2d(img22,Y_mask))/2
#print(IXY)
IT=(sp1.convolve2d(img11,t1)+sp1.convolve2d(img22,t2))
#print (IT)
	
############ CORNER DETECTION ###############################
Corner_parameters = dict(maxCorners = 400,
                       qualityLevel = 0.235,
                       minDistance = 7,
                       blockSize = 10)
Corners = cv2.goodFeaturesToTrack(Image1,mask=None, **Corner_parameters)
Corners =(Corners/2**2)
Corners = np.int32(Corners)

for Corner in Corners:
	i,j = Corner.ravel()
	cv2.circle(Image1,(i,j),3,255,-1)

#plt.imshow(Image1,cmap='gray')
#plt.show()

CImage = cv2.imread('C:/Users/saura/Desktop/CVassng/basketball1.png')
#CImage = cv2.imread('C:/Users/saura/Desktop/CVassng/grove1.png')

color = np.random.randint(0,255,(200,3))

for Corner in Corners:
	x,y=Corner.ravel()
	#print (x,y)
	Cx = (math.pow(IX[y,x],2))+(math.pow(IX[y+1,x],2))+(math.pow(IX[y,x+1],2))+(math.pow(IX[y-1,x],2))+(math.pow(IX[y,x-1],2))+(math.pow(IX[y,x+1],2))+(math.pow(IX[y+1,x-1],2))+(math.pow(IX[y+1,x+1],2))+(math.pow(IX[y-1,x-1],2))+(math.pow(IX[y-1,x+1],2))
	#print (Cx)
	Cy = (math.pow(IY[y,x],2))+(math.pow(IY[y+1,x],2))+(math.pow(IY[y,x+1],2))+(math.pow(IY[y-1,x],2))+(math.pow(IY[y,x-1],2))+(math.pow(IY[y,x+1],2))+(math.pow(IY[y+1,x-1],2))+(math.pow(IY[y+1,x+1],2))+(math.pow(IY[y-1,x-1],2))+(math.pow(IY[y-1,x+1],2))
	#print (Cy)
	Cxy = (IX[y,x]*IY[y,x])+(IX[y+1,x]*IY[y+1,x])+(IX[y,x+1]*IY[y,x+1])+(IX[y-1,x]*IY[y-1,x])+(IX[y,x-1]*IY[y,x-1])+(IX[y,x+1]*IY[y,x+1])+(IX[y+1,x-1]*IY[y+1,x-1])+(IX[y+1,x+1]*IY[y+1,x+1])+(IX[y-1,x-1]*IY[y-1,x-1])+(IX[y-1,x+1]*IY[y-1,x+1])
	#print (Cxy)
	Ct = (IT[y,x])+(IT[y+1,x])+(IT[y,x+1])+(IT[y-1,x])+(IT[y,x-1])+(IT[y,x+1])+(IT[y+1,x-1])+(IT[y+1,x+1])+(IT[y-1,x-1])+(IT[y-1,x+1])
	#print(Ct)
	
	O=np.matrix([[Cx,Cxy],[Cxy,Cy]])
	P=np.matrix([[Cx*Ct],[Cy*Ct]])

	m,n=np.linalg.pinv(O)*-P
	Corners = np.int32(Corners)
	
	M=[]
	N=[]
	if np.all(y!= y+m) and np.all(x!= x+n):
		M.append(m)
		N.append(n)


	i=0
   
	color = np.random.randint(0,255,(100,3))
	cv2.circle(CImage, (((x+M)*4),((y+N)*4)), 3, (color[i].tolist()), thickness=3, lineType=8, shift=0)
	
	cv2.circle(CImage, ((x*4),(y*4)), 3, (color[i].tolist()), thickness=3, lineType=8, shift=0)
	i=i+2
	

plt.imshow(CImage)
plt.show()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	