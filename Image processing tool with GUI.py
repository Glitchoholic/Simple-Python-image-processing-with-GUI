
#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################
#######################                                                                                     #############################                     
#######################                                                                                     ############################# 
#######################     Karim Mohamed Naguib Abd-Alaziz Mohamed      كريم محمد نجيب عبد العزيز محمد   ############################# 
#######################                                                                                     ############################# 
#######################         karim.naguib.eg@gmail.com              Academic NO.: 1600225                ############################# 
#######################                                                                                     ############################# 
#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################



from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
import os
import random
from skimage.util import random_noise
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread, imshow, show, subplot, title, get_cmap, hist
from skimage.exposure import equalize_hist
import sys 
import math
from scipy import ndimage

interface = Tk()
interface.geometry('650x700+150+150')
interface.title('Image processing project. Karim Mohamed Naguib, 1600225')


def openImg():
    global orgImage
    openImg.imagePath = filedialog.askopenfilename(initialdir=r'C:\Users', title="Choose an image..")
    orgImage = ImageTk.PhotoImage(Image.open(openImg.imagePath))



loadImageFrame = LabelFrame(interface, text="Load Image", padx=40, pady=26)
loadImageFrame.grid(row=0, column=0, padx=20)
loadImage      = Button(loadImageFrame, text="Open", command=openImg).pack()



def grayScale():
    orgImg = cv2.imread(openImg.imagePath)
    grayImg= cv2.cvtColor(orgImg, cv2.COLOR_BGR2GRAY)
    cv2.imshow('GrayScale image', grayImg)

def originalImage():
    orgImg= cv2.imread(openImg.imagePath)
    cv2.imshow('Original image', orgImg)




convertVar = IntVar()
convertVar.set(0)
convertFrame = LabelFrame(interface,text="Convert", padx=20, pady=17)
convertFrame.grid(row=0, column=1, padx=15)
Radiobutton(convertFrame, text="Gray scale", variable= convertVar, value=0, command= lambda: grayScale(), width=20).grid(row=0, column=0, padx=5)
Radiobutton(convertFrame, text="Original colors",    variable= convertVar, value=1, command= lambda: originalImage() , width=20).grid(row=1, column=0, padx=5)





def noisy(noise_typ,image,imgVer):
    image= cv2.imread(openImg.imagePath, imgVer)
    if noise_typ == "gauss":
        mean=0.13
        var=0.017
        image=np.array (image/255, dtype=float)
        noise=np.random.normal (mean, var ** 0.65, image.shape)
        out=image + noise
        if out.min ()<0:
            low_clip=-1.
        else:
            low_clip=0.
        out=np.clip (out, low_clip, 1.0)
        out=np.uint8 (out * 255)
        cv2.imshow('Gauss image', out)
    
    elif noise_typ == "s&p":
        output=np.zeros (image.shape, np.uint8)
        thres=1-0.04
        for i in range (image.shape [0]):
            for j in range (image.shape [1]):
                rdn=random.random ()
                if rdn<0.05:
                    output [i] [j]=0
                elif rdn>thres:
                    output [i] [j]=255
                else:
                    output [i] [j]=image [i] [j]
        
        cv2.imshow('Salt&Pepper image', output)
        
    elif noise_typ == "poisson":
      noisy = random_noise(image, mode="poisson")
      cv2.imshow('Poisson image', noisy)
      

    


noiseVar = IntVar()
noiseVar.set(3)
noiseFrame = LabelFrame(interface,text="Add Noise", padx=12, pady=6)
noiseFrame.grid(row=0, column=2, padx=15)
Radiobutton(noiseFrame, text="Salt & Papper noise", variable= noiseVar, value=0, command= lambda: noisy('s&p', openImg.imagePath, convertVar.get()),  width=20).grid(row=0, column=0, padx=5)
Radiobutton(noiseFrame, text="Gaussian noise",      variable= noiseVar, value=1, command= lambda: noisy('gauss', openImg.imagePath, convertVar.get()), width=20).grid(row=1, column=0, padx=5)
Radiobutton(noiseFrame, text="Poisson noise",       variable= noiseVar, value=2, command= lambda: noisy('poisson', openImg.imagePath, convertVar.get()), width=20).grid(row=2, column=0, padx=5)

def adjBritCont(imageIn):
    
    image = cv2.imread(imageIn, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    MAX_VALUE = 10
    MAX_VALUE2 = 100
    MIN_VALUE = 0
    cv2.namedWindow("Brit&Cont. Press ESC to exit!!!!!", cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar("Brightness", "Brit&Cont. Press ESC to exit!!!!!", MIN_VALUE, MAX_VALUE, lambda x:x)
    cv2.createTrackbar("Contrast", "Brit&Cont. Press ESC to exit!!!!!", MIN_VALUE, MAX_VALUE2, lambda x:x)

    while True:

        hlsCopy = np.copy(hlsImg)
        Brightness = cv2.getTrackbarPos('Brightness', 'Brit&Cont. Press ESC to exit!!!!!')
        Contrast = cv2.getTrackbarPos('Contrast', 'Brit&Cont. Press ESC to exit!!!!!')
        hlsCopy[:, :, 1] = (1.0 + Brightness / float(MAX_VALUE)) * hlsCopy[:, :, 1]
        hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
        hlsCopy[:, :, 2] = (1.0 + Contrast / float(MAX_VALUE2)) * hlsCopy[:, :, 2]
        hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
        lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
        cv2.imshow("Brit&Cont. Press ESC to exit!!!!!", lsImg)
        ch = cv2.waitKey(5)
        if ch == 27:
            break


def histoAndEq(image):
    
    img = cv2.imread(image, 0)
    eq = np.asarray(equalize_hist(img) * 255, dtype='uint8')
    subplot(221); imshow(img, cmap= 'gray'); title('Original')
    subplot(222); hist(img.flatten(), 256, range=(0,256)); title('Histogram of origianl')
    subplot(223); imshow(eq, cmap= 'gray');  title('Histogram Equalized')
    subplot(224); hist(eq.flatten(), 256, range=(0,256)); show()


pointFrame = LabelFrame(interface,text="Point Transform operations", padx=32, pady=35)
pointFrame.place(x=20, y=120)
britCont = Button(pointFrame, text="Brightness & Contrast", command= lambda: adjBritCont(openImg.imagePath), width=21).grid(row=0,column=0)
histoEqui   = Button(pointFrame, text="Histogram & Equalization", command= lambda: histoAndEq(openImg.imagePath), width=21).grid(row=1,column=1)



def lHP(image, mode):
    
    kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
            [-1, 1, 2, 2, -1],
            [-1, 2, 4, 2, -1],
            [-1, 1, 2, 1, -1],
            [-1, -1, -1, -1, -1]])

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Original gray', img)
    k5 = ndimage.convolve(img, kernel_5x5)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hpf = img - blurred
    if (mode==0):
        cv2.imshow("5x5 Low-Pass filter", k5)
        
    elif (mode==1):   
        cv2.imshow("Original image - LPF = HPF", hpf)
    

def medianF(image):

    img = cv2.imread(image, 0)
    median = cv2.medianBlur(img,5)
    plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(median, cmap='gray'),plt.title('Median filter')
    plt.xticks([]), plt.yticks([])
    plt.show()
    


def averagingF(image):

    img = cv2.imread(image, 0)
    blur = cv2.blur(img,(5,5))
    plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur, cmap='gray'),plt.title('Averaging filter')
    plt.xticks([]), plt.yticks([])
    plt.show()


localFrame = LabelFrame(interface,text="Local Transform operations", padx=37, pady=10)
localFrame.place(x=431, y=120)
lowpass    = Button(localFrame, text="Low-pass filter", command=lambda: lHP(openImg.imagePath, 0), width=17).grid(row=0,column=0)
highpass   = Button(localFrame, text="High-pass filter", command=lambda: lHP(openImg.imagePath, 1), width=17).grid(row=1,column=0)
median     = Button(localFrame, text=" Median filter", command=lambda: medianF(openImg.imagePath), width=17).grid(row=2,column=0)
averaging  = Button(localFrame, text="Averaging filter", command=lambda: averagingF(openImg.imagePath), width=17).grid(row=3,column=0)



def edgeFilter(image, mode):
    
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

    #canny
    img_canny = cv2.Canny(img,100,200)

    #sobel
    img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
    img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
    img_sobel = img_sobelx + img_sobely


    #prewitt
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    #laplace
    ddepth = cv2.CV_16S
    kernel_size = 3
    img = cv2.GaussianBlur(img, (3, 3), 0)
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(dst)
    
    #zero crossing
    LoG = cv2.Laplacian(img, cv2.CV_16S)
    minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3,3)))
    maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3,3)))
    zeroCross = np.logical_or(np.logical_and(minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))

    cv2.imshow("Original Image", img)
    
    if(mode==7):
        
        cv2.imshow("Canny", img_canny)
        
    elif(mode==2):
        
        cv2.imshow("Sobel X", img_sobelx)
        
    elif(mode==3):
        
        cv2.imshow("Sobel Y", img_sobely)
        
    elif(mode==1):
        
        cv2.imshow("Sobel X+Y", img_sobel)
        
    elif(mode==4):
        
        cv2.imshow("Prewitt X", img_prewittx)
        
    elif(mode==5):
        
        cv2.imshow("Prewitt Y", img_prewitty)
        
    elif(mode==6):
        
        cv2.imshow("Prewitt X+Y", img_prewittx + img_prewitty)
        
    elif(mode==0):
        
        cv2.imshow('Lablassian', abs_dst)
    




edgeVar = IntVar()
edgeVar.set(12)
edgeDetectionFrame = LabelFrame(interface, text="Edge Detection Filters")
edgeDetectionFrame.place(x=20, y=280)

Radiobutton(edgeDetectionFrame, text="Laplacian",  variable= edgeVar, command= lambda: edgeFilter(openImg.imagePath, edgeVar.get()),   value=0, width=10).grid(row=0, column=0, padx=28, pady=10)
Radiobutton(edgeDetectionFrame, text="Sobel X+Y",  variable= edgeVar, command= lambda: edgeFilter(openImg.imagePath, edgeVar.get()),   value=1, width=10).grid(row=0, column=1, padx=28, pady=10)
Radiobutton(edgeDetectionFrame, text="Sobel X",  variable= edgeVar, command= lambda: edgeFilter(openImg.imagePath, edgeVar.get()),   value=2, width=10).grid(row=0, column=2, padx=28, pady=10)
Radiobutton(edgeDetectionFrame, text="Sobel Y",  variable= edgeVar, command= lambda: edgeFilter(openImg.imagePath, edgeVar.get()),   value=3, width=10).grid(row=0, column=3, padx=28, pady=10)
Radiobutton(edgeDetectionFrame, text="Prewitt X",  variable= edgeVar, command= lambda: edgeFilter(openImg.imagePath, edgeVar.get()),   value=4, width=10).grid(row=1, column=0, padx=28, pady=10)
Radiobutton(edgeDetectionFrame, text="Prewitt Y",  variable= edgeVar, command= lambda: edgeFilter(openImg.imagePath, edgeVar.get()),   value=5, width=10).grid(row=1, column=1, padx=28, pady=10)
Radiobutton(edgeDetectionFrame, text="Prewitt X+Y",  variable= edgeVar, command= lambda: edgeFilter(openImg.imagePath, edgeVar.get()),   value=6, width=10).grid(row=1, column=2, padx=28, pady=10)
Radiobutton(edgeDetectionFrame, text="Canny",  variable= edgeVar, command= lambda: edgeFilter(openImg.imagePath, edgeVar.get()),   value=7, width=10).grid(row=1, column=3, padx=28, pady=10)


def hLine(img):
    
    src = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    dst = cv2.Canny(src, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
    cv2.imshow("Original image (gray)", src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv2.waitKey()
    return 0
    

def hCir(img):
    image = cv2.imread(img,0)
    output = cv2.imread(img,1)
    cv2.imshow("Original image", image)
    blurred = cv2.GaussianBlur(image,(11,11),0)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
                             param1=100,param2=90,minRadius=0,maxRadius=200)
    
    if circles is not None:
        
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0,255,0), 3)
            cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0,255,0), -1)

    cv2.imshow("Circles detected",output)
    




globalFrame = LabelFrame(interface,text="Global Transform operations", padx=25, pady=37)
globalFrame.place(x=20, y=450)
line        = Button(globalFrame, text="Line detection using Hough Transform", command= lambda: hLine(openImg.imagePath), width=30).grid(row=0,column=0, pady=8)
circle      = Button(globalFrame, text="Circles detection using Hough Transform", command= lambda: hCir(openImg.imagePath), width=30).grid(row=1,column=0, pady=8)


def morphTrans(image, mode):
    img = cv2.imread(image,0)
    cv2.imshow('Original grey', img)
    kernel = np.ones((5,5),np.uint8)
    if(mode==0):
        dilate = cv2.dilate(img,kernel,iterations = 1)
        cv2.imshow('Dilation', dilate)

    elif(mode==1):
        erose = cv2.erode(img,kernel,iterations = 1)
        cv2.imshow('Erosion', erose)

    elif(mode==2):
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('Closing', closing)

    elif(mode==3):
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('Opening', opening)


morphFrame = LabelFrame(interface,text="Morphological operations", padx=22, pady=10)
morphFrame.place(x=325, y=450)
dilation = Button(morphFrame, text="Dilation", command= lambda: morphTrans(openImg.imagePath,0), width=20).grid(row=0,column=0, pady=4)
erosion  = Button(morphFrame, text="Erosion",  command= lambda: morphTrans(openImg.imagePath,1), width=20).grid(row=1,column=0, pady=4)
close    = Button(morphFrame, text="Close",    command= lambda: morphTrans(openImg.imagePath,2), width=20).grid(row=2,column=0, pady=3)
open    = Button(morphFrame, text="Open",     command= lambda: morphTrans(openImg.imagePath,3), width=20).grid(row=3,column=0, pady=4)



Exit = Button(interface, text = 'Exit', command = interface.quit,width=20).place(x=250,y=650)

interface.mainloop()
