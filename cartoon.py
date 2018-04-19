import cv2
import numpy as np
from os import walk

class Cartoonizer:
    """Cartoonizer effect
        A class that applies a cartoon effect to an image.
        The class uses a bilateral filter and adaptive thresholding to create
        a cartoon effect.
    """
    def __init__(self):
        pass

    def replacecolorwithanother(self, img,color1,color2):
        im = cv2.imread(img)
        im[np.where((im != color1).all(axis = 2))] = color2
        #im[np.where((im==[0]).all(axis=1))] = [255]
        cv2.imwrite('output.png', im)
        
    def drawHead(self, file):
        im = cv2.imread(file)
        cv2.ellipse(im,(256,256),(120,70),0,0,360,255,-1)
        cv2.imwrite('output.png', im)
        

    def detectMultipleColors(self, img):
        img = cv2.imread(img)
        #converting frame(img i.e BGR) to HSV (hue-saturation-value)

        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        #definig the range of red color
        red_lower=np.array([136,87,111],np.uint8)
        red_upper=np.array([180,255,255],np.uint8)

        #defining the Range of Blue color
        blue_lower=np.array([99,115,150],np.uint8)
        blue_upper=np.array([110,255,255],np.uint8)

        #defining the Range of black color
        black_lower=np.array([0,0,0],np.uint8)
        black_upper=np.array([180, 255, 50],np.uint8)
        
        #defining the Range of yellow color
        yellow_lower=np.array([22,60,200],np.uint8)
        yellow_upper=np.array([60,255,255],np.uint8)

        #finding the range of red,blue and yellow color in the image
        red=cv2.inRange(hsv, red_lower, red_upper)
        blue=cv2.inRange(hsv,blue_lower,blue_upper)
        yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)
        black=cv2.inRange(hsv,black_lower,black_upper)
        
        #Morphological transformation, Dilation     
        kernal = np.ones((5 ,5), "uint8")

        red=cv2.dilate(red, kernal)
        res=cv2.bitwise_and(img, img, mask = red)

        blue=cv2.dilate(blue,kernal)
        res1=cv2.bitwise_and(img, img, mask = blue)

        yellow=cv2.dilate(yellow,kernal)
        res2=cv2.bitwise_and(img, img, mask = yellow)

        black = cv2.dilate(black,kernal)
        res3=cv2.bitwise_and(img, img, mask = black)

        ## final mask and masked
        maskgen = cv2.bitwise_or(red, black)
        #maskgen = cv2.bitwise_or(maskgen, yellow) 
        res4=cv2.bitwise_and(img, img, mask = maskgen)
        cv2.imwrite('output_colors.png', res4)


        #Tracking the Red Color
        (_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                
                x,y,w,h = cv2.boundingRect(contour) 
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(img,"RED color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
                
        #Tracking the Blue Color
        (_,contours,hierarchy)=cv2.findContours(black,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour) 
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img,"Black color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))

        #Tracking the yellow Color
        #(_,contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #for pic, contour in enumerate(contours):
        #    area = cv2.contourArea(contour)
        #    if(area>300):
        #        x,y,w,h = cv2.boundingRect(contour) 
        #        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #        cv2.putText(img,"yellow  color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
        cv2.imwrite('output_colors_detecting.png', img) 
                    
           


    def render(self, img_rgb):
        img_rgb = cv2.imread(img_rgb)
        img_rgb = cv2.resize(img_rgb, (1366,768))
        numDownSamples = 2       # number of downscaling steps
        numBilateralFilters = 50  # number of bilateral filtering steps

        # -- STEP 1 --
        # downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)
        #cv2.imshow("downcolor",img_color)
        #cv2.waitKey(0)
        # repeatedly apply small bilateral filter instead of applying
        # one large filter
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        #cv2.imshow("bilateral filter",img_color)
        #cv2.waitKey(0)
        # upsample image to original size
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)
        #cv2.imshow("upscaling",img_color)
        #cv2.waitKey(0)
        # -- STEPS 2 and 3 --
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        #remove noise
        img_blur = cv2.medianBlur(img_gray, 3)
        #cv2.imshow("grayscale+median blur",img_color)
        #cv2.waitKey(0)
        # -- STEP 4 --
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        #img_edge = cv2.adaptiveThreshold(img_blur, 255,
        #                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                 cv2.THRESH_BINARY, 9, 2)
        #cv2.imshow("edge",img_edge)
        #cv2.waitKey(0)

        # -- STEP 5 --
        # convert back to color so that it can be bit-ANDed with color image
        (x,y,z) = img_color.shape
        img_edge = cv2.resize(img_edge,(y,x)) 
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        # cv2.imwrite("edge.png",img_edge)
        #cv2.imshow("step 5", img_edge)
        #cv2.waitKey(0)
        #img_edge = cv2.resize(img_edge,(i for i in img_color.shape[:2]))
        #print img_edge.shape, img_color.shape
        return cv2.bitwise_and(img_color, img_edge)

    #draws head with eyes using the very first image
    #corners calcuulate using frames and eyebrows positions
    def drawHead(self, backgroundImg, conturs_coordinates):
        #draw head
        background = cv2.imread(backgroundImg)
        leftEyeBrowX = conturs_coordinates[0][0]
        leftEyeBrowY = conturs_coordinates[0][1]
        rightEyeBrowX = conturs_coordinates[1][0]
        rghtEyeBrowY = conturs_coordinates[1][1]
        if leftEyeBrowX > rightEyeBrowX:
            leftEyeBrowX = conturs_coordinates[1][0]
            leftEyeBrowY = conturs_coordinates[1][1]
            rightEyeBrowX = conturs_coordinates[0][0]
            rghtEyeBrowY = conturs_coordinates[0][1]

        eyeBrowsAngle = (rghtEyeBrowY - leftEyeBrowY)/2
        #pre-last param is a color constant
        #draw head in bgr
        headColor = (192, 128, 0)
        cv2.ellipse(background,(leftEyeBrowX + 92, leftEyeBrowY + 50),(130,80),eyeBrowsAngle,0,360,headColor,-1)
        eyeColor = (210,210,210)
        #draw eyes 
        eyeHeight = 20
        pupilColor = (0, 0, 0)
        cv2.ellipse(background,(leftEyeBrowX + 32, leftEyeBrowY + 38),(30,eyeHeight), eyeBrowsAngle,0,360, eyeColor, -1)
        cv2.ellipse(background,(leftEyeBrowX + 30, leftEyeBrowY + 37),(10,10),0,0,360, pupilColor, -1)
        cv2.ellipse(background,(rightEyeBrowX + 32, rghtEyeBrowY + 38),(30,eyeHeight), eyeBrowsAngle,0,360, eyeColor, -1)
        cv2.ellipse(background,(rightEyeBrowX + 30, rghtEyeBrowY + 37),(10,10),0,0,360, pupilColor, -1)
        #draw nose
        noseColor = (0, 0, 204)
        cv2.ellipse(background,(leftEyeBrowX + 30 + 63, leftEyeBrowY + 37 + 25),(10,10),0,0,360, noseColor, -1)
        return background
    
    def getCoordinates(self, img, lower_color, upper_color ):
        frame = cv2.imread(img)
        frame = cv2.resize(frame, (700,300))
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_color, upper_color)
        kernal = np.ones((5 ,5), "uint8")
        color=cv2.dilate(mask, kernal)
        
        geometricParams = []
        (_, contours, hierarchy) = cv2.findContours(color,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour)
                geometricParams.append((x,y,w,h))
        return geometricParams

    
    # Where to copy, from
    def copyContour(self, background, img, lower_color, upper_color,  color2fill):
        frame = cv2.imread(img)
        frame = cv2.resize(frame, (700,300))
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Threshold the HSV image to get only blue colors
        bluemask = cv2.inRange(hsv, lower_color, upper_color)
        kernal = np.ones((5 ,5), "uint8")
        color=cv2.dilate(bluemask, kernal)
        
        #find contour of the blue color
        (_, contours, hierarchy) = cv2.findContours(color,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #draw blue contour onto the backgound image
        cv2.drawContours(background,contours,-1,color2fill,-1)#last arg is thickness f outline
        #res = cv2.bitwise_and(frame,frame, mask= mask)
        output_img = img.replace('vika_images','output_images')
        print("saving: " + output_img);
        cv2.imwrite(output_img, background)

tmp_canvas = Cartoonizer()
background_file = "background.jpg"
#file_names = ["output_0001.png", "output_0039.png"]

mypath = 'vika_images'
file_names = []
for (dirpath, dirnames, filenames) in walk(mypath):
    file_names.extend(filenames)
    break

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

#defining the range of red color
#[136,87,111]
red_lower=np.array([136,145,121],np.uint8)#saturation is contrast
red_upper=np.array([180,255,255],np.uint8)

for file_name in file_names:
    file_name = mypath + '//' + file_name
    print("processing: {}".format(file_name))
    conturs_coordinates = tmp_canvas.getCoordinates(file_name, lower_blue, upper_blue )
    print(conturs_coordinates)
    background = tmp_canvas.drawHead(background_file, conturs_coordinates)
    tmp_canvas.copyContour(background, file_name, lower_blue, upper_blue, (0,0,0))
    tmp_canvas.copyContour(background, file_name, red_lower, red_upper, (0,0,204))#bgr




cv2.waitKey(0)
cv2.destroyAllWindows()
