
from PIL import Image
import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

import math

import json
#from region_growing import region_growing
def canny(path,save_path):
	path = path
	img = cv2.imread(path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 25, 255)
	#cv2.imshow('edges',edges)
	#cv2.waitKey(0)
	cv2.imwrite(save_path,edges)
	return save_path


def write(id1,left_det,right_det,flag):

	with open('train.csv', mode='a') as csv_file:
		if flag==True:
			fieldnames = ['id','deduction']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()
		
		deduction = 'applied'
		if 'released' in left_det or 'released' in right_det:
			deduction = 'released'


		fieldnames = ['id','deduction']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writerow({'id': id1,'deduction':deduction})
		print(json.dumps({'id': id1[0:6], 'status': deduction}))


def hough(path,save_path,flag=False):
	img = cv2.imread(path)
	#print((img))
	
	height,width = img.shape[:2]
	blank_image_2 = np.zeros((height,width,3), np.uint8)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 25, 255)

	#gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
	lines = cv2.HoughLinesP(edges,1,np.pi/270,75)
	if lines is not None:
		if len(lines) is not None:
			print(len(lines))
			b = "brake released"
			for i in range(len(lines)):
				for line in lines[i]:

				    pt1 = (line[0],line[1])
				    pt2 = (line[2],line[3])
				    if flag is False:
				    	cv2.line(blank_image_2, pt1, pt2, (255,255,255), 3)
				    dist = math.sqrt( (line[0] - line[2])**2 + (line[1] - line[3])**2 )
				    
				    if flag:
					    if dist > 0 :
					    	print(dist,'Distance')
					    	cv2.line(blank_image_2, pt1, pt2, (255,255,255), 3)

			#cv2.imshow('blank_image',blank_image_2)

			#cv2.waitKey()
			cv2.imwrite(save_path,blank_image_2)
			return save_path
	else:
		a = 'applied'
		return 0



def hough_two(path,save_path,flag=False):
	img = cv2.imread(path)
	#print((img))
	
	height,width = img.shape[:2]
	blank_image_2 = np.zeros((height,width,3), np.uint8)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 25, 255)
	deduction = 'released'
	#gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
	lines = cv2.HoughLinesP(edges,1,np.pi/270,65)
	if lines is not None:
		if len(lines) is not None:
			print(len(lines))
			b = "brake released"
			for i in range(len(lines)):
				for line in lines[i]:

				    pt1 = (line[0],line[1])
				    pt2 = (line[2],line[3])
				    dist = math.sqrt( (line[0] - line[2])**2 + (line[1] - line[3])**2)
				    
				    if dist >=5:
				    	print(dist,'Distance')
				    	cv2.line(blank_image_2, pt1, pt2, (255,255,255), 3)
				    	deduction = 'released'
				    	flag = True

			if flag is False:
				deduction = 'applied'
			cv2.imwrite(save_path,blank_image_2)
			return deduction
			
	else:
		a = 'applied'
		return a

'''

def SI_DI(path,save_path):
	img = cv2.imread(path)
	#img = img
	print (img.shape)
	h, w = img.shape[:2]

	# Drop top and bottom area of image with black parts.
	#img= img[100:h-100, :]
	#cv2.imwrite('/media/arjun/119GB/Railways/skeleton_save/kk.jpg',img)
	h, w = img.shape[:2]

	# Threshold image
	img_rgb = cv2.medianBlur(img, 5)
	#cv2.imshow('img_rgb',img_rgb)
	#cv2.waitKey(0)
	Conv_hsv_Gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	print(mask,'mask')
	#cv2.imwrite(save_path+'thr.jpg',mask)
	# get rid of thinner lines
	kernel = np.ones((2,2),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)
	#cv2.imwrite(save_path+'th.jpg',mask)
	# execute contour of all blobs found
	_, contours0, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
	print(len(contours) ,'1234')
	# Draw all contours
	perimeter=[]
	area=[]
	i=0
	a = np.zeros((2, 1, 2), np.uint8)
	vis = np.zeros((h, w, 3), np.uint8)
	kernel = np.ones((2,2),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)

	for cnt in contours:
	    #area.append(cv2.contourArea(cnt))
	    #print(area[i])

	    perimeter.append(cv2.arcLength(cnt,True))
	    area.append(cv2.contourArea(cnt))
	    cord=np.nonzero(mask)
	    varx=np.var(cord[0] )
	    vary=np.var(cord[1])
	    v=np.sqrt(varx+vary)
	    DI=np.sqrt(area)/(1+v)
	    
	    SI=perimeter/((4*np.sqrt(area))+1)
	    

	    if SI[i] >1  :
	        print(SI[i],'11')
	        print(cnt)
	        cv2.drawContours( vis, contours, i, (255,255,255), 1, cv2.LINE_AA)
	        #cv2.fillPoly(vis, pts =[cnt], color=(255,255,255))
	    cv2.imwrite(save_path+'con.jpg',vis)
	    i+=1
'''

def skeleton(path,save_path):
	#img = img
	img = cv2.imread(path,0)
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)
	 
	ret,img = cv2.threshold(img,127,255,0)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False
	 
	while( not done):
	    eroded = cv2.erode(img,element)
	    temp = cv2.dilate(eroded,element)
	    temp = cv2.subtract(img,temp)
	    skel = cv2.bitwise_or(skel,temp)
	    img = eroded.copy()
	 
	    zeros = size - cv2.countNonZero(img)
	    if zeros==size:
	        done = True
	 
	#cv2.imshow("skel",skel)
	cv2.imwrite(save_path,skel)
	
	#cv2.waitKey(0)
	return save_path




def json_result(id1, left_det, right_det):
        deduction = 'applied'

        if 'released' in left_det or 'released' in right_det:
                 deduction = 'released'

        result = {'id': id1[0:6], 'status': deduction}
        return result



def execute(path,item,side):
			print(path + item)
			a = 'released'
			canny2 = canny(path,'/media/arjun/119GB/Railways/compiled/canny_applied'+side+item)
			hough2 = hough(canny2,'/media/arjun/119GB/Railways/compiled/first_hough'+side+item)
			if hough2 is not 0:
				skeleton2 = skeleton(hough2,'/media/arjun/119GB/Railways/compiled/skel'+side+item)
				#resized = cv2.resize('/media/arjun/119GB/Railways/compiled/first_hough'+side+item,(256,256))
				#region = region_growing(resized,seed = (0,100))
				#cv2.imshow('region',region)
				cv2.waitKey()
				hough2 = hough_two(skeleton2,'/media/arjun/119GB/Railways/compiled/second_hough'+side+item)
				a = hough2
				
			else:
				print('applied')
				a = 'applied'
			return a
			#hough2 = hough(skeleton2,'/media/arjun/119GB/Railways/compiled/third_hough.jpg')

			#contours = SI_DI(skeleton2,'/media/arjun/119GB/Railways/compiled/')
def main():
	dirs = os.listdir( sys.argv[1] )
	i=0
	flag= True
	for item in dirs:
		im = sys.argv[1] + item
		if i>0:
			flag = False

		if 'C01A02' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (700, 300, 800, 800) # applied A03
			cropped_im = im.crop(crop_rectangle)
			side = 'left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1200, 300, 1295, 800)
			cropped_im = im.crop(crop_rectangle)
			side = 'right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			i+=1

		elif 'C01A01' in im:
			im = Image.open(sys.argv[1]+item)
			#crop_rectangle = (600, 0, 800, 550) # applied A03
			#crop_rectangle = (260, 100, 300, 300)
			crop_rectangle = (650, 200, 750, 700)
			cropped_im = im.crop(crop_rectangle)
			side = 'left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			#crop_rectangle = (1200, 150, 1300, 850)
			#crop_rectangle = (425, 100, 460, 300)# applied B02
			crop_rectangle = (1100, 200, 1175, 650)# applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)	
			i+=1

		elif 'C01A03' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (575, 250, 675, 700) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side = 'left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1150, 250, 1250, 700)
			cropped_im = im.crop(crop_rectangle)
			side = 'right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			i+=1

		elif 'C01A04' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (675, 500, 750, 900) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side = 'left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1150, 500, 1300, 800)
			side = 'right'
			cropped_im = im.crop(crop_rectangle)
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			i+=1

		elif 'C01B01' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (725, 200, 800, 650) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side = 'left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1160, 200, 1250, 650)
			cropped_im = im.crop(crop_rectangle)
			side = 'right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			i+=1

		elif 'C01B02' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (700, 200, 800, 650) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1200, 200, 1300, 650)
			cropped_im = im.crop(crop_rectangle)
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			side = 'right'
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			i+=1

		elif 'C01B03' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (675, 200, 775, 650) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1200, 140, 1285, 740) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			i+=1

		elif 'C01B04' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (625, 300, 700, 650) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1100, 200, 1150, 650) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			i+=1
		
		elif 'C02A01' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (550, 250, 650, 750)# applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1000, 250, 1050, 750) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			result1 = json_result(item,left,right)	
			i+=1
	
		elif 'C02A02' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (575, 150, 650, 600)#applied B02
			cropped_im = im.crop(crop_rectangle)
			side= 'left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1025, 150, 1125, 600) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side,)
			write(item,left,right,flag)
			result2 = json_result(item,left,right)	
			i+=1

		elif 'C02A03' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (750, 100, 825, 600)# applied B02 # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1150, 100, 1225, 600) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			result3 = json_result(item,left,right)	
			i+=1

		elif 'C02A04' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (750, 200, 850, 800)# applied B02 # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1250, 100, 1375, 800)# applied B02  # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			result4 = json_result(item,left,right)	
			i+=1

		elif 'C02B01' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (900, 200, 1000, 600)#applied B0
			cropped_im = im.crop(crop_rectangle)
			side = 'left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1300, 200, 1400, 600)  # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			result5 = json_result(item,left,right)	
			i+=1

		elif 'C02B02' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (575, 100, 650, 600)#applied B0
			cropped_im = im.crop(crop_rectangle)
			side ='left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1050, 100, 1150, 600)  # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			result6 = json_result(item,left,right)
			i+=1

		elif 'C02B03' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (800, 100, 900, 600)#pplied B02
			cropped_im = im.crop(crop_rectangle)
			side ='left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1250, 100, 1350, 600)  # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			result7 = json_result(item,left,right)
			i+=1

		elif 'C02B04' in im:
			im = Image.open(sys.argv[1]+item)
			crop_rectangle = (600, 100, 700, 600)#pplied B02
			cropped_im = im.crop(crop_rectangle)
			side ='left'
			cropped_im.save(sys.argv[2]+'cropped_applied_left_'+item)
			left = execute(sys.argv[2]+'cropped_applied_left_'+item,item,side)
			crop_rectangle = (1050, 100, 1150, 600) # applied B02
			cropped_im = im.crop(crop_rectangle)
			side ='right'
			cropped_im.save(sys.argv[2]+'cropped_applied_right_'+item)
			right = execute(sys.argv[2]+'cropped_applied_right_'+item,item,side)
			write(item,left,right,flag)
			result8 = json_result(item,left,right)
			i+=1
		else:
			print("Unknown")
			i+=1




if __name__ == '__main__':
	main()