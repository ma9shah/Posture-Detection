import cv2
import math
import time
import numpy as np
import util
import os    
import glob
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model




tic=0
# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def process (input_image, params, model_params):
	''' Start of finding the Key points of full body using Open Pose.'''
	oriImg = cv2.imread(input_image)  # B,G,R order
	multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
	heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
	paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
	for m in range(1):
		scale = multiplier[m]
		imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
		imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
		input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
		output_blobs = model.predict(input_img)
		heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
		heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
		heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
		heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
		paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
		paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
		paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
		paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
		heatmap_avg = heatmap_avg + heatmap / len(multiplier)
		paf_avg = paf_avg + paf / len(multiplier)

	all_peaks = [] #To store all the key points which are detected.
	peak_counter = 0
	
	#prinfTick(1) #prints time required till now.

	for part in range(18):
	    map_ori = heatmap_avg[:, :, part]
	    map = gaussian_filter(map_ori, sigma=3)

	    map_left = np.zeros(map.shape)
	    map_left[1:, :] = map[:-1, :]
	    map_right = np.zeros(map.shape)
	    map_right[:-1, :] = map[1:, :]
	    map_up = np.zeros(map.shape)
	    map_up[:, 1:] = map[:, :-1]
	    map_down = np.zeros(map.shape)
	    map_down[:, :-1] = map[:, 1:]

	    peaks_binary = np.logical_and.reduce(
	        (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
	    peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
	    peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
	    id = range(peak_counter, peak_counter + len(peaks))
	    peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

	    all_peaks.append(peaks_with_score_and_id)
	    peak_counter += len(peaks)

	connection_all = []
	special_k = []
	mid_num = 10

	#prinfTick(2) #prints time required till now.
	print()
	degrees, position = checkPosition(all_peaks) #check position of spine.
	canvas1 = draw(input_image,all_peaks) #show the image.
	return degrees, position


def draw(input_image, all_peaks):
    canvas = cv2.imread(input_image)  # B,G,R order
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    return canvas


def checkPosition(all_peaks):
    try:
        t = 1
        degree = [0,0,0,0,0]
        f = 0
        if (all_peaks[16]):
            a = all_peaks[16][0][0:2] #Right Ear
            f = 1
        else:
            a = all_peaks[17][0][0:2] #Left Ear
        b = all_peaks[11][0][0:2] # Hip
        angle = calcAngle(a,b)
        degrees = round(math.degrees(angle))

        if (f):
            degrees = 180 - degrees    
        print("Degree: " + str(degrees))    
        if (degrees<80):
            t = 0 
        else:
            t = 1
        return degrees,t   

    except Exception as e:
        print("person not in lateral view and unable to detect ears or hip")
        return 0,2

#calculate angle between two points with respect to x-axis (horizontal axis)
def calcAngle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if (ax == bx):
            return 1.570796
        return math.atan2(by-ay, bx-ax)
    except Exception as e:
        print("unable to calculate angle")


	




def showimage(img): #sometimes opencv will oversize the image when using using `cv2.imshow()`. This function solves that issue.
    screen_res = 1280, 720 #my screen resolution.
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def prinfTick(i): #Time calculation to keep a trackm of progress
    toc = time.time()
    print ('processing time%d is %.5f' % (i,toc - tic))        


if __name__ == '__main__': #main function of the program
    img_dir = "./sample_images/slouch" # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path) 
    a = [0,0,0]
    degree = [0,0,0,0,0]
    for f1 in files:
        tic = time.time()
        print('start processing...')
        print(f1)
        model = get_testing_model()
        model.load_weights('./model/keras/model.h5')
        vi=False
        if(vi == False):
            time.sleep(2)
            params, model_params = config_reader()
            degrees, position = process(f1, params, model_params)
            print(position)
            if degrees <= 65:
                degree[0]+=1
            elif degrees in range(66,71):
                degree[1]+=1
            elif degrees in range(71,76):
                degree[2]+=1
            elif degrees in range(76,81):
                degree[3]+=1
            else:
                degree[4]+=1                

            if (position == 0):
                a[0]+=1
            elif position == 1:
                a[1]+=1
            else:
                a[2]+=1        
    print("slouch")
    print(a)
    print("degrees")
    print(degree)
    
    
    #showimage(canvas)
        
        
