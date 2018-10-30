import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

images = glob.glob('camera_cal/calib*.jpg')

def calibrateCameraAndSaveValues():

    objpoints = []
    imgpoints = []

    objP = np.zeros((6*9, 3), np.float32)
    objP[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) #x, y coordinates

    for fname in images:
        # Read in each image
        img = mpimg.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # if corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objP)

            # draw and display the corners
            #cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            #plt.imshow(img)
            #plt.show()

    #Camera calibration, given object points, image points, and the shape of the grayscale image:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the camera calibration result for later use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle['objpoints'] = objpoints
    dist_pickle['imgpoints'] = imgpoints
    pickle.dump(dist_pickle, open("camera_cal/camera_cal_pickle.p", "wb"))

    return mtx, dist