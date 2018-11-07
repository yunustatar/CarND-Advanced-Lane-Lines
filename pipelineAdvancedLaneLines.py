import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from calibrateCameraAndSaveValues import *

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        # 3) Take the absolute value of the derivative or gradient
        abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    elif orient == 'y':
        # 3) Take the absolute value of the derivative or gradient
        abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def sobelxAndSColour(image):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    LAB_l_channel = lab[:, :, 0]
    LAB_a_channel = lab[:, :, 1]
    LAB_b_channel = lab[:, :, 2]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    HSV_h_channel = hsv[:, :, 0]
    HSV_s_channel = hsv[:, :, 1]
    HSV_v_channel = hsv[:, :, 2]

    RGB_r_channel = image[:, :, 0]
    RGB_g_channel = image[:, :, 1]
    RGB_b_channel = image[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sobel x
    #sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobelx = cv2.Sobel(LAB_l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x

    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 240
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 80
    s_thresh_max = 250
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    #f.tight_layout()
    #ax1.imshow(color_binary,)
    #ax1.set_title('Colour binary', fontsize=50)
    #ax2.imshow(combined_binary, cmap='gray')
    #ax2.set_title('Combined binary', fontsize=50)
    #if __debug__:
    #    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #    plt.show()

    #return color_binary, combined_binary
    return combined_binary


def corners_unwarp(img, chess, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Define calibration box in source and destination coordinates
    img_size = (img.shape[1], img.shape[0])

    # 1) Undistort using mtx and dist
    #dst = cv2.undistort(img, mtx, dist, None, mtx)
    #plt.imshow(dst)
    #plt.show()

    # 2) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(gray, cmap = 'gray')
    #plt.show()

    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(chess, (nx, ny), None)

    # 4) If corners found:
    if ret == True:
        offset = 10
        # a) draw corners
        #cv2.drawChessboardCorners(chess, (nx, ny), corners, ret)
        #plt.imshow(chess)
        #plt.show()
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])

        width, height = img.shape[1], img.shape[0]

        bottom_left = [260, 669]
        top_left = [530, 492]
        top_right = [760, 492]
        bottom_right = [1040, 669]

        left_offset = 320.0
        right_offset = (width - bottom_right[0]) / np.float32(bottom_left[0]) * left_offset
        top_offset = 470.0
        bottom_offset = 1.0

        bottom_left_dst = [left_offset, height - bottom_offset]
        top_left_dst = [left_offset, top_offset]
        top_right_dst = [width - right_offset, top_offset]
        bottom_right_dst = [width - right_offset, height - bottom_offset]

        src = np.array([bottom_left, top_left, top_right, bottom_right], np.float32)
        dst = np.array([bottom_left_dst, top_left_dst, top_right_dst, bottom_right_dst], np.float32)


    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M


def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0] // 2:, :]

    # Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    #return out_img
    return left_fit, right_fit


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped, left_fit, right_fit):
#def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.show()
    ## End visualization steps ##

    #return result
    return ploty, left_fitx, right_fitx


def measure_curvature_pixels(ploty, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    #ploty, left_fit, right_fit = generate_data()

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curverad, right_curverad


def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    radius_of_curvature = round(np.mean([left_curverad, right_curverad]),0)

    return left_curverad, right_curverad, radius_of_curvature

def second_ord_poly(line, val):
    '''
    Simple function being used to help calculate distance from center.
    Only used within Draw Lines below. Finds the base of the line at the
    bottom of the image.
    '''
    a = line[0]
    b = line[1]
    c = line[2]
    formula = (a*val**2)+(b*val)+c

    return formula


def pipeline(image):
    test_image6 = mpimg.imread('test_images/test6.jpg')
    #image = mpimg.imread('test_images/straight_lines1.jpg')
    chess = mpimg.imread('camera_cal/calibration2.jpg')

    # Run this only once and save calibration data
    #mtx, dist = calibrateCameraAndSaveValues()

    dist_pickle = pickle.load(open("camera_cal/camera_cal_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    undistorted_chess = cv2.undistort(chess, mtx, dist, None, mtx)
    cv2.imwrite('output_images/undistorted_chess_output.png', undistorted_chess)

    undistorted_test_image6 = cv2.undistort(test_image6, mtx, dist, None, mtx)
    cv2.imwrite('output_images/undistorted_test_image6.png', undistorted_test_image6)

    undistorted = cv2.undistort(image, mtx, dist, None, mtx)

    if __debug__:
        plt.imshow(undistorted)
        plt.show()

    thresholded = sobelxAndSColour(undistorted)

    # Gradient combinations
    ksize = 17  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(undistorted, orient='x', sobel_kernel=ksize, thresh=(20, 200))
    grady = abs_sobel_thresh(undistorted, orient='y', sobel_kernel=ksize, thresh=(20, 200))
    mag_binary = mag_thresh(undistorted, sobel_kernel=ksize, mag_thresh=(80, 120))
    if __debug__:
        plt.imshow(mag_binary)
        plt.show()
    dir_binary = dir_threshold(undistorted, sobel_kernel=ksize, thresh=(1.2, np.pi/2))
    if __debug__:
        plt.imshow(dir_binary)
        plt.show()

    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 0))] = 1
    #combined[(gradx == 1) | (thresholded) | ((mag_binary == 1) & (dir_binary == 1))] = 1


    combinedWithColour = np.zeros_like(dir_binary)
    combinedWithColour[(thresholded == 1) & (dir_binary == 0) & (mag_binary == 0)] = 1
    out_img = np.dstack((combinedWithColour, combinedWithColour, combinedWithColour)) * 255
    cv2.imwrite('output_images/combinedWithColour_image.jpg', out_img)
    if __debug__:
        plt.imshow(combinedWithColour)
        plt.show()

    nx = 9
    ny = 6

    #binary_warped, M = corners_unwarp(thresholded, chess, nx, ny, mtx, dist)
    binary_warped, M = corners_unwarp(combinedWithColour, chess, nx, ny, mtx, dist)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    cv2.imwrite('output_images/binary_warped.png', out_img)

    if __debug__:
        plt.imshow(binary_warped)
        plt.show()

    out_img = fit_polynomial(binary_warped)

    left_fit, right_fit = fit_polynomial(binary_warped)

    # Run image through the pipeline
    ploty, left_fitx, right_fitx = search_around_poly(binary_warped, left_fit, right_fit)

    mid_point_x = 0.5 * (right_fitx + left_fitx)

    #calculate lane width dynamically
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    ms_per_px = 0.5 * xm_per_pix + 0.5 * 3.7/(right_fitx - left_fitx)

    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    left_curverad, right_curverad, radius_of_curvature = measure_curvature_real(ploty, left_fit_cr, right_fit_cr)

    # Calculating middle of the image, aka where the car camera is
    middle_of_image = image.shape[1] / 2
    car_position = middle_of_image * xm_per_pix

    # Calculating middle of the lane
    left_line_base = second_ord_poly(left_fit_cr, image.shape[0] * ym_per_pix)
    right_line_base = second_ord_poly(right_fit_cr, image.shape[0] * ym_per_pix)
    lane_mid = (left_line_base + right_line_base) / 2

    # Calculate distance from center and list differently based on left or right
    dist_from_center = lane_mid - car_position
    if dist_from_center >= 0:
        center_text = "{} meters left of center".format(round(dist_from_center, 2))
    else:
        center_text = "{} meters right of center".format(round(-dist_from_center, 2))

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = np.linalg.inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    # add annotation for radius and offset to image:
    cv2.putText(result, 'Lane radius : {:2.2f}m'.format(radius_of_curvature),
                (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Vehicle Offset : {}'.format(center_text),
                (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

    if __debug__:
        plt.imshow(result)
        plt.show()
    return result


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_image(image):
    """ This processes through everything above.
    Will return the image with car position, lane curvature, and lane lines drawn.
    """
    result = pipeline(image)

    return result

# Convert to video
# vid_output is where the image will be saved to
#vid_output = 'challenge_video_output.mp4'
vid_output = 'project_video_output.mp4'

# The file referenced in clip1 is the original video before anything has been done to it
clip1 = VideoFileClip("project_video.mp4")
#clip1 = clip1.subclip(0, 1)
# NOTE: this function expects color images
vid_clip = clip1.fl_image(process_image)
vid_clip.write_videofile(vid_output, audio=False)

'''
#Grab some frames for testing
clip1.save_frame("frame0_challenge.jpeg", t='00:00:00') # frame at time t=1h
clip1.save_frame("frame1_challenge.jpeg", t='00:00:01') # frame at time t=1h
clip1.save_frame("frame2_challenge.jpeg", t='00:00:02') # frame at time t=1h
clip1.save_frame("frame3_challenge.jpeg", t='00:00:03') # frame at time t=1h
clip1.save_frame("frame4_challenge.jpeg", t='00:00:04') # frame at time t=1h
clip1.save_frame("frame5_challenge.jpeg", t='00:00:05') # frame at time t=1h
clip1.save_frame("frame6_challenge.jpeg", t='00:00:06') # frame at time t=1h
clip1.save_frame("frame7_challenge.jpeg", t='00:00:07') # frame at time t=1h
clip1.save_frame("frame8_challenge.jpeg", t='00:00:08') # frame at time t=1h
clip1.save_frame("frame9_challenge.jpeg", t='00:00:09') # frame at time t=1h
clip1.save_frame("frame10_challenge.jpeg", t='00:00:10') # frame at time t=1h
clip1.save_frame("frame11_challenge.jpeg", t='00:00:11') # frame at time t=1h
clip1.save_frame("frame12_challenge.jpeg", t='00:00:12') # frame at time t=1h
clip1.save_frame("frame13_challenge.jpeg", t='00:00:13') # frame at time t=1h
clip1.save_frame("frame14_challenge.jpeg", t='00:00:14') # frame at time t=1h
clip1.save_frame("frame15_challenge.jpeg", t='00:00:15') # frame at time t=1h
clip1.save_frame("frame16_challenge.jpeg", t='00:00:16') # frame at time t=1h
'''

'''
#Test on images
#images = glob.glob('test_images/*.jpg')
#images = glob.glob('test_images/challenge/*.jpeg')
input = mpimg.imread('test_images/test2.jpg')
i = 0
#for fname in images:
#input = mpimg.imread(fname)
output = pipeline(input)
cv2.imwrite('output_images/test2_output.jpg', output)
'''
