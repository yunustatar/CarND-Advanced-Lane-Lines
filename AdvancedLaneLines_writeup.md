
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_chess_output.png "Undistorted"
[image2]: ./output_images/undistorted_test_image6.png "Road Transformed"
[image3]: ./output_images/combinedWithColour_image.jpg "Binary Example"
[image4]: ./output_images/binary_warped.png "Warp Example"
[image6]: ./output_images/test2_output.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I've used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 519 through 542 in `pipelineAdvancedLaneLines.py`).  

For the Sobel thresholding, I've used the S channel from HLS colour space and I've also used L channel from LAB colour space. Then I've tried different combinations of magnitude and direction thresholding. I've ended up only using inverse of direction threshold and inverse of magnitude combined with the Sobel threshold previously mentioned. Threshold values have been determined empirically.
Here's an example of my output for this step

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination                                  | 
|:-------------:|:-------------:                               | 
| 530, 492      | left_offset, top_offset                      | top left
| 260, 669      | left_offset, height - bottom_offset          | bottom left
| 1040, 669     | width - right_offset, height - bottom_offset | bottom right
| 760, 492      | width - right_offset, top_offset             | top right

Here is a warped and perspective transformed binary image:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

find_lane_pixels function (line 219) is used for finding the lane pixels.

Then I've used search_around_poly and second_ord_poly to fit my lane lines with a 2nd order polynomial.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I'e used measure_curvature_real from the quiz to calculate curvature. This is on lines 430 through 450 in my code in `pipelineAdvancedLaneLines.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 532 through 589 in my code in `pipelineAdvancedLaneLines.py ` in the function `pipeline()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I've used HLS and LAB colour spaces for thresholding but potentially other colour spaces can be utilized for different cases. Also magnitude and direction thresholding can potentially be further fine tuned.

My pipeline will likely fail if there are strong shae transitions or there are asphalt lines (like in the challenge video). Filtering based on predicted positioning of the lanes can be used to mitigate this, as such the detected lines would be filtered out if they do not fall into the narrow area where the lane lines are expected to be.
