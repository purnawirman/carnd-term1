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

[image1]: ./output_images/test_undistort6.jpg "Undistorted chessboard"
[image2]: ./output_images/test_undistort1.jpg "Undistorted image"
[image3]: ./output_images/test_threshold1.jpg "Thresholded binary image"
[image4]: ./output_images/test_calibrate_perspective1.jpg "Calibrate perspective"
[image5]: ./output_images/test_perspective_transform1.jpg "perspective transform"
[image6]: ./output_images/test_fit_first.jpg "test fit first"
[image7]: ./output_images/test_next_fit1.jpg "test fit first"
[image8]: ./output_images/test_project_back1.jpg "test project back"
[image9]: ./output_images/project_video_roc.jpg "roc"
[image10]: ./output_images/project_video_center.jpg "center"

<!-- 
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video" -->

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.   

Ok.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All the codes are contained in './submission.ipynb'. For this section, look at the codes in section '1. Calibrate camera'.

First we calibrate the camera by using the chessboard images provided. I use a function `calibrate_camera(images, nx, ny)` where images are the input files of chessboard images, nx and ny are the number of chessboard corners to detect. The output of the function is to obtain objpoints and imgpoints, where objpoints denote corners in 3D world space, and imgpoints in 2D space image space. Similar to lecture, the detection is done by `cv2.findChessboardCorners()` function, and if detected, the pair of objpoint and imgpoint are attached to the output lists.

We then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image in function `cal_undistort()`,  using the `cv2.undistort()` function and obtained this result for undistorted chessboard: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The corrections are more obvious looking at the edge of the image. See below:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

See codes in section '2. Color and gradient threshold'.

I used a function 'threshold_image(img, s_thresh, sx_thresh)' to perform a binary image thresholding. The input parameters are img as the input image matrix, s_thresh and sx_thres are threshold values for color (for S in HLS space) and gradient along x direction. Combining both of these threshold to obtain the final binary image thresholding. The threshold parameters are picked by trial and errors, but the numbers are mostly inspired from the lecture. Below is the example of the thresholded binary image.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The codes are in section '3. Perspective transform'.

First, we calibrate the perspective transform by searching for the calibration matrix M. The calibration is done in function 'calibrate_perspective(img, objpoints, imgpoints)'. The image is first undistorted by camera calibration. Then we use a polygon that consists of 4 points in the original camera image (src) and 4 points in the would-be transformed top-down perspective image (dst). The calibration matrix can be obtained by the transformation of src to dst, otherwise for the inverse transformation from dst to src. I choose to hardcode the points in src and dst by the following manner

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 45, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1111, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

Once the perspective transformation matrix M is obtained, we can use this to perform transformation on other images. I used the function 'perspective_transform(img, M, objpoints, imgpoints)' to combine all the pipeline so far. It starts with thresholding, followed by distortion correction, and then perspective transformation. One example of such transformation is shown below:

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Codes are in section '4. Fit line'.

There are two main functions used to fit line, 'fit_first_binary_warped(binary_warped)' and 'fit_next_binary_warped(binary_warped, left_fit, right_fit)'. The former function is to fit the lane by using information just from a thresholded, and perspective transformed binary image, and the latter to fit the lane by using the knowledge of fitting in previous frame.

The first fit use a sliding window approach to identify peak intensities in the binary image. The binary warped image is divided into nwindows=10 sections in y direction. Each section has width of 2 x margin (100 pixels). Inside the window, we identify the (x, y) of the lane position by taking average of the nonzero pixels. These (x,y) are recorded per iteration from one window to the next, by shifting the window position in x necessarily to ensure it always capture the lane pixels.

Once the (x,y) are obtained for all windows in the first frame, we can find the quadratic polynomial fit of (x,y) by using 'np.polyfit(y, x, 2)'. We return this as ones of the output of the function to provide a good starting point for 'fit_next_binary_warped' function.

Example of the sliding window approach and first fitting is shown below:

![alt text][image6]

In 'fit_next_binary_warped' function, we no longer divide the image into several y window sections, instead we identify the non-zero pixel in the binary warped image and combine it with the polynomial fitting from previous frame to find the area of interest. From this selected area, we refit the quadratic parameters on all the nonzero pixels. From here, we can use the new fitted parameters as the input of the next frame, leveraging on the assumption of continuity in the frame.

Example of fitting using previous fitted parameters is shown below:

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Codes for roc are shown in section '4. Fit line'.

We use the function 'get_radius(x, y, y_eval)' to evaluate radius of curvature at the bottom of the camera image (closest to the car position). The roc is calculated by first converting the (x,y) in pixel space to world space (in meters). And then the (x,y) is refitted. Using formula given in the lecture, we calculate the ROC, evaluated at the position closest to the car (bottom pixels in original image).

Codes for calculating car-to-lane offset is in section '5. Project back'.

We use the function 'get_center_offset(newwarp)' to obtain the car-to-lane offset. The input 'newwarp' is a binary image that has been warped back to the original camera view, with known position of the lane filled with nonzero values. It returns
the offset value, positive numbers mean the car is slightly on the right of the lane center, and otherwise is true. The way we calculate this offset is by taking the the last row of bottom pixels and observed first and last nonzero values indices. We assume the center of the lane is calculated by the average of first and last nonzero value indices, and center of the car is calculated by the center of the image. The difference between them is thus the offset.

An example of the value of roc and car-to-lane offset from the test images are given below:

| Left,right roc| car-to-lane   | 
|:-------------:|:-------------:| 
| 866, 1946     | -0.33     	| 
| 925, 4554     | -0.45     	|
| 6165, 923     | -0.23     	|
| 1758, 688	    | -0.46     	|
| 2982, 1873	| -0.01			|
| 2004, 608		| -0.47			|

We notice the value of the roc and offset seems to be reasonable (roc ~ 1 km, offset ~ 0.5 m).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Codes for this section are shown in '5. Project back'.

I use the function 'project_back(image, M, Minv, objpoints, imgpoints, left_fit, right_fit)' to project back the lane fitting into the original image. This function combines all the previous image pipelines to generate an image with fitted lane. The input parameters consists of the input 'image', perspective transformation matrix 'M' and 'Minv', the camera distortion calibration points 'objpoints' and 'imgpoints', and the initial quadratic polynomial fitting of the lane 'left_fit' and 'right_fit'. It return the new matrix consist of fitted lane overlay on top the (undistorted) original image. It also retursn the roc of left and right lane, and the car-to-lane offset distance, and the new quadratic fit parameters. Example of the output of roc and offset is shown in previous answer. Example of the resulting images is shown below:

![alt text][image8]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

The codes are shown in '6. Video pipeline'.

Here I created an object pipeline from class Pipeline. The function of this object is to keep some useful state information from previous frame to the next frame. As discuss in lecture, this also allows reset and smoothing in the lane-finding algorithm. However, I did not implement such functionalities here, it seems for the required project_video, this is not a required features.

To check if most of the results make sense, I plotted the detected roc and car-to-lane offset in below:
![alt text][image9]
![alt text][image10]

As shown in the result, the roc of left and right lane tend to be parallel to each other, and the radius ~ 1 km. The center offset also tend to be negatives (car is slightly to the left from lane center), which can be visually verified from the video.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This is a tedious project. With a lot of manual tuning of the knobs, without the example of the given codes in the lecture, it will take some time to obtain comparable result to what I have currently. And it is likely with the given parameters, it only works on some nice situation.

In the result of the video pipeline, I noticed some wobbly in the lane finding. It's caused by shadow of an object. Several method can be used to overcome this, e.g. smoothing functionality from previous n frames or better thresholding in color space. However, I believe even with this fix, I think the algorithm may fail, for example, due to existence of another car in the same lane, lane markers fading, bad lighting, etc. 

Improvement on the algorithm includes the reset and smoothing functionalities, or perhaps a smarter / dynamic tuning of the threshold parameters.
