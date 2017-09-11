## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/test_hog.png
[image3]: ./output_images/window_all_search.png
[image4]: ./output_images/test_images_vd.png
[image5]: ./output_images/heatmap.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first and second cells of the IPython notebook main.ipynb (Part 1: HOG + SVC Tuning) and section "DATA EXPLORATION", "HOG FEATURES", and "FEATURE_EXTRACTION" in vehicle_detection_functions.py.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then test extracting the hog features by choosing several `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:

![alt text][image2]


#### 2. Explain how you settled on your final choice of HOG parameters.

The code for this step is contained in the third cell Part 1: HOG + SVC Tuning of main.ipynb.

To choose the right HOG parameters for this task, my initial idea is to find which features are going to give me the best accuracy during training with SVC.
Two main parameters that I played with was color space and inclusion of spatial bin and color histogram features. I also tried to use / not use the scaler on the data. For other parameters, we use the default parameters with orientation=11, pixels_per_cell=(16,16), and cells_per_block=(2,2), spatial_size=(32,32) and hist_bins=32. 

Below is the table of some parameters search that I use:

| colorspace	|	bin_feat & hist_feat	|	scaler	|	accuracy	|
|:-------------:|:-------------------------:|:---------:|:-------------:|
|RGB			|			no				|	no		|	0.9713		|
|HSV			|			no				|	no		|	0.9817		|
|LUV			|			no				|	no		|	0.9806		|
|HLS			|			no				|	no		|	0.9769		|
|YUV			|			no				|	no		|	0.9800		|
|YCrCb			|			no				|	no		|	0.9814		|
|RGB			|			no				|	yes		|	0.9581		|
|HSV			|			no				|	yes		|	0.9676		|
|LUV			|			no				|	yes		|	0.9676		|
|HLS			|			no				|	yes		|	0.9640		|
|YUV			|			no				|	yes		|	0.9741		|
|YCrCb			|			no				|	yes		|	0.9721		|
|RGB			|			yes				|	yes		|	0.9842		|
|HSV			|			yes				|	yes		|	0.9924		|
|LUV			|			yes				|	yes		|	0.9873		|
|HLS			|			yes				|	yes		|	0.9882		|
|YUV			|			yes				|	yes		|	0.9882		|
|YCrCb			|			yes				|	yes		|	0.9885		|

From the search, it seems like the best config will be to use colorspace of HSV, with scaling of the features, and inclusion of spatial bin and color histogram features. But when later tested on the test images, it was found the colorspace YUV provided less false positives. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I use a linear SVC to train the model in third cell in "PART 1: HOG AND SVC TUNING". This is the cell I used to perform hyperparameter tuning. The steps for tuning the classifier began with extracting features (HOG + spatial bin + color histogram). The extracted features are scaled using sklearn StandardScaler. The vehicle and non-vehicle data are then split into training and validation set. I try to vary the parameter C in linear SVC, but did not observe significant improvement. 




### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To perform an efficient sliding window search, I use similar codes provided in the lesson, as seen in "SLIDING WINDOW SEARCH" in vehicle_detection_functions.py and PART 2: SLIDING WINDOW of main.ipynb. 

The function find_cars uses a given classifier to find a vehicle in the image in area bounded by ystart to ystop, with the size of the window scaled by the variable scale. Other parameters are chosen to follow the hand-picked parameters in previous sections. This function use hog features, spatial bin, and color histogram features with proper feature scaling.

I used a multi-scaled window as suggested in the lessons, and experimented with various configuration of ystart, ystop, and scale. One of the good starting points for the configurations are from the following blog: https://github.com/jeremy-shannon/CarND-Vehicle-Detection. From here I fine tune the parameters that will eventually obtain good vehicle detection in test images. The followings are the plot of the windows that I search:

![alt text][image3]

The basic configuration is to use 2 window for each scale from 1 to 3.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Final configuration: orientation=11, pixels_per_cell=(16,16), and cells_per_block=(2,2), spatial_size=(32,32) and hist_bins=32, colorspace='YUV', include spatial bin and color histogram features with feature scaling.  Here are the vehicle detections for the test images:

![alt text][image4]
---






### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_out2.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The codes for this section is provided in "PIPELINE VIDEO" in vehicle_detection_functions.py, last cell of "PART 2: SLIDING WINDOW", and "PART 3: VIDEO" in main.ipynb.

I will describe the filter for false positives using thresholded heatmap of the detection. Using multi-scaled windows search mentioned in previous sections, I obtain heatmap of prediction of where the vehicle can be, using the function process_frame. Here is the plot of the heatmap for test images:

![alt text][image5]

I then use a threshold of 1 to identify the vehicle position and used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. It seems like my current classifier does not produce a lot of false positives. 

For detection in a video, I want to make use of the nature of time series of the frame in the video. I constructed an instance that maintain the heatmaps from 10 frames before, and increase the threshold as I have more historical heatmaps. I play with the parameters, and finally decided to use threshold of 1 + len(history) / 3.

---



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As expected from the hand tuned feature engineering, the difficulty is always to pick the right parameters, and whether these parameters can generalize. 

While it is principally straightforward to find the best hyper parameters for SVC classifier (train test split), the accuracy of classifier may not correspond directly to the performance of vehicle detections with multi-scaled windows search. The difficulty of parameters tuning become more apparent in choosing the right image window scale and range to search. To obtain the best vehicle detection, perhaps a labelled video data of the position of the vehicle can be a more direct and helpful training set.

At this point, with all these tunings, I am not sure if the data can generalize to other circumstances (e.g. in city driving, different weather conditions, intersections, etc).

One alternative to hand tuned feature extraction is to use other more robust image segmentation algorithms.