
** Vehicle Detection Project **

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/test_images_out.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.    

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this project is in a combination of an Ipython notebook (vehicle_detection.ipynb) and a python module containing several utilities function (project_utils.py).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Above can be found in Cells 1 to 4 of vehicle_detection.ipynb

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters on the test images and arrived at one that seemed to give the best results on the test images. Results on the test images can be found later in the writeup.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the LinearSVC module of the scikit-learn package. I was able to obtain a high accuracy score of 98.5% that seemed Ok for the project.

Thi scan be found in Cell 5 of the notebook.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The hog subsampling approach was used to implement the sliding window search. The serach was restricted to the space between 400 and 600 pixels in the 'Y' dimension. In addition a varying scale was used for different sections of the images to detect near and far away objects. The code can be found in the function find_cars() in Cell 6  and the process_image function in cell 8. An illustration of the scaled and sliding window is in the following figure.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I also used a heatmap that ensured that a vehicle was detected by at least two bounding boxes which eliminated most of the false positives. There are occasional false positives and smaller bounding boxes. Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

After having used the pipeline on the test images, initial attempts on the test and project videos resulted in a few false positives and some occasions of missed detections.In addition, the bounding box was not fully enclosing the white car in many frames.

To eliminate the false positives as well as to make use of the temporal correlation of the car positions in the video, I decided to integrate the heatmap over several frames of the video. This has the effect of removing the false positives which only persist for a frame or two. In addition, combining the heatmap also has the effect of unifying overlapping bounding boxes over successive frames. This results in a bounding box that encloses most of the white car.

The code is in the process_image function in cell 8 of the notebook as well as the integration_length setting in Cell 22.

In my implementation, I used heatmap integration over the past 15 frames. The resulting heatmap and the output bounding boxes are shown in the next two images

![alt text][image6]

![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had a tough time detecting and bounding the white car until I I used the YCbCr color space and also integrated over multiple frames. I still see that boxes are a bit wobbly at times and, on rare occasions, I do miss the white car over a couple of frames. One aspect that I haven't explored is to change the search space in response to predicted car positions. This idea may help improve te robustness of detection. Another idea would be to train multiple classifiers over different color spaces. Then unify the bounding boxes detected by both classifiers.

