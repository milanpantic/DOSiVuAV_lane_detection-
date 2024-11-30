## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

Done

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Za izračunavanje matrice kamere i koeficijenata distorzije, korišćene su slike šahovske table snimljene iz različitih uglova. Korišćene su funkcije cv2.findChessboardCorners za pronalaženje unutrašnjih uglova šahovske table i cv2.calibrateCamera za izračunavanje matrice kamere (mtx) i koeficijenata distorzije (dist). Dobijeni rezultati sačuvani su u fajl (calib.npz) za dalju upotrebu.

Kasnije je fajl učitan i korišćen za korekciju distorzije pomoću cv2.undistort funkcije, a nova matrica kamere je izračunata pomoću cv2.getOptimalNewCameraMatrix.

![distortion corrected image](exported_img/undisorted.png)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![distortion corrected image](exported_img/undisorted_road.png)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

binary_image.png

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

transformedImg.png

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Not done

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

final_image.png

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

