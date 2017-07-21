% Status report: ANN for live depth estimation from video streams
% Sebastian Höffner ; Andrea Suckro
% June 30, 2017


# Reminder: Our Goals

Specific
  ~ Create an ANN to estimate depth maps for a self recorded video

Measurable
  ~ Quality of the estimate is the difference between the depth value of the
  recording and our predicted value

Attainable
  ~ There is work available for single images and the information of the
  connected frames could be used to improve the result

Relevant
  ~ Depth information can be used for many tasks: robot navigation, object segmentation,
  enrichment for regular cameras ...

Time-bound
  ~ Given by the time of the semester


# What has happened so far

We dealt a lot with data acquisition and preprocessing so far:

* Collect datasets that are useful for training the network (2D image data and depth maps)
* Producing a pipeline to read this data in and standardize it, such that we can use  different sources of pictures for training
* Data visualization to get a feeling for the datasets
* Automating the whole process to make it easily runnable from everywhere (aiming at the grid)


# Datasets

We started with the current literature and the datasets referenced in there.

make3d1
~ contains test and train folders of images (1704 x 2272) and their depth images as tar files

make3d2
~ contains test and train folders of images (2272 x 1704) and their depth images as tar files

nyu
~ a 2.97 GB MATLAB file, that contains images of different rooms (basement, kitchen, office etc.) with respective depth images recorded with a Kinect


# Data preprocessing

The datasets had different formats and characteristics to be dealt with. These items have been completed for preprocessing the data:

* We wrote a data downloader for the different datasets
* We normalized all images to a size of 640 x 480 and their depth maps to 73 x 55 (rotation and scaling)


# Data visualization

![Databrowser](docs/presentations/imgs/databrowser.png)


# Learning Setup

We have a simplistic network running and can now build up on this with a more advanced architecture.


# What about Videos?

We are currently evaluating more direct applications on the 2D data. One possible interesting problem deals with scaling the depth images, since their resolution is quite restricted at the moment.


# Ways forward

Our next work items include (but may not be restricted to):

* writing one or more deep networks to estimate the depth information from the single images in the different datasets
* learn them on the IKW grid
* apply the network to a self recorded video with depth information and evaluate results
    * no pursuit of learning on video data directly


# Last slide

![Everything will work out just fine](docs/presentations/imgs/lastslide.jpg)
