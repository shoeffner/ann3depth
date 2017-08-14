# Introduction

Generating depth maps from single monocular images is still an unsolved
computer vision task. While there were many advances during the last few years,
it is still not possible to achieve depth maps with the same level of detail as
ground truth depth maps using laser scanners or IR sensors allow. Even relative
estimates, which allow humans to interpret scenes in images are difficult to
achieve.

At the start of our project we set out to build an artificial neural network
which would be capable of identifying depth in video streams, potentially using
information about movements between images to make better estimates than single
images would allow. But over the course of the project it turned out to be hard
enough to reproduce state of the art models for single monocular images, such
that we tuned down our project ideas.

There are two approaches which we decided to tackle: Liu et al.'s Deep
Convolutional Neural Fields (DCNF) [@liu2016] and the slightly earlier paper
about Multi-Scale Deep Networks (MSDN) by Eigen at al. [@eigen2014]
which inspired parts of Liu et al.'s approach. While implementing Liu et al.'s
DCNF, we quickly stumbled over a problem: Many matrices became simply too
big to be computed on a single GPU. Eventually we managed to get it to run by
simplifying the model greatly and using a batch size of only one sample. We
then tried something new: We tried to run the model distributed across the
university's on premise data center.

This worked out quite well, but for the DCNF implementation we didn't really
succeed in getting a better performance as the model itself was not written
well to be run on a distributed system. Since the model is fairly complex, we
figured it would be much easier to try the distributed solution on a simpler
model, so we set out to reimplement Eigen et al.'s MSDN.

In this report we will focus on the challenges we had to overcome to get
TensorFlow, a general computational library specialized for distributed machine
learning applications, and the Sun Grid Engine (SGE), a system to manage
computer clusters and assign computation time to multiple users, to work
together, present some of our solutions and give ideas on how the system and
workflows we came up with could be refined and improved.
First we will introduce the SGE and explain its purpose and give some usage
examples. Then we will quickly explain TensorFlow's approach to machine
learning models and distributed computing.  After both frameworks are
introduced, we will explain why it is difficult to run distributed TensorFlow
models on the SGE and how we tackled this problem.
We well then briefly go over the application and discuss our results in
reimplementing Eigen et al.'s MSDN. Eventually we will outline shortcomings of
our approach and ways to potentially solve these problems in future
applications.


# Sun Grid Engine


# TensorFlow


# Distributed computing using the Sun Grid Engine


# Results of reimplementing Eigen et al.'s MSDN


# Shortcomings, issues and possible solutions to distributed TensorFlow on the Sun Grid Engine


# Conclusion


# References
