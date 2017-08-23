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
The institute has a cluster of several hosts that is managed by the Sun Grid
Engine (now known as the Oracle Grid Engine). There is one master/cluster
management server that monitors the current status of all nodes in the grid and
that uses a scheduler to determine on which node and when new jobs should be run.
New jobs can be submitted via a a command line interface (find the whole
reference [here](http://gridscheduler.sourceforge.net/htmlman/htmlman1/qsub.html))
to this scheduler. By specifying resources that are necessary to run this job
(like the minimum of available RAM) the users can ensure that the machine their
job is placed on by the scheduler has the necessary computational resources.
The requested resources are split into two categories: testable and consumable.
Testable are those resources of a machine that have to exist in order to run the
job but are not used up or exhausted by the job itself. An example would be the
speed of the CPU - if a job requires (for whatever reason) a minimum of 4GHz to
be run, it is still possible to have other jobs in parallel that also use the
CPU. The consumable resources now in contrast are used up by the job. Examples
would be the minimum RAM or Cuda cores. These resources are not monitored live
by the scheduler, but are also fixed parameters that have been recorded with the
nodes upon integrating them in the grid. Other use cases include the specific
submission to a host by requiring the hostname or exclusion of hosts. When the
job is scheduled to a node that fulfills all requirements it is executed either
until the job is finished or the walltime is reached whatever happens first.
Another layer of organization are queues. Computational nodes belong to 1 up
to n queues and different users can have differing permissions in respect to
those queues. For example members of the workgroup have extended walltimes
compared to users from different workgroups and so forth. Jobs are per default
committed to a specific queue, but it is also possible to submit to a specific
queue.


# TensorFlow
TensorFlow is an open source software library that abstracts complex
mathematical models and data flows to allow for optimized computation. We used
version 1.3r of the Python API (documentation can be found
[here](https://www.tensorflow.org/api_docs/python/)) to implement the models
presented in the papers. Conceptually TensorFlow requires the User to build a
computational graph - a recipe for the data and the operations applied to it.
The nodes in the graph are either tensors, multi-dimensional arrays that contain
the data, or operations, that compute a function given some input tensors. As
the chain of those operations grows so does the graph that represents the
overall program. Additional mechanisms like scoping helps to reuse computational
building blocks like a convolutional layer again and again and structuring
the graph while constructing it. The final graph can be then executed in a
TensorFlow session to operate on the real input data. TensorFlow then takes care
of computational and hardware based optimizations that meant for our project the
usage of the graphics card of the hosts. TensorFlow comes with an own monitoring
tool that allows for visualization of the training process as well as
representations of the graph.



# Distributed computing using the Sun Grid Engine


# Results of reimplementing Eigen et al.'s MSDN


# Shortcomings, issues and possible solutions to distributed TensorFlow on the Sun Grid Engine


# Conclusion


# References
