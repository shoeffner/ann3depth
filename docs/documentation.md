# Introduction

Generating depth maps from single monocular images is still an unsolved
computer vision task. While there were many advances during the last few years,
it is still not possible to achieve depth maps with the same level of detail as
ground truth depth maps using laser scanners or IR sensors allow. Even relative
estimates are difficult to achieve.

At the start of our project we set out to build an artificial neural network (ANN)
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

This worked out quite well, but for the DCNF implementation we did not
succeed in getting a better performance as the model itself was not written
well to be run on a distributed system. Since the model is fairly complex, we
figured it would be much easier to try the distributed solution on a simpler
model, so we decided to reimplement Eigen et al.'s MSDN.

In this report we will focus on the challenges we had to overcome to get
TensorFlow, a general computational library specialized for distributed machine
learning applications, and the Sun Grid Engine (SGE), a system to manage
computer clusters and assign computation time to multiple users, to work
together, present some of our solutions and give ideas on how the system and
workflows we came up with could be refined and improved.
First we will introduce the SGE and explain its purpose and give some usage
examples. Then we will quickly explain TensorFlow's approach to machine
learning models and distributed computing. After both frameworks are
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


# Distributed computing using the SGE

Sometimes machine learning models are too big for a single processing unit,
even too big for the available random accesss memory (RAM). To handle such
models, TensorFlow supports the concept of distributed computing. In the
distributed setting, multiple processing units ("workers") process a stream of
data stored on multiple parameter servers ("ps node"). TensorFlow is designed
specifically with the Google Cloud Platform services[^gcp] in mind, a
webservice where one can start up multiple virtual machines as a computing
cluster of nodes. Each of the virtual machines is completely encapsulated,
receives a static IP address and can, e.g. in case of failure, simply be
restarted again with the same settings, the same IP address etc. This approach
has some advantages. It is possible to set up specific machines for specific
tasks, e.g. ps nodes benefit from having high amounts of RAM and CPU power,
while workers can use the power of multiple GPUs efficiently. Additionally
there are tools which handle cluster initializations and take care of keeping
instances alive and restarting them ("reprovisioning") in case of failures.

[^gcp]: https://cloud.google.com/ \linebreak
    There are other cloud services, e.g. Amazon Web Services, Digitial Ocean,
    Microsoft Azure, and more, which support similar approaches to virtualization
    and cluster provisioning. There are also several competitors like Rackspace and
    Oracle, which take a different approach and provide on premise solutions to
    build one's own cloud server infrastructure.

The SGE does not follow these modern principles of distributed
computing using virtualization or clusters to perform large scale computations.
In those settings, one usually can assume nearly unlimited resources and just
specifies what they want for their computations and provision instances
accordingly. The SGE instead allows to distribute resources and
computation fairly among its users. While this is a very efficient and
advantageous method for single process computations, it has some flaws when it
comes to multi process set ups, like the one with multiple workers and ps nodes
we want to build for the distributed computation of TensorFlow models.

To execute a distributed model, TensorFlow processes need to know about the
other devices of the cluster and have to be able to open TCP connections between
them. The information is provided via a cluster specification (`ClusterSpec`),
which can be represented as a simple JSON:

```{ .json }
{"worker": [
    "cippy18.cogsci.uni-osnabrueck.de:5001",
    "cippy19.cogsci.uni-osnabrueck.de:5001",
    "cippy20.cogsci.uni-osnabrueck.de:5001",
    "cippy21.cogsci.uni-osnabrueck.de:5001"
 ],
 "ps": [
    "cippy14.cogsci.uni-osnabrueck.de:5002",
    "cippy16.cogsci.uni-osnabrueck.de:5002",
    "cippy17.cogsci.uni-osnabrueck.de:5002",
    "cippy01.cogsci.uni-osnabrueck.de:5002"
 ]
}
```

Here we use eight different hosts, four as workers and four as ps nodes. In the
following sections we describe how the processs can communicate with each
other, how we generate the cluster specifications automatically, and how we
overcome the problem of walltimes, the time after which the SGE kills a running
process to free resources for other users.


## Routing inside the SGE: Process communication

Assume a simple scenario for a distributed TensorFlow model. We have one worker
and one ps node and want to run both simultaneously on the SGE. After
submitting both tasks, e.g. using a command like

    qsub -N worker -l mem=2G worker.sge && \
        qsub -N ps_node -l mem=16G ps.sge

it will always be uncertain on which host machines they will eventually be
executed. We could make some assumptions about which machines will be available
(maybe the grid only has two hosts and one of them has only 8 GB of memory),
but very often that is not the case. We therefore have no simple way to
communicate between the jobs, unless they perform some lookups. For example,
if submitting the worker first and then the ps node, it is possible to pass the
worker's job ID to the ps node:

    WORKER_JID=$(qsub -N worker -l mem=2G worker.sge | cut -d' ' -f3)
    WORKER_JID=$WORKER_JID qsub -N ps_node -l mem=16G ps.sge

The ps node could now query the job status of the worker (`qstat -j
$WORKER_JID`) and find out the host its running on. The two jobs could now
communicate[^zmqnote].

[^zmqnote]: Assuming there is a way to communicate between the two hosts inside
the network. We were able to simply use some open ports for which TCP and UDP
were enabled for local network traffic. Alternatively we also built a small
solution using a complex network of automatic ssh tunnels between all hosts of
some jobs' "network". As an alternative, our system administrator also
suggested [ZeroMQ](http://zeromq.org), however since we were able to use ports
directly for our TCP connections, we did not try it out.

For multiple hosts this process gets more difficult: Each process needs to know
about all other processes' hosts to be able to build up a `ClusterSpec` and
open up TCP connections between each other. To make this process easier, we
decided to force the computations to be on specific hosts to build the
`ClusterSpec` beforehand, rather than having each job figure out this
information.


## Network architecture and cluster specifications

To effeciently build cluster specifications we divide our grid jobs into a
single master job and several client jobs. The master job, which we will also
call "keepalive", will not perform any computations for the TensorFlow models,
but instead only takes care of proper initialization of the client jobs and
restarts them as needed. The client jobs will then either run worker or ps node
processes using a cluster specification provided by the keepalive job.

The keepalive job goes through several steps:

1. Determining the list of available hosts for the requesting user.
2. Creating the `cluster_spec`.
3. Starting the clients.
4. Monitoring the clients.
5. Shutting down the clients.
6. Optionally resubmitting itself.

Step 1 produces a list of hosts which is processed in step 2:

```bash
$ qstat -f -F h -U $(whoami) | python tools/grid/extract_hosts.py
cippy11.cogsci.uni-osnabrueck.de
cippy12.cogsci.uni-osnabrueck.de
cippy13.cogsci.uni-osnabrueck.de
cippy14.cogsci.uni-osnabrueck.de
cippy15.cogsci.uni-osnabrueck.de
```

Step 2 polls the SGE for information about the resources of each of these
hosts:

```changelog
$ qhost -F cuda,cuda_cores -q -h cippy11.cogsci.uni-osnabrueck.de
HOSTNAME                ARCH         NCPU  LOAD  MEMTOT  MEMUSE  SWAPTO  SWAPUS
-------------------------------------------------------------------------------
global                  -               -     -       -       -       -       -
cippy11.cogsci.uni-osnabrueck.de lx26-amd64      4  0.00   31.4G    2.1G   16.0G   58.2M
    Host Resource(s):      hc:cuda=1.000000
   hf:cuda_cores=384.000000
   training.q           BIP   0/0/2
   all.q                BIP   0/0/3
```

The information is parsed and stored. In the example above we have the
following relevant resources: available memory of 31.4 GB RAM, a graphics card
with CUDA support and 384 CUDA cores. Additionally we can see how many slots
are available per queue for this host (for training.q it is `0/0/2`, which
means 0 reserved, 0 in use, 2 available). The Python script
`split_resources.py` uses
the information available to distribute the tasks in a greedy fashion among the
available devices. Assuming e.g. four workers and four ps nodes, it would
assign devices with high RAM values as ps nodes and devices with many CUDA
cores to the workers. The resulting cluster specification is stored as a csv
file (which is easier to parse with the keepalive job) and as a JSON (which
maps nicely into Python dictionaries used for the `ClusterSpec` needed by
TensorFlow)[^excs]:

```csv
cippy18.cogsci.uni-osnabrueck.de,5001,worker
cippy19.cogsci.uni-osnabrueck.de,5001,worker
cippy20.cogsci.uni-osnabrueck.de,5001,worker
cippy21.cogsci.uni-osnabrueck.de,5001,worker
cippy14.cogsci.uni-osnabrueck.de,5002,ps
cippy16.cogsci.uni-osnabrueck.de,5002,ps
cippy17.cogsci.uni-osnabrueck.de,5002,ps
cippy01.cogsci.uni-osnabrueck.de,5002,ps
```

[^excs]: An example cluster specification JSON can be found at the beginning of this
  chapter.

The keepalive job parses the information from the stored csv file and submits
client jobs to the specified hosts, using some environment variables to specify
the jobs' tasks.

Each client can, after successful submission, read the JSON file and provide a
`ClusterSpec` for TensorFlow.



# Results of reimplementing Eigen et al.'s MSDN


# Shortcomings, issues and possible solutions to distributed TensorFlow on the SGE


# Conclusion


# References
