import argparse
import json
import logging
import logging.config
import os
import signal
import sys

import tensorflow as tf

import data
import models
import tfhelper


def main():
    """Initializes training.

    Reads the program arguments and depending on the tasks starts the different
    processes.

    The local training is handled as a chief worker (worker with task 0) without
    sharded variable placement.

    The parameter servers don't use server.join() but a custom solution to
    determine when to die (this works better with the Sun Grid Engine).

    The workers start training with sharded variable placement if parameter
    server tasks are available.
    """
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger('ann3depth')

    args = parse_args()
    logger.debug(args)

    logger.info(f'This is a {args.job_name} with index {args.task_index}')

    if args.cluster_spec:
        logger.info(f'Using cluster spec from {args.cluster_spec}')
        with open(args.cluster_spec, 'r') as jf:
            cluster_spec = json.load(jf)
    else:
        logger.info('Using default local cluster spec.')
        cluster_spec = {'local': ['localhost:0']}

    cluster = tf.train.ClusterSpec(cluster_spec)

    server = tf.train.Server(cluster,
                             job_name=args.job_name,
                             task_index=args.task_index)

    if args.job_name == 'ps':
        logger.info('Starting ps job.')
        queue = create_done_queue(args.task_index, len(cluster_spec['worker']))
        with tf.Session(server.target) as session:
            for i in range(len(cluster_spec['worker'])):
                session.run(queue.dequeue())
    elif args.job_name in ['worker', 'local']:
        chief = args.task_index == 0

        logger.info(f'Task: {args.task_index} -- Chief? {chief}')
        logger.info(f'Continue? {args.cont}.')

        ckptdir = None
        if chief:
            ckptdir = determine_checkpoint_dir(args.ckptdir,
                                               args.model,
                                               args.cont)
            logger.info(f'Checkpoint dir is {ckptdir}.')

        logger.info('Setting up replica settings.')
        ps_strategy = None
        if cluster_spec.get('ps'):
            ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
                len(cluster_spec['ps']),
                tf.contrib.training.byte_size_load_fn
            )
        device_setter = tf.train.replica_device_setter(
            cluster=cluster,
            worker_device=f'/job:{args.job_name}/task:{args.task_index}',
            ps_strategy=ps_strategy
        )

        logger.info(f'Loading model {args.model}.')
        with tf.device(device_setter):
            inputs, targets = data.inputs(args.datadir, args.dataset, args.batchsize)
            model_train_op = getattr(models, args.model)(inputs, targets)

            size_train = tfhelper.estimate_size_of(tf.GraphKeys.TRAINABLE_VARIABLES)
            logger.debug(f'Trainable variables have {size_train:.1f} MB')

        logger.info('Setting up config.')
        config = tf.ConfigProto(
            # device_count={'CPU': 1, 'GPU': 0},
            # allow_soft_placement=True,
            # log_device_placement=True,
        )

        logger.info('Setting up hooks.')
        hooks = [
            tf.train.StopAtStepHook(last_step=args.steps),
            tf.train.FinalOpsHook(create_ps_notifier(cluster_spec))
        ]

        logger.info('Starting session.')
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=chief,
                checkpoint_dir=ckptdir,
                scaffold=None,
                hooks=hooks,
                chief_only_hooks=None,
                save_checkpoint_secs=args.ckptfreq,
                save_summaries_steps=None,
                save_summaries_secs=args.sumfreq,
                config=config,
                stop_grace_period_secs=120,
                log_step_count_steps=100) as session:

            logger.info('Setting up signal handlers.')
            stop_reasons = []
            should_stop = handle_stop(session, stop_reasons, logger)
            if args.job_name != 'local':
                logger.info(f'Starting alarm: {args.timeout} s timeout.')
                signal.alarm(args.timeout)

            while not should_stop():
                session.run(model_train_op)

        logger.info('Session done.')
        if stop_reasons:
            sys.exit(stop_reasons[0])
    else:
        logger.warning(f'No suitable job description found! {args.job_name}')


def handle_stop(session, stop_request_list, logger):
    """Registers signal handlers and returns a function to check for stop
    requests.

    Handles these signals:
        SIGUSR1, SIGUSR2, SIGALRM, SIGINT, SIGTERM

    Args:
        session: The MonitoredTrainingSession.
        stop_request_list: A list object to which (side-effect!) the caught
                           signals will be appended. This can be used to
                           handle clean session shutdowns with status
                           codes != 0.
        logger: The logger (to notify about received signals).

    Returns:
        A function to check whether the training session should stop or not.
    """
    def signal_handler(signum, frame):
        logger.warning(f'Received signal {signal.Signals(signum).name}.')
        logger.info(f'Requesting session to stop.')
        stop_request_list.append(signum)
    for s in [signal.SIGUSR1, signal.SIGUSR2, signal.SIGALRM, signal.SIGINT,
              signal.SIGTERM]:
        signal.signal(s, signal_handler)
    return lambda: stop_request_list or not session or session.should_stop()

def create_done_queue(ps_task, num_workers):
    """Creates a queue on and for the ps task with the capacity of the number
    of workers.
    At the end (using a FinalOpsHook) each worker will write a 1 into all
    ps queues.
    The ps tasks stay alive until they read worker many times from their own
    queue.

    See
    https://github.com/tensorflow/tensorflow/issues/4713#issuecomment-269499287
    for more info.

    Args:
        ps_task: The ps task for this queue.
        num_workers: The total number of workers (to determine the queue
                     capacity)

    Returns:
        A FIFOQueue.
    """
    with tf.device(f'/job:ps/task:{ps_task}'):
        return tf.FIFOQueue(num_workers, tf.int32,
                            shared_name=f'shutdown_ps_{ps_task}')

def create_ps_notifier(cluster_spec):
    """Creates ops to stop all ps_tasks eventually, see create_done_queue.

    For each ps task a queue is created on those shards using create_done_queue.
    This method returns either tf.no_op() (in case this is only a local job) or
    a list of enqueue ops to enqueue a 1 to each ps task's queue.

    The ops should be used with a FinalOpsHook.

    See
    https://github.com/tensorflow/tensorflow/issues/4713#issuecomment-269499287
    for more info.

    Args:
        cluster_spec: The cluster spec dictionary.

    Returns:
        A list of enqueue(1)_ops, or a no_op for local workers.
    """
    num_ps = len(cluster_spec.get('ps', []))
    num_workers = len(cluster_spec.get('worker', []))
    if 'local' in cluster_spec or num_ps == 0:
        return tf.no_op()
    return [create_done_queue(i, num_workers).enqueue(1) for i in range(num_ps)]


def determine_checkpoint_dir(ckptdir, model, cont=False):
    """Determines which checkpoint directory to use.

    If training continues, this selects a directory
        ckptdir/model/x
    where x is the last available number.
    If not, it creates a directory x+1, or 0 if none
    exists yet.

    Args:
        ckptdir: The base directory for checkpoints.
        model: The model name.
        cont: If True, uses an existing directory.

    Return a checkpoint directory.
    """
    ckptdir = os.path.join(ckptdir, model)
    try:
        runs = os.listdir(ckptdir)
    except FileNotFoundError:
        runs = []
    num = len(runs) - (1 if cont and runs else 0)
    return os.path.join(ckptdir, str(num))

def parse_args():
    """Parses arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='nyu', type=str,
                        help='The dataset to use.')
    parser.add_argument('--model', '-m', default='', type=str,
                        help='Enter a model name.')
    parser.add_argument('--steps', '-s', default=1000000, type=int,
                        help='Total steps')
    parser.add_argument('--batchsize', '-b', default=32, type=int,
                        help='Batchsize')
    parser.add_argument('--ckptdir', '-p', default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--ckptfreq', '-f', default=900, type=int,
                        help='Create a checkpoint every N seconds.')
    parser.add_argument('--sumfreq', '-r', default=300, type=int,
                        help='Create a summary every N seconds.')
    parser.add_argument('--datadir', '-d', default='data', type=str,
                        help='The data directory containing the datasets.')
    parser.add_argument('--cont', '-c', action='store_true',
                        help='Continue from checkpoint')
    parser.add_argument('--timeout', '-k', default=4200, type=int,
                        help='The time after which the process dies.')
    parser.add_argument('--cluster-spec', default='', type=str,
                        help='The path to the cluster specification json.')
    parser.add_argument('--job-name', default='local', type=str,
                        help='"worker" or "ps" for distributed computations.')
    parser.add_argument('--task-index', default=0, type=int,
                        help='Task index for distributed computations.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
