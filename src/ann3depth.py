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

    logger.info('Setting up config.')
    config = tf.ConfigProto(
        device_count={'CPU': 1,
                      'GPU': get_num_GPU()},
        allow_soft_placement=True,
        log_device_placement=__debug__,
    )

    if args.job_name == 'ps':
        logger.info('Starting ps job.')
        queue = create_done_queue(args.task_index, len(cluster_spec['worker']))
        with tf.Session(server.target, config=config) as session:
            for i in range(len(cluster_spec['worker'])):
                session.run(queue.dequeue())
    elif args.job_name in ['worker', 'local']:
        chief = args.task_index == 0

        logger.info(f'Task: {args.task_index} -- Chief? {chief}')

        run_id = args.model + ('' if not args.id else f'_{args.id}')
        ckptdir = str(os.path.join(args.ckptdir, run_id))
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
            logger.debug(f'Trainable variables have about {size_train:.1f} MB')

        logger.info('Setting up hooks.')
        stop_at_signal_hook = tfhelper.StopAtSignalHook()
        hooks = [
            tf.train.StopAtStepHook(last_step=args.steps),
            stop_at_signal_hook,
            tf.train.FinalOpsHook(create_ps_notifier(cluster_spec)),
            tfhelper.create_summary_hook(tf.GraphKeys.LOSSES, ckptdir,
                                         args.sumfreq),
        ]

        if args.job_name != 'local':
            logger.info(f'Starting alarm: {args.timeout} s timeout.')
            signal.alarm(args.timeout)

        logger.info('Starting session.')
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=chief,
                checkpoint_dir=ckptdir if chief else None,
                scaffold=None,
                hooks=hooks,
                chief_only_hooks=None,
                save_checkpoint_secs=args.ckptfreq,
                save_summaries_steps=None,
                save_summaries_secs=args.sumfreq,
                config=config,
                stop_grace_period_secs=120,
                log_step_count_steps=100) as session:
            while not session.should_stop():
                session.run(model_train_op)
        logger.info('Session stopped.')
        sys.exit(stop_at_signal_hook.signal_received)
    else:
        logger.warning(f'No suitable job description found! {args.job_name}')


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


def get_num_GPU():
    """Determines the number of visible GPUs by checking the
    CUDA_VISIBLE_DEVICES environments variable.

    See http://acceleware.com/blog/cudavisibledevices-masking-gpus for
    information about how CUDA_VISIBLE_DEVICES works.

    This function splits the value at , and checks whether the first is -1.
    If that is the case, 0 GPUs are available.
    If it is another value, then the number of elements is the number of
    available GPUs (as those values represent the IDs).

    Returns:
        The number of available GPUs.
    """
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(',')
    if cuda_visible[0] == '-1':
        return 0
    else:
        return len(cuda_visible)


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
    parser.add_argument('--id', default='', type=str,
                        help='Checkpoint path suffix.')
    parser.add_argument('--ckptfreq', '-f', default=900, type=int,
                        help='Create a checkpoint every N seconds.')
    parser.add_argument('--sumfreq', '-r', default=150, type=int,
                        help='Create a summary every N seconds.')
    parser.add_argument('--datadir', '-d', default='data', type=str,
                        help='The data directory containing the datasets.')
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
