import argparse
import json
import logging
import logging.config
import signal

import tensorflow as tf

import data
import models


def main():
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
        server.join()
    else:
        logger.info('Starting worker job.')
        logger.info(f'Loading model {args.model}.')

        greedy_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
            0, #cluster.num_tasks('ps'),
            tf.contrib.training.byte_size_load_fn
        )
        device_setter = tf.train.replica_device_setter(
            cluster=cluster,
            worker_device=f'/job:{args.job_name}/task:{args.task_index}',
            ps_strategy=greedy_strategy
        )
        with tf.device(device_setter):
            model_train_op = getattr(models, args.model)()

        hooks = [
            tf.train.StopAtStepHook(last_step=args.steps)
        ]

        logger.info('Starting session.')
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=args.task_index==0,
                checkpoint_dir=args.ckptdir,
                scaffold=None,
                hooks=hooks,
                chief_only_hooks=None,
                save_checkpoint_secs=args.ckptfreq,
                save_summaries_steps=100,
                save_summaries_secs=None,
                config=None,
                stop_grace_period_secs=120,
                log_step_count_steps=100) as session:
            while not session.should_stop():
                session.run(model_train_op)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='*',
                        default=['make3d1', 'make3d2'],
                        help='The datasets to use.')
    parser.add_argument('--model', '-m', default='', type=str,
                        help='Enter a model name.')
    parser.add_argument('--steps', '-s', default=1000000, type=int,
                        help='Total steps')
    parser.add_argument('--batchsize', '-b', default=32, type=int,
                        help='Batchsize')
    parser.add_argument('--ckptdir', '-p', default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--ckptfreq', '-f', default=600, type=int,
                        help='Create a checkpoint every N seconds.')
    parser.add_argument('--tbdir', '-l', default='tb_logs',
                        help='Tensorboard directory')
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
