import functools
import itertools
import operator
import os
import signal

import tensorflow as tf


def create_reset_metric(metric, scope, **metric_args):
    """Creates a ops to handle streaming metrics.

    This is a wrapper function to create a streaming metric (usually
    tf.contrib.metrics.streaming_*) with a reset operation.

    Args:
        metric: The metric function
        scope: The variable scope name (should be unique, as the variables of
               this scope will be reset every time the reset op is evaluated)
        metric_args: The arguments to be passed on to the metric.

    Returns:
        Three ops: the metric read_out op, the update op and the reset op:
            metric_op, update_op, reset_op
    """
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op


def make_template(scope=None, create_scope_now_=False, unique_name_=None,
                  custom_getter_=None, **kwargs):
    """A decorator to map a function as a tf template using tf.make_template.

    This enables variable sharing between multiple instances of that function.

    Args:
        scope: The scope for this template. Defaults to the function name.
        create_scope_now_: Passed to the tf.make_template function.
        unique_name_: Passed to the tf.make_template function.
        custom_getter_: Passed to the tf.make_template function.
        kwargs: Passed to the tf.make_template function.

    Returns:
        The function wrapped inside a tf.make_template.
    """
    def make_tf_template(function):
        template = tf.make_template(function.__name__
                                    if scope is None or callable(scope)
                                    else scope,
                                    function,
                                    create_scope_now_=create_scope_now_,
                                    unique_name_=unique_name_,
                                    custom_getter_=custom_getter_,
                                    **kwargs)

        @functools.wraps(function)
        def wrapper(*caller_args, **caller_kwargs):
            return template(*caller_args, **caller_kwargs)
        return wrapper

    if callable(scope):
        return make_tf_template(scope)
    return make_tf_template


def name_scope(scope, *scopeargs, **scopekwargs):
    """A decorator to wrap a function into a tf.name_scope.

    Args:
        scope: The scope name.

    Returns:
        The wrapped function.
    """
    def add_scope(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with tf.name_scope(scope, *scopeargs, **scopekwargs):
                return function(*args, **kwargs)
        return wrapper
    return add_scope


def variable_scope(scope, *scopeargs, **scopekwargs):
    """A decorator to wrap a function into a tf.variable_scope.

    Args:
        scope: The scope name.

    Returns:
        The wrapped function.
    """
    def add_scope(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with tf.variable_scope(scope, *scopeargs, **scopekwargs):
                return function(*args, **kwargs)
        return wrapper
    return add_scope


def with_device(device):
    """A decorator to specify a device for a function.

    Args:
        device: The device name.

    Returns:
        The wrapped function.
    """
    def set_device(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with tf.device(device):
                return function(*args, **kwargs)
        return wrapper
    return set_device


def estimate_size_of(graphkey):
    """Estimates the size of all tensors in a collection.

    Args:
        graphkey: The GraphKey key.

    Returns:
        The estimated size in MB.
    """
    return sum([functools.reduce(operator.mul, [int(s) for s in v.shape])
                for v in tf.get_collection(graphkey)]) * 4 / 1024 / 1024


def create_summary_hook(graphkey, summary_writer, secs=150):
    """Adds a summary hook with scalar summaries of tensor values for
    tensors inside the collection of graphkey.

    Args:
        graphkey: The key which tensors should be summarized.
        summary_writer: The FileWriter to use.
        steps: The summary will be stored every N steps.

    Returns:
        A SummarySaverHook which saves the requested summaries.
    """
    tensors = tf.get_collection(graphkey)
    summaries = []
    for tensor in tensors:
        name = '/'.join(tensor.name.split('/')[0:2]).split(':')[0]
        summaries.append(tf.summary.scalar(name, tensor, []))
    summary_op = tf.summary.merge(summaries)
    return tf.train.SummarySaverHook(save_secs=secs,
                                     summary_writer=summary_writer,
                                     summary_op=summary_op)


class StopAtSignalHook(tf.train.SessionRunHook):
    """Hook that requests stop when a signal is received."""

    def __init__(self, signals=None):
        """Initializes a `StopAtSignalHook`.

        The hook requests stop if one of the specified signals is received.

        Handles by default these signals (if signals is None):
            SIGUSR1, SIGUSR2, SIGALRM, SIGINT, SIGTERM
        The list can be overwritten by setting signals manually.

        Args:
            signals: List of signals to handle.
        """
        self.signal_received = 0
        if signals is None:
            signals = [signal.SIGUSR1, signal.SIGUSR2,
                       signal.SIGALRM, signal.SIGINT, signal.SIGTERM]
        for s in signals:
            signal.signal(s, self.__signal_handler)

    def __signal_handler(self, signum, frame):
        """Sets self.signal_received to signum."""
        self.signal_received = signum

    def after_run(self, run_context, run_values):
        """If a signal was received, a stop will be requested."""
        if self.signal_received:
            run_context.request_stop()


class TraceHook(tf.train.SessionRunHook):
    """Hook to perform Traces every N steps."""

    def __init__(self, summary_writer, every_step=50,
                 trace_level=tf.RunOptions.FULL_TRACE):
        """Initializes the TraceHook.

        Traces the 0th and every N-th step.

        Args:
            summary_writer: The FileWriter to use.
            every_step: Each N-th step a trace will be performed.
            trace_level: The trace level to be passed to tf.RunOptions.
        """
        self._trace = True
        self.writer = summary_writer
        self.trace_level = trace_level
        self.every_step = every_step

    def begin(self):
        """Check if the global step is available inside the graph."""
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            RuntimeError("Global step should be created to use _TraceHook.")

    def before_run(self, run_context):
        """If a trace is requested, adds tf.RunOptions to the session args.

        Always requests the global step.

        Args:
            run_context: The run context.

        Returns:
            SessionRunArgs as described above.
        """
        if self._trace:
            options = tf.RunOptions(trace_level=self.trace_level)
        else:
            options = None
        return tf.train.SessionRunArgs(fetches=self._global_step_tensor,
                                       options=options)

    def after_run(self, run_context, run_values):
        """If a trace was requested for this run, store the results.
        Otherwise check if the next step should request a trace.

        Args:
            run_context: The original run context.
            run_values: The resulting run values.
        """
        global_step = run_values.results
        if self._trace:
            self._trace = False
            traced_step = global_step - 1
            self.writer.add_run_metadata(run_values.run_metadata,
                                         f'{traced_step}', traced_step)
        if not global_step % self.every_step:
            self._trace = True
