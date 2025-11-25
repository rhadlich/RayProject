import argparse
import json
import logging
import os
import shutil
import pprint
import random
import re
import time
import gzip
import base64
import struct
import signal
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
from multiprocessing import shared_memory

import ray
import torch
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback, WANDB_ENV_VAR
from ray.rllib.core import DEFAULT_MODULE_ID, Columns
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.utils.annotations import OldAPIStack
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY,
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune import CLIReporter
from ray.tune.result import TRAINING_ITERATION
from ray.rllib.core.rl_module.rl_module import RLModule

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.rllib.offline.dataset_reader import DatasetReader

from define_args import custom_args
from onnxruntime.tools import convert_onnx_models_to_ort as c2o
from pathlib import Path

import logging
import logging_setup

# Try to import zmq, but make it optional
try:
    import zmq
    zmq_available = True
except ImportError:
    zmq_available = False
    zmq = None


def on_sigterm(signum, frame):
    raise KeyboardInterrupt


def _get_current_onnx_model(module: RLModule,

                            *,
                            outdir: str = "model.onnx",
                            logger,
                            ):
    """
    Function to extract the policy model converted to ort.
    """
    # if os.path.exists(outdir):
    #     shutil.rmtree(outdir)
    assert module.get_ctor_args_and_kwargs()[1]["inference_only"]

    # act_dim = module.action_space.shape[0]
    # obs_dim = module.observation_space.shape[0]
    # dummy_obs = torch.randn(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(module,
                      {"batch": {"obs": torch.randn(1, *module.observation_space.shape)}},
                      outdir,
                      export_params=True)
    # convert .onnx to .ort (optimized for faster loading and inference in the minion)
    styles = [c2o.OptimizationStyle.Fixed]
    c2o.convert_onnx_models_to_ort(
        model_path_or_dir=Path(outdir),  # may also be a directory
        output_dir=None,  # None = same folder as the .onnx
        optimization_styles=styles,
        # target_platform="arm",  # Only in the Raspberry Pi
    )
    with open("model.ort", "rb") as f:
        ort_raw = f.read()

    return ort_raw


def run_rllib_shared_memory(
        base_config: "AlgorithmConfig",
        args: Optional[argparse.Namespace] = None,
        *,
        stop: Optional[Dict] = None,
        success_metric: Optional[Dict] = None,
        trainable: Optional[Type] = None,
        tune_callbacks: Optional[List] = None,
        keep_config: bool = False,
        keep_ray_up: bool = False,
        scheduler=None,
        progress_reporter=None,
) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    """Given an algorithm config and some command line args, runs an experiment.

    Use the custom_args function from the define_args.py script to generate "args".

    The function sets up an Algorithm object from the given config (altered by the
    contents of `args`), then runs the Algorithm via Tune (or manually, if
    `args.no_tune` is set to True) using the stopping criteria in `stop`.

    At the end of the experiment, if `args.as_test` is True, checks, whether the
    Algorithm reached the `success_metric` (if None, use `env_runners/
    episode_return_mean` with a minimum value of `args.stop_reward`).

    See https://github.com/ray-project/ray/tree/master/rllib/examples for an overview
    of all supported command line options.

    Args:
        base_config: The AlgorithmConfig object to use for this experiment. This base
            config will be automatically "extended" based on some of the provided
            `args`. For example, `args.num_env_runners` is used to set
            `config.num_env_runners`, etc...
        args: A argparse.Namespace object, ideally returned by calling
            `args = add_rllib_example_script_args()`. It must have the following
            properties defined: `stop_iters`, `stop_reward`, `stop_timesteps`,
            `no_tune`, `verbose`, `checkpoint_freq`, `as_test`. Optionally, for WandB
            logging: `wandb_key`, `wandb_project`, `wandb_run_name`.
        stop: An optional dict mapping ResultDict key strings (using "/" in case of
            nesting, e.g. "env_runners/episode_return_mean" for referring to
            `result_dict['env_runners']['episode_return_mean']` to minimum
            values, reaching of which will stop the experiment). Default is:
            {
            "env_runners/episode_return_mean": args.stop_reward,
            "training_iteration": args.stop_iters,
            "num_env_steps_sampled_lifetime": args.stop_timesteps,
            }
        success_metric: Only relevant if `args.as_test` is True.
            A dict mapping a single(!) ResultDict key string (using "/" in
            case of nesting, e.g. "env_runners/episode_return_mean" for referring
            to `result_dict['env_runners']['episode_return_mean']` to a single(!)
            minimum value to be reached in order for the experiment to count as
            successful. If `args.as_test` is True AND this `success_metric` is not
            reached with the bounds defined by `stop`, will raise an Exception.
        trainable: The Trainable sub-class to run in the tune.Tuner. If None (default),
            use the registered RLlib Algorithm class specified by args.algo.
        tune_callbacks: A list of Tune callbacks to configure with the tune.Tuner.
            In case `args.wandb_key` is provided, appends a WandB logger to this
            list.
        keep_config: Set this to True, if you don't want this utility to change the
            given `base_config` in any way and leave it as-is. This is helpful
            for those example scripts which demonstrate how to set config settings
            that are otherwise taken care of automatically in this function (e.g.
            `num_env_runners`).

    Returns:
        The last ResultDict from a --no-tune run OR the tune.Tuner.fit()
        results.
    """
    if args is None:
        parser = custom_args()
        args = parser.parse_args()

    # If run --as-release-test, --as-test must also be set.
    if args.as_release_test:
        args.as_test = True

    logger = logging.getLogger("MyRLApp.custom_runner")
    logger.info(f"custom_runner, PID={os.getpid()}")

    # Pin to CPU core if specified
    if args.cpu_core_learner is not None:
        from ray_primitives import pin_to_core
        pin_to_core(args.cpu_core_learner)
        logger.info(f"Pinned learner process to CPU core {args.cpu_core_learner}")

    # pass main driver PID down to EnvRunner

    logger.debug("custom_run: Started custom run function")

    # Initialize Ray.
    ray.init(
        num_cpus=args.num_cpus or None,
        local_mode=args.local_mode,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"RAY_DEBUG": "legacy"}},
    )

    logger.debug("custom_run: Concluded ray.init()")

    # Define one or more stopping criteria.
    if stop is None:
        stop = {
            f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
            f"{ENV_RUNNER_RESULTS}/{NUM_ENV_STEPS_SAMPLED_LIFETIME}": (
                args.stop_timesteps
            ),
            TRAINING_ITERATION: args.stop_iters,
        }

    config = base_config

    # Enhance the `base_config`, based on provided `args`.
    if not keep_config:
        # Set the framework.
        config.framework(args.framework)

        # Add an env specifier (only if not already set in config)?
        if args.env is not None and config.env is None:
            config.environment(args.env)

        # Disable the new API stack?
        if not args.enable_new_api_stack:
            config.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )

        # Define EnvRunner scaling and behavior.
        if args.num_env_runners is not None:
            config.env_runners(num_env_runners=args.num_env_runners)
        if args.num_envs_per_env_runner is not None:
            config.env_runners(num_envs_per_env_runner=args.num_envs_per_env_runner)
        if args.create_local_env_runner is not None:
            config.env_runners(create_local_env_runner=args.create_local_env_runner)

        # Define compute resources used automatically (only using the --num-learners
        # and --num-gpus-per-learner args).
        # New stack.
        if config.enable_rl_module_and_learner:
            if args.num_gpus is not None and args.num_gpus > 0:
                raise ValueError(
                    "--num-gpus is not supported on the new API stack! To train on "
                    "GPUs, use the command line options `--num-gpus-per-learner=1` and "
                    "`--num-learners=[your number of available GPUs]`, instead."
                )

            # Do we have GPUs available in the cluster?
            num_gpus_available = ray.cluster_resources().get("GPU", 0)
            # Number of actual Learner instances (including the local Learner if
            # `num_learners=0`).
            num_actual_learners = (
                                      args.num_learners
                                      if args.num_learners is not None
                                      else config.num_learners
                                  ) or 1  # 1: There is always a local Learner, if num_learners=0.
            # How many were hard-requested by the user
            # (through explicit `--num-gpus-per-learner >= 1`).
            num_gpus_requested = (args.num_gpus_per_learner or 0) * num_actual_learners
            # Number of GPUs needed, if `num_gpus_per_learner=None` (auto).
            num_gpus_needed_if_available = (
                                               args.num_gpus_per_learner
                                               if args.num_gpus_per_learner is not None
                                               else 1
                                           ) * num_actual_learners
            # Define compute resources used.
            config.resources(num_gpus=0)  # old API stack setting
            if args.num_learners is not None:
                config.learners(num_learners=args.num_learners)

            # User wants to use aggregator actors per Learner.
            if args.num_aggregator_actors_per_learner is not None:
                config.learners(
                    num_aggregator_actors_per_learner=(
                        args.num_aggregator_actors_per_learner
                    )
                )

            # User wants to use GPUs if available, but doesn't hard-require them.
            if args.num_gpus_per_learner is None:
                if num_gpus_available >= num_gpus_needed_if_available:
                    config.learners(num_gpus_per_learner=1)
                else:
                    config.learners(num_gpus_per_learner=0)
            # User hard-requires n GPUs, but they are not available -> Error.
            elif num_gpus_available < num_gpus_requested:
                raise ValueError(
                    "You are running your script with --num-learners="
                    f"{args.num_learners} and --num-gpus-per-learner="
                    f"{args.num_gpus_per_learner}, but your cluster only has "
                    f"{num_gpus_available} GPUs!"
                )

            # All required GPUs are available -> Use them.
            else:
                config.learners(num_gpus_per_learner=args.num_gpus_per_learner)

            # Set CPUs per Learner.
            if args.num_cpus_per_learner is not None:
                config.learners(num_cpus_per_learner=args.num_cpus_per_learner)

        # Old stack (override only if arg was provided by user).
        elif args.num_gpus is not None:
            config.resources(num_gpus=args.num_gpus)

        # Evaluation setup.
        if args.evaluation_interval > 0:
            config.evaluation(
                evaluation_num_env_runners=args.evaluation_num_env_runners,
                evaluation_interval=args.evaluation_interval,
                evaluation_duration=args.evaluation_duration,
                evaluation_duration_unit=args.evaluation_duration_unit,
                evaluation_parallel_to_training=args.evaluation_parallel_to_training,
            )

        # Set the log-level (if applicable).
        if args.log_level is not None:
            config.debugging(log_level=args.log_level)

        # Set the output dir (if applicable).
        if args.output is not None:
            config.offline_data(output=args.output)

    logger.debug("custom_run: Done with setting config. Going into args.no_tune")

    signal.signal(signal.SIGTERM, on_sigterm)

    # Run the experiment w/o Tune (directly operate on the RLlib Algorithm object).
    # THIS IS WHAT WILL BE RUN ON THE RASPBERRY PI
    if args.no_tune:
        assert not args.as_test and not args.as_release_test

        # create flag shared memory block here
        # flag buffer has 4 flags:
        #   0 -> weights lock flag (locked_state=1)
        #   1 -> weights available flag (true_state=1)
        #   2 -> minion data collection flag (has it started collecting data?, true_state=1)
        #   3 -> episode buffer lock flag (needed because of race conditions with reading and writing, locked_state=1)
        f_shm = shared_memory.SharedMemory(
            create=True,
            name=args.flag_shm_name,
            size=4,
        )
        f_buf = f_shm.buf

        f_buf[0] = 1  # set weights lock flag to locked
        f_buf[1] = 0  # set weights-available flag to false
        f_buf[2] = 0  # set minion flag to false (minion has not started collecting rollouts)
        f_buf[3] = 0  # set episode lock flag to unlocked

        logger.debug("custom_run: created flag memory buffer")

        # build algorithm, EnvRunner is created in this call
        algo = config.build()

        logger.debug("custom_run: done with config.build()")

        # extract dimensions of weights in the networks
        ort_raw = _get_current_onnx_model(algo.get_module(), logger=logger)
        policy_nbytes = len(ort_raw)

        logger.debug(f"custom_run: ort_raw length is {policy_nbytes}")

        # create policy shared memory blocks
        # need to include one more float32 as the buffer header to contain length of ort_compressed.
        # python creates the length of the buffer to be the smallest number of pages that can hold the requested number
        # of bytes, but not the size requested (on Mac at least)
        header_offset = 4
        p_shm = shared_memory.SharedMemory(
            create=True,
            name=args.policy_shm_name,
            size=policy_nbytes + header_offset
        )

        logger.debug("custom_run: created policy memory buffer")

        # get reference to policy buffer
        p_buf = p_shm.buf

        logger.debug(f"custom_run: buffer length is {len(p_buf)}")

        # store initial weights and remove lock flags
        struct.pack_into("<I", p_buf, 0, policy_nbytes)
        p_buf[header_offset:header_offset + len(ort_raw)] = ort_raw  # insert raw weights
        f_buf[0] = 0  # set lock flag to unlocked
        f_buf[1] = 1  # set weights-available flag to 1 (true)

        logger.debug("custom_run: stored initial model weights")

        results = None

        logger.debug("custom_run: waiting until minion starts collecting rollouts.")
        # wait until the minion has started collecting rollouts
        while f_buf[2] == 0:
            time.sleep(0.1)
        logger.debug("custom_run: minion is now collecting rollouts")

        # debugging
        # logger.debug(f"custom_run: circular_buffer_num_batches -> {algo.config.circular_buffer_num_batches}")
        # logger.debug(
        #     f"custom_run: circular_buffer_iterations_per_batch -> {algo.config.circular_buffer_iterations_per_batch}")
        # logger.debug(f"custom_run: target_network_update_freq -> {algo.config.target_network_update_freq}")
        # logger.debug(
        #     f"custom_run: num_aggregator_actors_per_learner -> {algo.config.num_aggregator_actors_per_learner}")
        # logger.debug(
        #     f"custom_run: num_envs_per_env_runner -> {algo.config.num_envs_per_env_runner}")
        # logger.debug(f"custom_run: _skip_learners -> {algo.config._skip_learners}")
        # logger.debug(f"custom_run: enable_rl_module_and_learner? {algo.config.enable_rl_module_and_learner}")
        # logger.debug(f"custom_run: broadcast_env_runner_states? {algo.config.broadcast_env_runner_states}")
        # logger.debug(f"custom_run: num_learners -> {algo.config.num_learners}")
        # logger.debug(f"custom_run: min_sample_timesteps_per_iteration -> {algo.config.min_sample_timesteps_per_iteration}")
        # logger.debug(f"custom_run: min_env_steps_per_iteration -> {algo.config.min_env_steps_per_iteration}")
        # logger.debug(f"custom_run: min_time_s_per_iteration -> {algo.config.min_time_s_per_iteration}")

        merge = (
                        not algo.config.enable_env_runner_and_connector_v2
                        and algo.config.use_worker_filter_stats
                ) or (
                        algo.config.enable_env_runner_and_connector_v2
                        and (
                                algo.config.merge_env_runner_states is True
                                or (
                                        algo.config.merge_env_runner_states == "training_only"
                                        and not algo.config.in_evaluation
                                )
                        )
                )
        broadcast = (
                            not algo.config.enable_env_runner_and_connector_v2
                            and algo.config.update_worker_filter_stats
                    ) or (
                            algo.config.enable_env_runner_and_connector_v2
                            and algo.config.broadcast_env_runner_states
                    )
        logger.debug(f"custom_run: merge -> {merge}")
        logger.debug(f"custom_run: broadcast -> {broadcast}")

        module = algo.get_module()
        dist_cls = module.get_inference_action_dist_cls()
        logger.debug(f"policy dist_class: {dist_cls}, {dist_cls.__name__}")

        # set up data broadcasting to GUI (optional)
        pub = None
        ctx = None
        if args.enable_zmq and zmq_available and zmq is not None:
            try:
                ctx = zmq.Context()
                pub = ctx.socket(zmq.PUB)
                pub.bind("ipc:///tmp/training.ipc")
                logger.info("ZMQ publisher initialized for GUI communication")
            except Exception as e:
                logger.warning(f"Failed to initialize ZMQ publisher: {e}. Continuing without ZMQ.")
                pub = None
                ctx = None
        elif args.enable_zmq and not zmq_available:
            logger.warning("ZMQ requested but not available (zmq not installed). Continuing without ZMQ.")
        else:
            logger.debug("ZMQ disabled via --enable-zmq flag")

        try:
            # start counter
            train_iter = 0
            while True:
                logger.debug("custom_run: in the train loop now.")

                # perform one logical iteration of training
                results = algo.train()

                state = algo.learner_group.get_state(components="learner")
                if 'metrics_logger' in state['learner']:
                    stats = state['learner']['metrics_logger']['stats']
                    try:
                        logger.debug(f"step {train_iter:>4}: "
                                     f"Qloss={list(stats['default_policy--qf_loss']['values'])} | "
                                     f"Ploss={list(stats['default_policy--policy_loss']['values'])} | "
                                     f"α={list(stats['default_policy--alpha_value']['values'])} "
                                     f"(αloss={list(stats['default_policy--alpha_loss']['values'])}) | "
                                     f"Qµ={list(stats['default_policy--qf_mean']['values'])}"
                                     )
                    except Exception as e:
                        logger.debug(f"could not print stats due to error {e}")

                seq_learn = state["learner"]["weights_seq_no"]

                logger.debug(f"custom_run: learner weights_seq_no="
                             f"{seq_learn}")

                # walk_keys(results)
                # logger.debug(f"custom_run: results.keys()={results.keys()}.")
                logger.debug("custom_run: printing BehaviourAudit.")
                try:
                    msg = {
                        "topic": "policy",
                        "ratio_max": float(results["env_runners"]["ratio_max"]),
                    }
                    # send results to be logged in the GUI
                    if pub is not None:
                        pub.send_json(msg)

                    logger.debug(f'ratio_max={results["env_runners"]["ratio_max"]}, '
                                 f'ratio_p99={results["env_runners"]["ratio_p99"]}, '
                                 f'delta_logp={results["env_runners"]["delta_logp"]}')
                except KeyError:
                    logger.debug("Could not find the keys in results dictionary")


                # attempt at debugging
                # logger.debug("custom_run: ran algo.train()")
                # target_updates = algo._counters["num_target_updates"]
                # last_update = algo._counters["last_target_update_ts"]
                # cur_ts = algo._counters[
                #     (
                #         "num_agent_steps_sampled"
                #         if algo.config.count_steps_by == "agent_steps"
                #         else "num_env_steps_sampled"
                #     )
                # ]
                # logger.debug(f"custom_run: enable_rl_module_and_learner? {algo.config.enable_rl_module_and_learner}")
                # logger.debug(f"custom_run: tentative update frequency: {algo.config.num_epochs * algo.config.minibatch_buffer_size}")
                # logger.debug(f"custom_run: update math: {cur_ts - last_update}")
                # logger.debug(f"custom_run: number of target updates: {target_updates}")
                # last_synch = algo.metrics.peek(
                #     "num_training_step_calls_since_last_synch_worker_weights",
                # )
                # logger.debug(f"custom_run: num training steps since last synch: {last_synch}")
                # logger.debug(f"custom_run: num_weights_broadcast -> {algo.metrics.peek('num_weight_broadcasts')}")

                msg = {"topic": "training", "iteration": train_iter}  # to send to GUI

                # print results
                if ENV_RUNNER_RESULTS in results:
                    mean_return = results[ENV_RUNNER_RESULTS].get(
                        EPISODE_RETURN_MEAN, np.nan
                    )
                    logger.debug(f"iter={train_iter} R={mean_return}")
                    msg.update({"mean_return": float(mean_return)})
                    # print(f"iter={train_iter} R={mean_return}", end="")
                if EVALUATION_RESULTS in results:
                    Reval = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][
                        EPISODE_RETURN_MEAN
                    ]
                    print(f" R(eval)={Reval}", end="")
                    msg.update({"eval_return": float(Reval)})
                print()

                # send results to be logged in the GUI
                if pub is not None:
                    pub.send_json(msg)

                # increment counter
                train_iter += 1

        except KeyboardInterrupt:
            if not keep_ray_up:
                # del f_arr
                ray.shutdown()

        return results

    # Run the experiment using Ray Tune.

    # Log results using WandB.
    tune_callbacks = tune_callbacks or []
    if hasattr(args, "wandb_key") and (
            args.wandb_key is not None or WANDB_ENV_VAR in os.environ
    ):
        wandb_key = args.wandb_key or os.environ[WANDB_ENV_VAR]
        project = args.wandb_project or (
                args.algo.lower() + "-" + re.sub("\\W+", "-", str(config.env).lower())
        )
        tune_callbacks.append(
            WandbLoggerCallback(
                api_key=wandb_key,
                project=project,
                upload_checkpoints=True,
                **({"name": args.wandb_run_name} if args.wandb_run_name else {}),
            )
        )
    # Auto-configure a CLIReporter (to log the results to the console).
    # Use better ProgressReporter for multi-agent cases: List individual policy rewards.
    if progress_reporter is None and args.num_agents > 0:
        progress_reporter = CLIReporter(
            metric_columns={
                **{
                    TRAINING_ITERATION: "iter",
                    "time_total_s": "total time (s)",
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "combined return",
                },
                **{
                    (
                        f"{ENV_RUNNER_RESULTS}/module_episode_returns_mean/" f"{pid}"
                    ): f"return {pid}"
                    for pid in config.policies
                },
            },
        )

    # Force Tuner to use old progress output as the new one silently ignores our custom
    # `CLIReporter`.
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

    # Run the actual experiment (using Tune).
    start_time = time.time()
    results = tune.Tuner(
        trainable or config.algo_class,
        param_space=config,
        run_config=tune.RunConfig(
            stop=stop,
            verbose=args.verbose,
            callbacks=tune_callbacks,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=args.checkpoint_at_end,
            ),
            progress_reporter=progress_reporter,
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent_trials,
            scheduler=scheduler,
        ),
    ).fit()
    time_taken = time.time() - start_time

    if not keep_ray_up:
        ray.shutdown()

    # Error out, if Tuner.fit() failed to run. Otherwise, erroneous examples might pass
    # the CI tests w/o us knowing that they are broken (b/c some examples do not have
    # a --as-test flag and/or any passing criteris).
    if results.errors:
        raise RuntimeError(
            "Running the example script resulted in one or more errors! "
            f"{[e.args[0].args[2] for e in results.errors]}"
        )

    # If run as a test, check whether we reached the specified success criteria.
    test_passed = False
    if args.as_test:
        # Success metric not provided, try extracting it from `stop`.
        if success_metric is None:
            for try_it in [
                f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
            ]:
                if try_it in stop:
                    success_metric = {try_it: stop[try_it]}
                    break
            if success_metric is None:
                success_metric = {
                    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
                }
        # Get maximum value of `metric` over all trials
        # (check if at least one trial achieved some learning, not just the final one).
        success_metric_key, success_metric_value = next(iter(success_metric.items()))
        best_value = max(
            row[success_metric_key] for _, row in results.get_dataframe().iterrows()
        )
        if best_value >= success_metric_value:
            test_passed = True
            print(f"`{success_metric_key}` of {success_metric_value} reached! ok")

        if args.as_release_test:
            trial = results._experiment_analysis.trials[0]
            stats = trial.last_result
            stats.pop("config", None)
            json_summary = {
                "time_taken": float(time_taken),
                "trial_states": [trial.status],
                "last_update": float(time.time()),
                "stats": stats,
                "passed": [test_passed],
                "not_passed": [not test_passed],
                "failures": {str(trial): 1} if not test_passed else {},
            }
            with open(
                    os.environ.get("TEST_OUTPUT_JSON", "/tmp/learning_test.json"),
                    "wt",
            ) as f:
                try:
                    json.dump(json_summary, f)
                # Something went wrong writing json. Try again w/ simplified stats.
                except Exception:
                    from ray.rllib.algorithms.algorithm import Algorithm

                    simplified_stats = {
                        k: stats[k] for k in Algorithm._progress_metrics if k in stats
                    }
                    json_summary["stats"] = simplified_stats
                    json.dump(json_summary, f)

        if not test_passed:
            raise ValueError(
                f"`{success_metric_key}` of {success_metric_value} not reached!"
            )

    return results
