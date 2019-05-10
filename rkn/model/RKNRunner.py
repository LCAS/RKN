import numpy as np
import tensorflow as tf

from util import GPUUtil as gpu_util
from model import RKN
import time
from tensorflow.python.client import timeline
import os

class RKNRunner:

    def __init__(self, model, log_options=None):
        """ Helper class to run, i.e train, evaluate and use, the RKN model
        :param model: The model to run
        """
        self.model = model
        self.c = model.c
        #defaluts
        self.logger = None
        self.complete_summary = None

        self.log_options = log_options

        self._initialize()
        self._train_fn = self._train_step

    def _initialize(self):
        """ Initialize the session
        Todo: implement logging, saving and loading of parameters
        """
        self._track_runtime = False

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.log_device_placement = False

        self.c.mprint("Initializing model...")
        self.tf_session = tf.Session(config=config)
        self.c.mprint("Initialize with random parameters")
        self.tf_session.run(tf.global_variables_initializer())
        self.c.mprint("...model successfully initialized")

    def initialize_logger(self, log_path, scalars_dict={}, historgrams_dict={}, track_runtime=False):
        """ Initializes a logger
        :param log_path: path to write the logs to
        :param scalars_dict: dictionary of {name: tensor} pairs, logged as scalars
        :param historgrams_dict: dictionary of {name: tensor} pairs, logged as histograms
        """
        self.c.mprint("Initializing Logging, saving to", log_path)
        self.logger = tf.summary.FileWriter(log_path, self.tf_session.graph)

        self._track_runtime = track_runtime

        scalar_logs = [tf.summary.scalar(name, tensor) for name, tensor in scalars_dict.items()]
        if len(scalar_logs) == 0:
            scalar_logs.append(tf.summary.scalar("loss", self.model.loss[0]))
        histogram_logs = [tf.summary.histogram(name, tensor) for name, tensor in historgrams_dict.items()]

        self._log_step = 0

        self.complete_summary = tf.summary.merge_all()

        self._train_fn = self._train_step_with_logging

    def train(self,
              observations,
              targets,
              training_epochs,
              initial_latent_state=None,
              observations_valid=None,
              verbose_interval=1):
        """ Trains the model
        :param observations: Input observations for training
        :param targets: Targets for training, either images or positions -> depending on mode
        :param training_epochs: Number of training epochs
        :param initial_latent_state: Initial latent state - default depends on initialization mode
        (see RKNTransitionCell for details)
        :param observations_valid: Indicating which observations are valid - default: all valid
        :param verbose_interval: How often to print the current training loss
        """
        t0 = time.time()
        self.c.mprint("Start training for", training_epochs, "epochs ...")
        if self.model.use_gpu:
            observations = gpu_util.put_channels_first(observations)
            if self.model.output_mode == RKN.OUTPUT_MODE_OBSERVATIONS:
                targets = gpu_util.put_channels_first(targets)

        num_of_seq = np.shape(targets)[0]
        sequence_length = np.shape(targets)[1]

        if num_of_seq % self.c.batch_size != 0:
            tf.logging.warn("Amount of sequences not divided by batch_size - ignoring a few elements each epoch")
        if sequence_length % self.c.bptt_length != 0:
            tf.logging.warn("Sequence length not divided by bptt_length - last part shorter")
        num_of_batches = int(np.floor(num_of_seq / self.c.batch_size))
        seq_parts = int(np.ceil(sequence_length / self.c.bptt_length))
        t1 = time.time()
        for epoch in range(training_epochs):
            avg_loss = 0
            shuffled_idx = np.random.permutation(num_of_seq)
            for i in range(num_of_batches):
                batch_slice = shuffled_idx[i * self.c.batch_size: (i + 1) * self.c.batch_size]
                latent_state = initial_latent_state[batch_slice, :] if initial_latent_state is not None else None
                for j in range(seq_parts):
                    seq_slice = slice(j * self.c.bptt_length, (j + 1) * self.c.bptt_length)
                    loss, latent_state = \
                        self._train_on_batch(observation_batch=observations[batch_slice, seq_slice, :],
                                             target_batch=targets[batch_slice, seq_slice, :],
                                             num_of_batch=i,
                                             num_of_seq_part=j,
                                             initial_latent_state=latent_state,
                                             observation_valid_batch=observations_valid[batch_slice, seq_slice, :] if observations_valid is not None else None)
                   # print(loss)
                    avg_loss += np.array(loss) / (num_of_batches * seq_parts)
                    if np.any(np.isnan(avg_loss)):
                        self.c.mprint("Loss = NaN - abort")
                        return None

            if (epoch + 1) % verbose_interval == 0:
                self.c.mprint('Epoch', (epoch + 1), ': Loss', avg_loss)
                tf.logging.warn(self.c.name + " Epoch " + str(epoch + 1) + ": Loss " + str(avg_loss))
        t2 = time.time()
        self.c.mprint("... finished training - total time:", t2 - t0, "Without Preprocessing", t2 - t1)
        tf.logging.warn(self.c.name + ": ... finished training - total time: " + str(t2 - t0) + " Without Preprocessing " + str(t2 - t1))

    def _train_on_batch(self,
                        observation_batch,
                        target_batch,
                        num_of_batch,
                        num_of_seq_part,
                        observation_valid_batch=None,
                        initial_latent_state=None):
        """ Runs training on a single batch
        :param observation_batch: Batch of observations
        :param target_batch: Batch of targets
        :param num_of_batch: Number of current batch
        :param num_of_seq_part: Number of current sequence part
        :param observation_valid_batch: Batch of flags indicating if observations are valid - default: all true
        :param initial_latent_state: batch of initial latent states - default depends on initialization mode
        (see RKNTransitionCell for details)
        """
        feed_dict = self._feed_inputs(observations=observation_batch,
                                      targets=target_batch,
                                      observation_valid=observation_valid_batch,
                                      initial_latent_state=initial_latent_state)
        feed_dict[self.model.training] = True
        if self._track_runtime and num_of_batch == num_of_seq_part == 0:
            return self._train_step_with_runtime(feed_dict=feed_dict)
        else:
            return self._train_fn(feed_dict=feed_dict)

    def _train_step(self, feed_dict):
        """ Single training step
        :param feed_dict: dictionary fed to session """
        fetches = [self.model.optimizer, self.model.loss, self.model.last_state] #+ self.model.debug_vars
        out_list = self.tf_session.run(fetches=fetches, feed_dict=feed_dict)
        loss = out_list[1]
        last_latent = out_list[2]
        #print(*zip(self.model.debug_vars_labels, out_list[3:]))
        return loss, last_latent

    def _train_step_with_logging(self, feed_dict):
        """ Single training step - tensors specified during initialization of logging are logged
        :param feed_dict: dictionary fed to session """
        assert self.logger is not None and self.complete_summary is not None, \
            "It seems like logging was requested without calling 'initialize_logger' first"
        _, loss, log_summary,  last_latent = self.tf_session.run(fetches=(self.model.optimizer,
                                                                          self.model.loss,
                                                                          self.complete_summary,
                                                                          self.model.last_state),
                                                                 feed_dict=feed_dict)
        self._log_step += 1
        self.logger.add_summary(log_summary, global_step=self._log_step)
        self.logger.flush()
        return loss, last_latent

    def _train_step_with_runtime(self, feed_dict):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _, loss, log_summary, last_latent = self.tf_session.run(fetches=(self.model.optimizer,
                                                                         self.model.loss,
                                                                         self.complete_summary,
                                                                         self.model.last_state),
                                                   feed_dict=feed_dict,
                                                   run_metadata=run_metadata,
                                                   options=run_options)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        if not os.path.exists("rt_timelines"):
            os.makedirs("rt_timelines")

        filename = "timeline{:05d}".format(self._log_step)
        with open(os.path.join("rt_timelines", filename), 'w') as f:
            f.write(chrome_trace)

        self.logger.add_summary(log_summary)
        self.logger.add_run_metadata(run_metadata, "step: %d" % self._log_step, global_step=self._log_step)
        self.logger.flush()
        self._log_step += 1
        return loss, last_latent

    def _feed_inputs(self, observations, targets=None, initial_latent_state=None, observation_valid=None):
        """Creates the feed - handles the defaults for 'initial_latent_state' and 'observation_valid'
        :param observations: Observations to feed in
        :param targets: Targets to feed in
        :param initial_latent_state: Initial latent states to feed in
        :param observation_valid: Observation valid flags to feed in
        """

        if observation_valid is None:
            obs_shape = np.shape(observations)
            observation_valid = np.ones([obs_shape[0], obs_shape[1], 1], dtype=np.bool)

        feed_dict = {self.model.observations: observations,
                     self.model.observations_valid: observation_valid}

        if initial_latent_state is not None:
            feed_dict[self.model.initial_latent_state] = initial_latent_state
        if targets is not None:
            feed_dict[self.model.targets] = targets

        return feed_dict

    def evaluate(self,
                 observations,
                 targets,
                 observation_valid=None,
                 initial_latent_state=None,
                 test_batch_size=-1):
        """
        Evaluate the model
        :param observations: Observations for evaluation
        :param targets: Targets for evaluation
        :param observation_valid: Flags indicating if observation valid
        :param initial_latent_state: Initial latent state - default depending on mode
        :param test_batch_size: Size of the test batches, default -1 -> batch size = number of sequences only needed if test set to large for (gpu) memory
        """
        self.c.mprint("Start evaluation ...", end=" ")
        if self.model.use_gpu:
            observations = gpu_util.put_channels_first(observations)
            if self.model.output_mode == RKN.OUTPUT_MODE_OBSERVATIONS:
                targets = gpu_util.put_channels_first(targets)

        if test_batch_size < 0:
            test_batch_size = targets.shape[0]
        num_batches = int(targets.shape[0] / test_batch_size)

        avg_loss = 0
        for i in range(num_batches):
            cur_slice = slice(i * test_batch_size, (i + 1) * test_batch_size)
            observation_valid_batch = None if observation_valid is None else observation_valid[cur_slice]
            initial_latent_state_batch = None if initial_latent_state is None else initial_latent_state[cur_slice]
            feed_dict = self._feed_inputs(observations=observations[cur_slice],
                                          targets=targets[cur_slice],
                                          observation_valid=observation_valid_batch,
                                          initial_latent_state=initial_latent_state_batch)
            feed_dict[self.model.training] = False
            loss = self.tf_session.run(fetches=self.model.loss, feed_dict=feed_dict)
            avg_loss += np.array(loss) / num_batches
        self.c.mprint("... Evaluation finished, Loss:", avg_loss)
        tf.logging.warn(self.c.name + ": ... Evaluation finished, Loss: " + str(avg_loss))
        return avg_loss

    def predict(self,
                observations,
                initial_latent_state=None,
                observation_valid=None,
                additional_fetch=[]):
        """ Runs Inference
        :param observations: Model inputs
        :param initial_latent_state: Initial latent state - default depending on mode
        :param observation_valid: Flags indicating which observations are valid - default: all valid
        :param additional_fetch: list of additional tensors to fetch and return
        :return: predicted model outputs, values of tensors in additional fetch
        """
        self.c.mprint("Start prediction...")
        if self.model.use_gpu:
            observations = gpu_util.put_channels_first(observations)

        feed_dict = self._feed_inputs(observations,
                                      initial_latent_state=initial_latent_state,
                                      observation_valid=observation_valid)
        feed_dict[self.model.training] = False
        res = self.tf_session.run(fetches=[self.model.predictions] + additional_fetch, feed_dict=feed_dict)
        self.c.mprint("... prediction finished")

        return res[0] if len(res) == 1 else (res[0], res[1:])

