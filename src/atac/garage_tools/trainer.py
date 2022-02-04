import os
import time
from collections import namedtuple
from dowel import logger, tabular
import numpy as np

from garage.trainer import Trainer as garageTrainer
from garage.trainer import TrainArgs, NotSetupError
from garage.experiment.experiment import dump_json
from .utils import read_attr_from_csv


class Trainer(garageTrainer):
    """ A modifed version of the Garage Trainer.

        This subclass adds
            1) a light saving mode to minimze the stroage usage (only saving the
               networks, not the trainer and the full algo.)
            2) a ignore_shutdown flag for running multiple experiments.
            3) a return_attr option.
            4) a cpu data collection mode.
            5) logging of sampling time.
            6) logging of current epoch index.
    """

    # Add a light saving mode to minimze the stroage usage.
    # Add return_attr, return_mode options.
    # Add a cpu data collection mode.
    def setup(self, algo, env,
              force_cpu_data_collection=False,
              save_mode='light',
              return_mode='average',
              return_attr='Evaluation/AverageReturn'):
        """Set up trainer for algorithm and environment.

        This method saves algo and env within trainer and creates a sampler.

        Note:
            After setup() is called all variables in session should have been
            initialized. setup() respects existing values in session so
            policy weights can be loaded before setup().

        Args:
            algo (RLAlgorithm): An algorithm instance. If this algo want to use
                samplers, it should have a `_sampler` field.
            env (Environment): An environment instance.
            save_mode (str): 'light' or 'full'
            return_mode (str): 'full', 'average', or 'last'
            return_attr (str): the name of the logged attribute

        """
        super().setup(algo, env)
        assert save_mode in ('light', 'full')
        assert return_mode in ('full', 'average', 'last')
        self.save_mode = save_mode
        self.return_mode = return_mode
        self.return_attr = return_attr
        self.force_cpu_data_collection = force_cpu_data_collection
        self._sampling_time = 0.

    # Add a light saving mode (which saves only policy and value functions of an algorithm)
    def save(self, epoch):
        """Save snapshot of current batch.

        Args:
            epoch (int): Epoch.

        Raises:
            NotSetupError: if save() is called before the trainer is set up.

        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup trainer before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        params['seed'] = self._seed
        params['train_args'] = self._train_args
        params['stats'] = self._stats

        if self.save_mode=='light':
            # Only save networks
            networks = self._algo.networks
            keys = []
            values = []
            for k, v in self._algo.__dict__.items():
                if v in networks:
                    keys.append(k)
                    values.append(v)

            AlgoData = namedtuple(type(self._algo).__name__+'Networks',
                                  field_names=keys,
                                  defaults=values,
                                  rename=True)
            params['algo'] = AlgoData()

        elif self.save_mode=='full':
            # Default behavior: save everything
            # Save states
            params['env'] = self._env
            params['algo'] = self._algo
            params['n_workers'] = self._n_workers
            params['worker_class'] = self._worker_class
            params['worker_args'] = self._worker_args
        else:
            raise ValueError('Unknown save_mode.')

        self._snapshotter.save_itr_params(epoch, params)

        logger.log('Saved')

    # Include ignore_shutdown flag
    def train(self,
              n_epochs,
              batch_size=None,
              plot=False,
              store_episodes=False,
              pause_for_plot=False,
              ignore_shutdown=False):
        """Start training.

        Args:
            n_epochs (int): Number of epochs.
            batch_size (int or None): Number of environment steps in one batch.
            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
            pause_for_plot (bool): Pause for plot.

        Raises:
            NotSetupError: If train() is called before setup().

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._has_setup:
            raise NotSetupError(
                'Use setup() to setup trainer before training.')

        # Save arguments for restore
        self._train_args = TrainArgs(n_epochs=n_epochs,
                                     batch_size=batch_size,
                                     plot=plot,
                                     store_episodes=store_episodes,
                                     pause_for_plot=pause_for_plot,
                                     start_epoch=0)

        self._plot = plot
        self._start_worker()

        log_dir = self._snapshotter.snapshot_dir
        if self.save_mode !='light':
            summary_file = os.path.join(log_dir, 'experiment.json')
            dump_json(summary_file, self)

        # Train the agent
        last_return = self._algo.train(self)

        # XXX Ignore shutdown, if needed
        if not ignore_shutdown:
            self._shutdown_worker()

        # XXX Return other statistics from logged data
        csv_file = os.path.join(log_dir,'progress.csv')
        progress = read_attr_from_csv(csv_file, self.return_attr)
        progress = progress if progress is not None else 0
        if self.return_mode == 'average':
            score = np.mean(progress)
        elif self.return_mode == 'full':
            score = progress
        elif self.return_mode == 'last':
            score = last_return
        else:
            NotImplementedError
        return score

    # Add a cpu data collection mode
    def obtain_episodes(self,
                        itr,
                        batch_size=None,
                        agent_update=None,
                        env_update=None):
        """Obtain one batch of episodes.

        Args:
            itr (int): Index of iteration (epoch).
            batch_size (int): Number of steps in batch. This is a hint that the
                sampler may or may not respect.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before doing sampling episodes. If a list is
                passed in, it must have length exactly `factory.n_workers`, and
                will be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Raises:
            ValueError: If the trainer was initialized without a sampler, or
                batch_size wasn't provided here or to train.

        Returns:
            EpisodeBatch: Batch of episodes.

        """
        if self._sampler is None:
            raise ValueError('trainer was not initialized with `sampler`. '
                             'the algo should have a `_sampler` field when'
                             '`setup()` is called')
        if batch_size is None and self._train_args.batch_size is None:
            raise ValueError(
                'trainer was not initialized with `batch_size`. '
                'Either provide `batch_size` to trainer.train, '
                ' or pass `batch_size` to trainer.obtain_samples.')
        episodes = None
        if agent_update is None:
            policy = getattr(self._algo, 'exploration_policy', None)
            if policy is None:
                # This field should exist, since self.make_sampler would have
                # failed otherwise.
                policy = self._algo.policy
            agent_update = policy.get_param_values()

        # XXX Move the tensor to cpu.
        if self.force_cpu_data_collection:
            for k,v in agent_update.items():
                if v.device.type != 'cpu':
                    agent_update[k] = v.to('cpu')

        # XXX Time data collection.
        _start_sampling_time = time.time()
        episodes = self._sampler.obtain_samples(
            itr, (batch_size or self._train_args.batch_size),
            agent_update=agent_update,
            env_update=env_update)
        self._sampling_time = time.time() - _start_sampling_time
        self._stats.total_env_steps += sum(episodes.lengths)
        return episodes

    # Log sampling time and Epoch
    def log_diagnostics(self, pause_for_plot=False):
        """Log diagnostics.

        Args:
            pause_for_plot (bool): Pause for plot.

        """
        logger.log('Time %.2f s' % (time.time() - self._start_time))
        logger.log('EpochTime %.2f s' % (time.time() - self._itr_start_time))  # XXX
        logger.log('SamplingTime %.2f s' % (self._sampling_time))
        tabular.record('TotalEnvSteps', self._stats.total_env_steps)
        tabular.record('Epoch', self.step_itr)
        logger.log(tabular)

        if self._plot:
            self._plotter.update_plot(self._algo.policy,
                                      self._algo.max_episode_length)
            if pause_for_plot:
                input('Plotting evaluation run: Press Enter to " "continue...')

class BatchTrainer(Trainer):
    """ A batch version of Trainer that disables environment sampling. """

    def obtain_samples(self,
                       itr,
                       batch_size=None,
                       agent_update=None,
                       env_update=None):
        """ Return an empty list. """
        return []