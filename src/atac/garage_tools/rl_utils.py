# Some helper functions for using garage


import numpy as np
import torch

from garage.sampler import FragmentWorker, LocalSampler, RaySampler
from garage.torch import prefer_gpu
from garage.torch._functions import set_gpu_mode


### Run experiments
from garage import wrap_experiment

def get_log_dir_name(config, keys=None):
    keys = keys or list(config.keys())
    log_dir = ''
    for k in keys:
        if k in config:
            log_dir = log_dir + '_' + k + '_' + str(config[k])
    log_dir = log_dir[1:]
    return log_dir

def get_snapshot_info(snapshot_frequency):
    snapshot_gap = snapshot_frequency if snapshot_frequency>0 else 1
    snapshot_mode = 'gap_and_last' if snapshot_frequency>0 else 'last'
    return snapshot_gap, snapshot_mode

def train_agent(train_func,
                *,
                train_kwargs,
                log_dir=None,
                snapshot_frequency=0,
                use_existing_dir=True,
                x_axis='TotalEnvSteps',
                ):
    """ A helper method to run experiments in garage. """
    snapshot_gap, snapshot_mode = get_snapshot_info(snapshot_frequency)
    save_mode = train_kwargs.get('save_mode', 'light')
    wrapped_train_func = wrap_experiment(train_func,
                          log_dir=log_dir,  # overwrite
                          snapshot_mode=snapshot_mode,
                          snapshot_gap=snapshot_gap,
                          archive_launch_repo=save_mode!='light',
                          use_existing_dir=use_existing_dir,
                          x_axis=x_axis)  # overwrites existing directory
    score = wrapped_train_func(**train_kwargs)
    return score

def load_algo(path, itr='last'):
    from garage.experiment import Snapshotter
    snapshotter = Snapshotter()
    data = snapshotter.load(path, itr=itr)
    return data['algo']

def setup_gpu(algo, gpu_id=-1):
    if gpu_id>=0:
        set_gpu_mode(torch.cuda.is_available(), gpu_id=gpu_id)
        if callable(getattr(algo, 'to', None)):
            algo.to()


def collect_episode_batch(policy, *,
                          env,
                          batch_size,
                          sampler_mode='ray',
                          n_workers=4):
    """Obtain one batch of episodes."""
    sampler = get_sampler(policy, env=env, sampler_mode=sampler_mode, n_workers=n_workers)
    agent_update = policy.get_param_values()
    episodes = sampler.obtain_samples(0, batch_size, agent_update)
    return episodes

from garage.sampler import Sampler
import copy
from garage._dtypes import EpisodeBatch
class BatchSampler(Sampler):

    def __init__(self, episode_batch, randomize=True):
        self.episode_batch = episode_batch
        self.randomize = randomize
        self._counter = 0

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):

        ns = self.episode_batch.lengths
        if num_samples<np.sum(ns):
            if self.randomize:
                # Sample num_samples from episode_batch
                ns = self.episode_batch.lengths
                ind = np.random.permutation(len(ns))
                cumsum_permuted_ns = np.cumsum(ns[ind])
                itemindex = np.where(cumsum_permuted_ns>=num_samples)[0]
                if len(itemindex)>0:
                    ld = self.episode_batch.to_list()
                    j_max = min(len(ld), itemindex[0]+1)
                    ld = [ld[i] for i in ind[:j_max].tolist()]
                    sampled_eb = EpisodeBatch.from_list(self.episode_batch.env_spec,ld)
                else:
                    sampled_eb = None
            else:
                ns = self.episode_batch.lengths
                ind = np.arange(len(ns))
                cumsum_permuted_ns = np.cumsum(ns[ind])
                counter = int(self._counter)
                itemindex = np.where(cumsum_permuted_ns>=num_samples*(counter+1))[0]
                itemindex0 = np.where(cumsum_permuted_ns>num_samples*counter)[0]
                if len(itemindex)>0:
                    ld = self.episode_batch.to_list()
                    j_max = min(len(ld), itemindex[0]+1)
                    j_min = itemindex0[0]
                    ld = [ld[i] for i in ind[j_min:j_max].tolist()]
                    sampled_eb = EpisodeBatch.from_list(self.episode_batch.env_spec,ld)
                    self._counter+=1
                else:
                    sampled_eb = None
        else:
            sampled_eb = self.episode_batch

        return sampled_eb

    def shutdown_worker(self):
        pass


from garage.sampler import DefaultWorker, VecWorker
def get_sampler(policy, env,
                n_workers=4):
    if n_workers>1:
        return RaySampler(agents=policy,
                          envs=env,
                          max_episode_length=env.spec.max_episode_length,
                          n_workers=n_workers)
    else:
        return LocalSampler(agents=policy,
                            envs=env,
                            max_episode_length=env.spec.max_episode_length,
                            worker_class=FragmentWorker,
                            n_workers=n_workers)


def get_algo(Algo, algo_config):
    import inspect
    algospec = inspect.getfullargspec(Algo)
    allowed_args = algospec.args + algospec.kwonlyargs
    for k in list(algo_config.keys()):
        if k not in allowed_args:
            del algo_config[k]
    algo = Algo(**algo_config)
    return algo