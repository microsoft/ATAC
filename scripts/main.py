import gym, d4rl, torch, os

import numpy as np
from urllib.error import HTTPError
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.replay_buffer import PathBuffer
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


from atac.atac import ATAC
from atac.garage_tools.rl_utils import train_agent, get_sampler, setup_gpu, get_algo, get_log_dir_name, load_algo
from atac.garage_tools.trainer import Trainer


def load_d4rl_data_as_buffer(dataset, replay_buffer):
    assert isinstance(replay_buffer, PathBuffer)
    replay_buffer.add_path(
        dict(observation=dataset['observations'],
             action=dataset['actions'],
             reward=dataset['rewards'].reshape(-1, 1),
             next_observation=dataset['next_observations'],
             terminal=dataset['terminals'].reshape(-1,1),
    ))

def train_func(ctxt=None,
               *,
               algo='ATAC',
               # Environment parameters
               env_name,
               # Evaluation mode
               evaluation_mode=False,
               policy_path=None,
               # Trainer parameters
               n_epochs=3000,  # number of training epochs
               batch_size=0,  # number of samples collected per update
               replay_buffer_size=int(2e6),
               # Network parameters
               policy_hidden_sizes=(256, 256, 256),
               policy_activation='ReLU',
               policy_init_std=1.0,
               value_hidden_sizes=(256, 256, 256),
               value_activation='ReLU',
               min_std=1e-5,
               # Algorithm parameters
               discount=0.99,
               policy_lr=5e-7,  # optimization stepsize for policy update
               value_lr=5e-4,  # optimization stepsize for value regression
               target_update_tau=5e-3, # for target network
               minibatch_size=256,  # optimization/replaybuffer minibatch size
               n_grad_steps=2000,  # number of gradient updates per epoch
               n_warmstart_steps=200000,  # number of warm-up steps
               fixed_alpha=None,  # whether to fix the temperate parameter
               use_deterministic_evaluation=True,  # do evaluation based on the deterministic policy
               num_evaluation_episodes=5, # number of episodes to evaluate (only affect off-policy algorithms)
               # ATAC parameters
               beta=1.0,  # weight on the Bellman error
               norm_constraint=100,
               use_two_qfs=True,  # whether to use two q function
               q_eval_mode='0.5_0.5',
               init_pess=False,
               # Compute parameters
               seed=0,
               n_workers=1,  # number of workers for data collection
               gpu_id=-1,  # try to use gpu, if implemented
               force_cpu_data_collection=True,  # use cpu for data collection.
               # Logging parameters
               save_mode='light',
               ignore_shutdown=False,  # do not shutdown workers after training
               return_mode='average', # 'full', 'average', 'last'
               return_attr='Evaluation/AverageReturn',  # the log attribute
               ):

    """ Train an agent in batch mode. """

    # Set the random seed
    set_seed(seed)

    # Initialize gym env
    dataset = None
    d4rl_env = gym.make(env_name)  # d4rl env
    while dataset is None:
        try:
            dataset = d4rl.qlearning_dataset(d4rl_env)
        except (HTTPError, OSError):
            print('Unable to download dataset. Retry.')
            pass

    if init_pess:  # for ATAC0
        dataset_raw = d4rl_env.get_dataset()
        ends = dataset_raw['terminals']+ dataset_raw['timeouts']
        starts = np.concatenate([[True], ends[:-1]])
        init_observations = dataset_raw['observations'][starts]
    else:
        init_observations = None

    # Initialize replay buffer and gymenv
    env = GymEnv(d4rl_env)
    replay_buffer = PathBuffer(capacity_in_transitions=int(replay_buffer_size))
    load_d4rl_data_as_buffer(dataset, replay_buffer)
    reward_scale = 1.0

    # Initialize the algorithm
    env_spec = env.spec

    policy = TanhGaussianMLPPolicy(
                env_spec=env_spec,
                hidden_sizes=policy_hidden_sizes,
                hidden_nonlinearity=eval('torch.nn.'+policy_activation),
                init_std=policy_init_std,
                min_std=min_std)

    qf1 = ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=value_hidden_sizes,
                hidden_nonlinearity=eval('torch.nn.'+value_activation),
                output_nonlinearity=None)

    qf2 = ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=value_hidden_sizes,
                hidden_nonlinearity=eval('torch.nn.'+value_activation),
                output_nonlinearity=None)

    sampler = get_sampler(policy, env, n_workers=n_workers)

    Algo = globals()[algo]

    algo_config = dict(  # union of all algorithm configs
                env_spec=env_spec,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                sampler=sampler,
                replay_buffer=replay_buffer,
                discount=discount,
                policy_lr=policy_lr,
                qf_lr=value_lr,
                target_update_tau=target_update_tau,
                buffer_batch_size=minibatch_size,
                gradient_steps_per_itr=n_grad_steps,
                use_deterministic_evaluation=use_deterministic_evaluation,
                min_buffer_size=int(0),
                num_evaluation_episodes=num_evaluation_episodes,
                fixed_alpha=fixed_alpha,
                reward_scale=reward_scale,
    )

    # ATAC
    extra_algo_config = dict(
        beta=beta,
        norm_constraint=norm_constraint,
        use_two_qfs=use_two_qfs,
        n_warmstart_steps=n_warmstart_steps,
        q_eval_mode=q_eval_mode,
        init_observations=init_observations,
        )

    algo_config.update(extra_algo_config)

    algo = Algo(**algo_config)

    setup_gpu(algo, gpu_id=gpu_id)

    # Initialize the trainer
    from atac.garage_tools.trainer import BatchTrainer as Trainer
    trainer = Trainer(ctxt)
    trainer.setup(algo=algo,
                  env=env,
                  force_cpu_data_collection=force_cpu_data_collection,
                  save_mode=save_mode,
                  return_mode=return_mode,
                  return_attr=return_attr)

    return trainer.train(n_epochs=n_epochs,
                         batch_size=batch_size,
                         ignore_shutdown=ignore_shutdown)


def run(log_root='.',
        torch_n_threads=2,
        snapshot_frequency=0,
        **train_kwargs):
    torch.set_num_threads(torch_n_threads)
    log_dir = get_log_dir_name(train_kwargs, ['beta', 'discount', 'norm_constraint',
                                                'policy_lr', 'value_lr',
                                                'use_two_qfs',
                                                'fixed_alpha',
                                                'q_eval_mode',
                                                'n_warmstart_steps', 'seed'])
    train_kwargs['return_mode'] = 'full'

    # Offline training
    log_dir_path = os.path.join(log_root,'exp_data','Offline'+train_kwargs['algo']+'_'+train_kwargs['env_name'], log_dir)
    full_score =  train_agent(train_func,
                    log_dir=log_dir_path,
                    train_kwargs=train_kwargs,
                    snapshot_frequency=snapshot_frequency,
                    x_axis='Epoch')

    window = 50
    score = np.median(full_score[-min(len(full_score),window):])
    print('Median of performance of last {} epochs'.format(window), score)
    return {'score': score,  # last 50 epochs
            'mean': np.mean(full_score)}

if __name__=='__main__':
    import argparse
    from atac.garage_tools.utils import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='ATAC')
    parser.add_argument('-e', '---env_name',  type=str, default='hopper-medium-replay-v2')
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gpu_id', type=int, default=-1)  # use cpu by default
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--force_cpu_data_collection', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_warmstart_steps', type=int, default=100000)
    parser.add_argument('--fixed_alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=16)
    parser.add_argument('--norm_constraint', type=float, default=100)
    parser.add_argument('--policy_lr', type=float, default=5e-7)
    parser.add_argument('--value_lr', type=float, default=5e-4)
    parser.add_argument('--target_update_tau', type=float, default=5e-3)
    parser.add_argument('--use_deterministic_evaluation', type=str2bool, default=True)
    parser.add_argument('--use_two_qfs', type=str2bool, default=True)
    parser.add_argument('--q_eval_mode', type=str, default='0.5_0.5')
    parser.add_argument('--init_pess', type=str2bool, default=False)  # turn this on of ATAC0

    train_kwargs = vars(parser.parse_args())
    run(**train_kwargs)