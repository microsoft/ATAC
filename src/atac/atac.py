# yapf: disable
from collections import deque
import copy

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import log_performance, obtain_evaluation_episodes, StepType
from garage.np.algos import RLAlgorithm
from garage.torch import as_torch_dict, global_device
# yapf: enable

torch.set_flush_denormal(True)

def normalized_sum(loss, reg, w):
    return loss/w + reg if w>1 else loss + w*reg

def l2_projection(constraint):
    @torch.no_grad()
    def fn(module):
        if hasattr(module, 'weight') and constraint>0:
            w = module.weight
            norm = torch.norm(w)
            w.mul_(torch.clip(constraint/norm, max=1))
    return fn

def weight_l2(model):
    l2 = 0.
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2 += torch.norm(param)**2
    return l2

class ATAC(RLAlgorithm):
    """ Adversarilly Trained Actor Critic """
    def __init__(
            self,
            env_spec,
            policy,
            qf1,
            qf2,
            replay_buffer,
            sampler,
            *,  # Everything after this is numbers.
            max_episode_length_eval=None,
            gradient_steps_per_itr,
            fixed_alpha=None,
            target_entropy=None,
            initial_log_entropy=0.,
            discount=0.99,
            buffer_batch_size=256,
            min_buffer_size=int(1e4),
            target_update_tau=5e-3,
            policy_lr=5e-7,
            qf_lr=5e-4,
            reward_scale=1.0,
            optimizer='Adam',
            steps_per_epoch=1,
            num_evaluation_episodes=10,
            eval_env=None,
            use_deterministic_evaluation=True,
            # ATAC parameters
            beta=1.0,  # the regularization coefficient in front of the Bellman error
            lambd=0., # coeff for global pessimism
            init_observations=None, # for ATAC0 (None or np.ndarray)
            n_warmstart_steps=200000,
            norm_constraint=100,
            q_eval_mode='0.5_0.5', # 'max' 'w1_w2', 'adaptive'
            q_eval_loss='MSELoss', # 'MSELoss', 'SmoothL1Loss'
            use_two_qfs=True,
            terminal_value=None,
            Vmin=-float('inf'), # min value of Q (used in target backup)
            Vmax=float('inf'), # max value of Q (used in target backup)
            debug=True,
            stats_avg_rate=0.99, # for logging
            bellman_surrogate='td', #'td', None, 'target'
            ):

        #############################################################################################

        assert beta>=0
        assert norm_constraint>=0
        # Parsing
        optimizer = eval('torch.optim.'+optimizer)
        policy_lr = qf_lr if policy_lr is None or policy_lr < 0 else policy_lr # use shared lr if not provided.

        ## ATAC parameters
        self.beta = torch.Tensor([beta]) # regularization constant on the Bellman surrogate
        self._lambd = torch.Tensor([lambd])  # global pessimism coefficient
        self._init_observations = torch.Tensor(init_observations) if init_observations is not None else init_observations  # if provided, it runs ATAC0
        self._n_warmstart_steps = n_warmstart_steps  # during which, it performs independent C and Bellman minimization
        # q update parameters
        self._norm_constraint = norm_constraint  # l2 norm constraint on the qf's weight; if negative, it gives the weight decay coefficient.
        self._q_eval_mode = [float(w) for w in q_eval_mode.split('_')] if '_' in q_eval_mode else  q_eval_mode  # residual algorithm
        self._q_eval_loss = eval('torch.nn.'+q_eval_loss)(reduction='none')
        self._use_two_qfs = use_two_qfs
        self._Vmin = Vmin  # lower bound on the target
        self._Vmax = Vmax  # upper bound on the target
        self._terminal_value = terminal_value if terminal_value is not None else lambda r, gamma: 0.

        # Stepsizes
        self._alpha_lr = qf_lr # potentially a larger stepsize, for the most inner optimization.
        self._bc_policy_lr = qf_lr  # potentially a larger stepsize

        # Logging and algorithm state
        self._debug = debug
        self._n_updates_performed = 0 # Counter of number of grad steps performed
        self._cac_learning=False
        self._stats_avg_rate = stats_avg_rate
        self._bellman_surrogate = bellman_surrogate
        self._avg_bellman_error = 1.  # for logging; so this works with zero warm-start
        self._avg_terminal_td_error = 1

        #############################################################################################
        # Original SAC parameters
        self._qf1 = qf1
        self._qf2 = qf2
        self.replay_buffer = replay_buffer
        self._tau = target_update_tau
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._initial_log_entropy = initial_log_entropy
        self._gradient_steps = gradient_steps_per_itr
        self._optimizer = optimizer
        self._num_evaluation_episodes = num_evaluation_episodes
        self._eval_env = eval_env

        self._min_buffer_size = min_buffer_size
        self._steps_per_epoch = steps_per_epoch
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._reward_scale = reward_scale
        self.max_episode_length = env_spec.max_episode_length
        self._max_episode_length_eval = env_spec.max_episode_length

        if max_episode_length_eval is not None:
            self._max_episode_length_eval = max_episode_length_eval
        self._use_deterministic_evaluation = use_deterministic_evaluation

        self.policy = policy
        self.env_spec = env_spec
        self.replay_buffer = replay_buffer

        self._sampler = sampler

        # use 2 target q networks
        self._target_qf1 = copy.deepcopy(self._qf1)
        self._target_qf2 = copy.deepcopy(self._qf2)
        self._policy_optimizer = self._optimizer(self.policy.parameters(),
                                                 lr=self._bc_policy_lr) #  lr for warmstart
        self._qf1_optimizer = self._optimizer(self._qf1.parameters(),
                                              lr=self._qf_lr)
        self._qf2_optimizer = self._optimizer(self._qf2.parameters(),
                                              lr=self._qf_lr)

        # automatic entropy coefficient tuning
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self._log_alpha = torch.Tensor([self._initial_log_entropy
                                            ]).requires_grad_()
            self._alpha_optimizer = optimizer([self._log_alpha],
                                              lr=self._alpha_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha]).log()
        self.episode_rewards = deque(maxlen=30)


    def optimize_policy(self,
                        samples_data,
                        warmstart=False):
        """Optimize the policy q_functions, and temperature coefficient.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """

        obs = samples_data['observation']
        next_obs = samples_data['next_observation']
        actions = samples_data['action']
        rewards = samples_data['reward'].flatten() * self._reward_scale
        terminals =  samples_data['terminal'].flatten()

        ##### Update Critic #####
        def compute_bellman_backup(q_pred_next):
            assert rewards.shape == q_pred_next.shape
            return rewards + (1.-terminals) * self._discount * q_pred_next + terminals * self._terminal_value(rewards, self._discount)

        def compute_bellman_loss(q_pred, q_pred_next, q_target):
            assert q_pred.shape == q_pred_next.shape == q_target.shape
            target_error = self._q_eval_loss(q_pred, q_target)
            q_target_pred = compute_bellman_backup(q_pred_next)
            td_error = self._q_eval_loss(q_pred, q_target_pred)
            w1, w2 = self._q_eval_mode
            bellman_loss = w1*target_error+ w2*td_error
            return bellman_loss, target_error, td_error

        ## Compute Bellman error
        with torch.no_grad():
            new_next_actions_dist = self.policy(next_obs)[0]
            _, new_next_actions = new_next_actions_dist.rsample_with_pre_tanh_value()
            target_q_values = self._target_qf1(next_obs, new_next_actions)
            if self._use_two_qfs:
                target_q_values = torch.min(target_q_values, self._target_qf2(next_obs, new_next_actions))
            target_q_values = torch.clip(target_q_values, min=self._Vmin, max=self._Vmax)  # projection
            q_target = compute_bellman_backup(target_q_values.flatten())

        qf1_pred = self._qf1(obs, actions).flatten()
        qf1_pred_next = self._qf1(next_obs, new_next_actions).flatten()
        qf1_bellman_losses, qf1_target_errors, qf1_td_errors = compute_bellman_loss(qf1_pred, qf1_pred_next, q_target)
        qf1_bellman_loss = qf1_bellman_losses.mean()

        qf2_bellman_loss = qf2_target_error = qf2_td_error = torch.Tensor([0.])
        if self._use_two_qfs:
            qf2_pred = self._qf2(obs, actions).flatten()
            qf2_pred_next = self._qf2(next_obs, new_next_actions).flatten()
            qf2_bellman_losses, qf2_target_errors, qf2_td_errors = compute_bellman_loss(qf2_pred, qf2_pred_next, q_target)
            qf2_bellman_loss = qf2_bellman_losses.mean()

        # Compute GAN error
        # These samples will be used for the actor update too, so they need to be traced.
        new_actions_dist = self.policy(obs)[0]
        new_actions_pre_tanh, new_actions = new_actions_dist.rsample_with_pre_tanh_value()

        gan_qf1_loss = gan_qf2_loss = 0
        if not warmstart:  # Compute gan_qf1_loss, gan_qf2_loss
            if self._init_observations is None:
                # Compute value difference
                qf1_new_actions = self._qf1(obs, new_actions.detach())
                gan_qf1_loss = (qf1_new_actions*(1+self._lambd) - qf1_pred).mean()
                if self._use_two_qfs:
                    qf2_new_actions = self._qf2(obs, new_actions.detach())
                    gan_qf2_loss = (qf2_new_actions*(1+self._lambd) - qf2_pred).mean()
            else: # initial state pessimism
                idx_ = np.random.choice(len(self._init_observations), self._buffer_batch_size)
                init_observations = self._init_observations[idx_]
                init_actions_dist = self.policy(init_observations)[0]
                init_actions_pre_tanh, init_actions = init_actions_dist.rsample_with_pre_tanh_value()
                qf1_new_actions = self._qf1(init_observations, init_actions.detach())
                gan_qf1_loss = qf1_new_actions.mean()
                if self._use_two_qfs:
                    qf2_new_actions = self._qf2(init_observations, init_actions.detach())
                    gan_qf2_loss = qf2_new_actions.mean()


        ## Compute full q loss
        # We normalized the objective to prevent exploding gradients
        # qf1_loss = gan_qf1_loss + beta * qf1_bellman_loss
        # qf2_loss = gan_qf2_loss + beta * qf2_bellman_loss
        with torch.no_grad():
            beta = self.beta
        qf1_loss = normalized_sum(gan_qf1_loss, qf1_bellman_loss, beta)
        qf2_loss = normalized_sum(gan_qf2_loss, qf2_bellman_loss, beta)

        if beta>0 or not warmstart:
            self._qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self._qf1_optimizer.step()
            self._qf1.apply(l2_projection(self._norm_constraint))

            if self._use_two_qfs:
                self._qf2_optimizer.zero_grad()
                qf2_loss.backward()
                self._qf2_optimizer.step()
                self._qf2.apply(l2_projection(self._norm_constraint))

        ##### Update Actor #####

        # Compuate entropy
        log_pi_new_actions = new_actions_dist.log_prob(value=new_actions, pre_tanh_value=new_actions_pre_tanh)
        policy_entropy = -log_pi_new_actions.mean()

        alpha_loss = 0
        if self._use_automatic_entropy_tuning:  # it comes first; seems to work also when put after policy update
            alpha_loss = self._log_alpha * (policy_entropy.detach() - self._target_entropy)  # entropy - target
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        with torch.no_grad():
            alpha = self._log_alpha.exp()

        lower_bound = 0
        if warmstart: # BC warmstart
            policy_log_prob = new_actions_dist.log_prob(samples_data['action'])
            # policy_loss = - policy_log_prob.mean() - alpha * policy_entropy
            policy_loss = normalized_sum(-policy_log_prob.mean(), -policy_entropy, alpha)
        else:
            # Compute performance difference lower bound
            min_q_new_actions = self._qf1(obs, new_actions)
            lower_bound = min_q_new_actions.mean()
            # policy_loss = - lower_bound - alpha * policy_kl
            policy_loss = normalized_sum(-lower_bound, -policy_entropy, alpha)

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        log_info = dict(
                    policy_loss=policy_loss,
                    qf1_loss=qf1_loss,
                    qf2_loss=qf2_loss,
                    qf1_bellman_loss=qf1_bellman_loss,
                    gan_qf1_loss=gan_qf1_loss,
                    qf2_bellman_loss=qf2_bellman_loss,
                    gan_qf2_loss=gan_qf2_loss,
                    beta=beta,
                    alpha_loss=alpha_loss,
                    policy_entropy=policy_entropy,
                    alpha=alpha,
                    lower_bound=lower_bound,
                    )

        # For logging
        if self._debug:
            with torch.no_grad():
                if self._bellman_surrogate=='td':
                    qf1_bellman_surrogate = qf1_td_errors.mean()
                    qf2_bellman_surrogate = qf2_td_errors.mean()
                elif self._bellman_surrogate=='target':
                    qf1_bellman_surrogate = qf1_target_errors.mean()
                    qf2_bellman_surrogate = qf2_target_errors.mean()
                elif self._bellman_surrogate is None:
                    qf1_bellman_surrogate = qf1_bellman_loss
                    qf2_bellman_surrogate = qf2_bellman_loss

                bellman_surrogate = torch.max(qf1_bellman_surrogate, qf2_bellman_surrogate)  # measure the TD error
                self._avg_bellman_error = self._avg_bellman_error*self._stats_avg_rate + bellman_surrogate*(1-self._stats_avg_rate)

                if terminals.sum()>0:
                    terminal_td_error = (qf1_td_errors * terminals).sum() / terminals.sum()
                    self._avg_terminal_td_error = self._avg_terminal_td_error*self._stats_avg_rate + terminal_td_error*(1-self._stats_avg_rate)

                qf1_pred_mean = qf1_pred.mean()
                qf2_pred_mean = qf2_pred.mean() if self._use_two_qfs else 0.
                q_target_mean = q_target.mean()
                target_q_values_mean = target_q_values.mean()
                qf1_new_actions_mean = qf1_new_actions.mean() if not warmstart else 0.
                qf2_new_actions_mean = qf2_new_actions.mean() if not warmstart and self._use_two_qfs else 0.
                action_diff = torch.mean(torch.norm(samples_data['action'] - new_actions, dim=1))


            debug_log_info = dict(
                    avg_bellman_error=self._avg_bellman_error,
                    avg_terminal_td_error=self._avg_terminal_td_error,
                    qf1_pred_mean=qf1_pred_mean,
                    qf2_pred_mean=qf2_pred_mean,
                    q_target_mean=q_target_mean,
                    target_q_values_mean=target_q_values_mean,
                    qf1_new_actions_mean=qf1_new_actions_mean,
                    qf2_new_actions_mean=qf2_new_actions_mean,
                    action_diff=action_diff,
                    qf1_target_error=qf1_target_errors.mean(),
                    qf1_td_error=qf1_td_errors.mean(),
                    qf2_target_error=qf2_target_errors.mean(),
                    qf2_td_error=qf2_td_errors.mean()
            )
            log_info.update(debug_log_info)

        return log_info

    # Below is overwritten for general logging with log_info
    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        for _ in trainer.step_epochs():
            for _ in range(self._steps_per_epoch):
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                trainer.step_episode = trainer.obtain_samples(
                    trainer.step_itr, batch_size)
                path_returns = []
                for path in trainer.step_episode:
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=np.array([
                                 step_type == StepType.TERMINAL
                                 for step_type in path['step_types']
                             ]).reshape(-1, 1)))
                    path_returns.append(sum(path['rewards']))
                assert len(path_returns) == len(trainer.step_episode)
                self.episode_rewards.append(np.mean(path_returns))

                for _ in range(self._gradient_steps):
                    log_info = self.train_once()

            if self._num_evaluation_episodes>0:
                last_return = self._evaluate_policy(trainer.step_itr)
            self._log_statistics(log_info)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)
            trainer.step_itr += 1

        return np.mean(last_return) if last_return is not None else 0

    def train_once(self, itr=None, paths=None):
        """Complete 1 training iteration of ATAC.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        del itr
        del paths
        if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
            warmstart = self._n_updates_performed<self._n_warmstart_steps
            if not warmstart and not self._cac_learning:
                self._cac_learning = True
                # Reset optimizers since the objective changes
                if self._use_automatic_entropy_tuning:
                    self._log_alpha = torch.Tensor([self._initial_log_entropy]).requires_grad_().to(self._log_alpha.device)
                    self._alpha_optimizer = self._optimizer([self._log_alpha], lr=self._alpha_lr)
                self._policy_optimizer = self._optimizer(self.policy.parameters(), lr=self._policy_lr)

            samples = self.replay_buffer.sample_transitions(self._buffer_batch_size)
            samples = as_torch_dict(samples)
            log_info = self.optimize_policy(samples, warmstart=warmstart)
            self._update_targets()
            self._n_updates_performed += 1
            log_info['n_updates_performed']=self._n_updates_performed
            log_info['warmstart'] = warmstart

        return log_info


    # Update also the target policy if needed
    def _update_targets(self):
        """Update parameters in the target q-functions."""
        if self._use_two_qfs:
            target_qfs = [self._target_qf1, self._target_qf2]
            qfs = [self._qf1, self._qf2]
        else:
            target_qfs = [self._target_qf1]
            qfs = [self._qf1]

        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._tau) + param.data * self._tau)

    # Set also the target policy if needed
    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)
        if not self._use_automatic_entropy_tuning:
            self._log_alpha = torch.Tensor([self._fixed_alpha
                                            ]).log().to(device)
        else:
            self._log_alpha = torch.Tensor([self._initial_log_entropy
                                            ]).to(device).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                                    lr=self._alpha_lr)
        self.beta = torch.Tensor([self.beta]).to(device)

    # Return also the target policy if needed
    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        if self._use_two_qfs:
            networks = [
                self.policy, self._qf1, self._qf2, self._target_qf1,
                self._target_qf2
            ]
        else:
            networks = [
                self.policy, self._qf1, self._target_qf1
            ]

        return networks

    # Evaluate both the deterministic and the stochastic policies
    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        eval_episodes = obtain_evaluation_episodes(  #TODO parallel evaluation
            self.policy,
            self._eval_env,
            self._max_episode_length_eval,
            num_eps=self._num_evaluation_episodes,
            deterministic=self._use_deterministic_evaluation)
        last_return = log_performance(epoch,
                                      eval_episodes,
                                      discount=self._discount)

        return last_return

    def _log_statistics(self, log_info):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """
        for k, v in log_info.items():
            tabular.record('Algorithm/'+k, float(v))
        tabular.record('ReplayBuffer/buffer_size', self.replay_buffer.n_transitions_stored)
