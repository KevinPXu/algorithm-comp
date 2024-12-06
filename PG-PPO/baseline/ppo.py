import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import ale_py
from ale_py import ALEInterface
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from Preprocess_env import AtariPreprocessing

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    obs_dim = np.prod(sizes[0])
    sizes[0] = obs_dim
    # print(sizes)
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(env_name='BankHeist-v4', hidden_sizes=[32], lr=1e-2, 
          epochs=3000, batch_size=10000, render=False):

    # make environment, check spaces, get obs / act dims
    gym_env = gym.make(env_name, frameskip=1)
    env = AtariPreprocessing(gym_env,
                        noop_max=30,
                        frame_skip=5,
                        screen_size=84,
                        terminal_on_life_loss=False,
                        grayscale_obs=False,  # set to True
                        grayscale_newaxis=False,
                        scale_obs=True)


    # assert isinstance(env.observation_space, Box), \
    #     "This example only works for envs with continuous state spaces."
    # assert isinstance(env.action_space, Discrete), \
    #     "This example only works for envs with discrete action spaces."

    obs_dim = 84*84*3
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])


    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)


    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        # print("obs", obs.shape, "act", act.shape, "weights", weights.shape)
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs, _ = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            # if (not finished_rendering_this_epoch) and render:
            #     env.render()

            # save obs
            batch_obs.append(obs.flatten())

            # act in the environment
            act = get_action(torch.as_tensor(obs.flatten(), dtype=torch.float32))
            obs, rew, done, truncated, _ = env.step(act)
            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, _ = env.reset()
                done = False
                ep_rews = []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break
                

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        if i % 500 == 0:  
            torch.save(logits_net.state_dict(), f'trained_policy_model_{i}.pth')
        print('epoch: %3d \t loss: %.3f \t ave_awards: %.3f \t ep_len: %.3f'%(i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='BankHeist-v4')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)