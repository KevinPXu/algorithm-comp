import random
import ale_py
from ale_py import ALEInterface
import gymnasium as gym
import numpy as np
import torch
from Preprocess_env import AtariPreprocessing
from cnn import cnn as CNN
import matplotlib.pyplot as plt
import time
# from replay_buffer import ReplayBuffer
import torch.nn as nn
from collections import deque
from collections import defaultdict
import os
from matplotlib.ticker import MaxNLocator

BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.00025
TARGET_UPDATE_FREQ = 10000
REPLAY_MEMORY_SIZE = 1000000
REPLAY_MEMORY_START_SIZE = 50000
EPSILON_START = 1.0
MIN_EPSILON = 0.2
EPSILON_DECAY_FRAMES = 2000000
# MAX_EPISODES = 500 #500
SEED = 42
MAX_FRAMES = 2000000 
PLOT_INTERVAL = 10  # Plot every N episodes
SAVE_DIR = "training_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

param_str = f"BS={BATCH_SIZE} G={GAMMA} LR={LEARNING_RATE} TUF={TARGET_UPDATE_FREQ} ES={EPSILON_START} ME={MIN_EPSILON} EDF={EPSILON_DECAY_FRAMES} MF={MAX_FRAMES} SEED={SEED}"

def compute_epsilon(steps_done):
    epsilon = MIN_EPSILON + (EPSILON_START - MIN_EPSILON) * max(0, (EPSILON_DECAY_FRAMES - steps_done) / EPSILON_DECAY_FRAMES)
    return epsilon


from collections import deque
class ReplayBuffer:
    '''
    This class stores gameplay states for training a CNN
    (Consider moving to its own file for a final implementation)
    '''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

def train(policy_nn, target_nn, replay_buffer, optimizer, batch_size, gamma, device):
    '''
    Train a pair of CNNs using (1) game states from a replay_buffer and (2) an optimizer
    Models are trained to choose Q values that maximize gameplay reward
    Implemented using MSE Loss
    '''
    # ensure replay buffer has enough entries
    if len(replay_buffer) < batch_size:
        # can move this check to calling function instead
        print("ERROR ENCOUNTERED")
        return None
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # convert gameplay data to tensors
    states = torch.tensor(states, dtype=torch.float32).to(device)
    states = states.view(batch_size, -1, 84, 84)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    next_states = next_states.view(batch_size, -1, 84, 84)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Compute Q-values
    q_values = policy_nn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute target Q-values
    with torch.no_grad():
        max_next_q_values = target_nn(next_states).max(1)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Compute loss
    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

class TrainingPlotter:
    def __init__(self, param_str):
        self.param_str = param_str
        self.episode_rewards = []
        self.episode_losses = []
        self.moving_avg_rewards = []
        self.moving_avg_losses = []
        self.steps = []
        self.window_size = 100
        self.start_time = time.time()
        self.episode_times = []
        
        # Create figure with two subplots and extra space at bottom
        self.fig, self.axs = plt.subplots(2, 1, figsize=(15, 12))
        self.fig.suptitle(f'Training Metrics\n{param_str}')
        
        # Create a text box for metrics that we can update
        self.metrics_text = self.fig.text(0.02, 0.02, "", fontsize=10, va='bottom')
        
    def update(self, episode, reward, losses, total_steps, epsilon):
        # Store raw data
        self.episode_rewards.append(reward)
        if losses:
            avg_loss = np.mean(losses)
            self.episode_losses.append(avg_loss)
        self.steps.append(total_steps)
        self.episode_times.append(time.time() - self.start_time)
        
        # Calculate moving averages
        if len(self.episode_rewards) >= self.window_size:
            self.moving_avg_rewards.append(np.mean(self.episode_rewards[-self.window_size:]))
            self.moving_avg_losses.append(np.mean(self.episode_losses[-self.window_size:]))
        
        # Plot every PLOT_INTERVAL episodes
        if episode % PLOT_INTERVAL == 0:
            self.plot_metrics(episode, epsilon)
    
    def plot_metrics(self, episode, epsilon):
        # Clear the entire figure
        plt.clf()
        
        # Recreate subplots
        self.axs = self.fig.subplots(2, 1)
        self.fig.suptitle(f'Training Metrics\n{self.param_str}')
        
        # Plot 1: Rewards
        self.axs[0].plot(self.episode_rewards, 'b-', alpha=0.3, label='Raw Reward')
        if self.moving_avg_rewards:
            self.axs[0].plot(range(self.window_size-1, len(self.moving_avg_rewards) + self.window_size-1),
                            self.moving_avg_rewards, 'r-', label=f'{self.window_size}-Episode Moving Avg')
        self.axs[0].set_title('Rewards per Episode')
        self.axs[0].set_xlabel('Episode')
        self.axs[0].set_ylabel('Reward')
        self.axs[0].legend()
        self.axs[0].grid(True)
        
        # Plot 2: Losses
        if self.episode_losses:
            self.axs[1].plot(self.episode_losses, 'b-', alpha=0.3, label='Raw Loss')
            if self.moving_avg_losses:
                self.axs[1].plot(range(self.window_size-1, len(self.moving_avg_losses) + self.window_size-1),
                                self.moving_avg_losses, 'r-', label=f'{self.window_size}-Episode Moving Avg')
            self.axs[1].set_title('Loss per Episode')
            self.axs[1].set_xlabel('Episode')
            self.axs[1].set_ylabel('Loss')
            self.axs[1].legend()
            self.axs[1].grid(True)
        
        # Update metrics text
        info_text = (f'Episodes: {episode}\n'
                    f'Total Steps: {self.steps[-1]:,}\n'
                    f'Epsilon: {epsilon:.4f}\n'
                    f'Avg Reward: {np.mean(self.episode_rewards[-100:]):.4f}\n'
                    f'Avg Loss: {np.mean(self.episode_losses[-100:]):.4f}\n'
                    f'Time (hrs): {(time.time() - self.start_time) / 3600:.2f}')
        
        # Adjust subplot spacing and update text
        plt.subplots_adjust(bottom=0.15)
        # Remove old text and add new text
        if hasattr(self, 'metrics_text'):
            self.metrics_text.remove()
        self.metrics_text = self.fig.text(0.02, 0.02, info_text, fontsize=10, va='bottom')
        
        plt.savefig(os.path.join(SAVE_DIR, f'training_metrics_{self.param_str}.png'))
        
    def save_data(self):
        """Save the training data to a file"""
        np.savez(os.path.join(SAVE_DIR, f'training_data_{self.param_str}.npz'),
                 rewards=self.episode_rewards,
                 losses=self.episode_losses,
                 moving_avg_rewards=self.moving_avg_rewards,
                 moving_avg_losses=self.moving_avg_losses,
                 steps=self.steps,
                 times=self.episode_times)

def main():
    '''
    Initialize a gym environment of the "Bank Heist" Atari game
    Create policy and target CNNs to learn to play the game
    Train the networks on episodes equal in number to constant MAX_EPISODES
    '''
    import logging
    import time

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    # Create game environment
    gym_env = gym.make('BankHeist-v4', frameskip=1)
    env = AtariPreprocessing(gym_env,
                        noop_max=30,
                        frame_skip=4,
                        screen_size=84,
                        terminal_on_life_loss=False,
                        grayscale_obs=True,
                        grayscale_newaxis=False,
                        scale_obs=True)

    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    # Create CNNS   
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA available! Training on GPU.", flush=True)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS available! Training on GPU.", flush=True)
    else:
        device = torch.device('cpu')
        print("CUDA NOT available... Training on CPU.", flush=True)

    policy_nn = CNN(in_channels=4, num_actions=env.action_space.n).to(device)
    target_nn = CNN(in_channels=4, num_actions=env.action_space.n).to(device)
    target_nn.load_state_dict(policy_nn.state_dict())
    target_nn.eval()

    # Create optimizer and replay buffer
    optimizer = torch.optim.RMSprop(
        policy_nn.parameters(),
        lr=LEARNING_RATE,
        alpha=0.95,
        eps=0.01
    )
    replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

    # Pre-fill replay memory
    state, _ = env.reset()
    for _ in range(REPLAY_MEMORY_START_SIZE):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        reward = np.clip(reward, -1, 1)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]

    steps_done = 0
    episode = 0
    total_rewards = []
    losses = []
    episode_durations = []
    start_time = time.time()

    # Initialize the plotter
    plotter = TrainingPlotter(param_str)
    
    # Train over a defined number of gameplay episodes
    while steps_done < MAX_FRAMES:   
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps_this_episode = 0

        while not done:
            steps_this_episode += 1
            steps_done += 1
            epsilon = compute_epsilon(steps_done)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
                    state_tensor = state_tensor.view(1, -1, 84, 84)
                    q_values = policy_nn(state_tensor)
                    action = q_values.argmax(dim=1).item()
            
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            reward = np.clip(reward, -1, 1)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if steps_done % 4 == 0:
                loss = train(policy_nn, target_nn, replay_buffer, optimizer, BATCH_SIZE, GAMMA, device)
                if loss is not None:
                    losses.append(loss)
            

            # Update the target network less frequently than the policy network
            if steps_done % TARGET_UPDATE_FREQ == 0:
                target_nn.load_state_dict(policy_nn.state_dict())

        total_rewards.append(total_reward)
        episode_durations.append(time.time() - start_time)
        print(f"Episode {episode + 1} complete")
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            avg_loss = np.mean(losses[-(10 * (steps_this_episode // 4)):]) if losses else 0
            logger.info(f"Episode {episode + 1}")
            logger.info(f"  Average Reward (last 10 episodes): {avg_reward:.2f}")
            logger.info(f"  Average Loss (last 10 episodes): {avg_loss:.4f}")
            logger.info(f"  Epsilon: {epsilon:.4f}")
            logger.info(f"  Total Steps: {steps_done}")
            logger.info(f"  Steps This Episode: {steps_this_episode}")
            logger.info(f"  Time Elapsed: {episode_durations[-1]:.2f}s")

        # After episode completes
        plotter.update(episode, total_reward, losses, steps_done, epsilon)
        
        episode += 1
        start_time = time.time()

    # After training episodes are complete, save the trained CNNs
    torch.save(policy_nn.state_dict(), f"policy_nn_{param_str}.pth")
    torch.save(target_nn.state_dict(), f"target_nn_{param_str}.pth")
    print("Model saved as policy_nn.pth and target_nn.pth")

    # At the end of training
    plotter.save_data()

    env.close()

if __name__ == "__main__":
    main()
