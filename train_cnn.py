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

BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.00025
TARGET_UPDATE_FREQ = 10000
REPLAY_MEMORY_SIZE = 1000000
REPLAY_MEMORY_START_SIZE = 50000
EPSILON_START = 1.0
MIN_EPSILON = 0.1
EPSILON_DECAY_FRAMES = 2000000
# MAX_EPISODES = 500 #500
SEED = 42
MAX_FRAMES = 2_000_000  # Adjust as per your computational resources

param_str = f"BS={BATCH_SIZE} G={GAMMA} LR={LEARNING_RATE} TUF={TARGET_UPDATE_FREQ} ES={EPSILON_START} ME={MIN_EPSILON} EDF={EPSILON_DECAY_FRAMES} MF={MAX_FRAMES}  SD={SEED}"

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

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

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
    state, _ = env.reset(seed=SEED)
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

    # Train over a defined number of gameplay episodes
    while steps_done < MAX_FRAMES:   
        state, _ = env.reset(seed=SEED)
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

        episode += 1
        start_time = time.time()

    # After training episodes are complete, save the trained CNNs
    torch.save(policy_nn.state_dict(), f"policy_nn_{param_str}.pth")
    torch.save(target_nn.state_dict(), f"target_nn_{param_str}.pth")
    print("Model saved as policy_nn.pth and target_nn.pth")

    env.close()

if __name__ == "__main__":
    main()
