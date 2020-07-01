import gym
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule
import argparse

parser = argparse.ArgumentParser(
    description='Main function to train pong deepqn')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--gamma', type=float, help='gamma', default=0.99)
parser.add_argument('--replay_buffer_size', type=int, default=1000000,
                    help='size of replay buffer')
parser.add_argument('--learning_starts', type=int, default=50000,
                    help='episodes until learning starts')
parser.add_argument('--learning_freq', type=int, default=4,
                    help='frequency of learning')
parser.add_argument('--frame_history_len', type=int, default=4,
                    help='how many frames to use in q network')
parser.add_argument('--target_update_freq', type=int, default=10000,
                    help='freq of updating target')
parser.add_argument('--learning_rate', type=float, help='lr', default=0.00025)
parser.add_argument('--alpha', type=float, help='a', default=0.95)
parser.add_argument('--eps', type=float, help='exploration prob', default=0.01)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
GAMMA = args.gamma
REPLAY_BUFFER_SIZE = args.replay_buffer_size
LEARNING_STARTS = args.learning_starts
# LEARNING_STARTS = 50
LEARNING_FREQ = args.learning_freq
FRAME_HISTORY_LEN = args.frame_history_len
TARGET_UPDATE_FREQ = args.target_update_freq
# TARGET_UPDATE_FREQ = 100
LEARNING_RATE = args.learning_rate
ALPHA = 0.95
EPS = 0.01


def main(env, num_timesteps):
    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.SGD,
        kwargs=dict(lr=LEARNING_RATE, momentum=0.9),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGET_UPDATE_FREQ,
    )


if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    main(env, task.max_timesteps)
