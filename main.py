# -*- coding: utf-8 -*-

import gym
import random
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from skimage.color import rgb2gray
from models.agent import Agent


tf.app.flags.DEFINE_boolean('train', False, """Straing with train mode or not.""")
tf.app.flags.DEFINE_string('env_name', 'Breakout-v0', """Define env.""")
tf.app.flags.DEFINE_integer('frame_width', 84, """Resized frame width""")
tf.app.flags.DEFINE_integer('frame_height', 84, """Resized frame height""")
tf.app.flags.DEFINE_integer('num_episodes', 12000, """Number of episodes the agent plays""")
tf.app.flags.DEFINE_integer('state_length', 4, """Number of most recent frames to produce the input to the network""")
tf.app.flags.DEFINE_float('gamma', 0.99, """Discount factor""")
tf.app.flags.DEFINE_integer('exploration_steps', 1000000, """Number of steps over which the initial value of epsilon is linearly annealed to its final value""")
tf.app.flags.DEFINE_float('initial_epsilon', 1.0, """Initial value of epsilon in epsilon-greedy""")
tf.app.flags.DEFINE_float('final_epsilon', 0.1, """Final value of epsilon in epsilon-greedy""")
tf.app.flags.DEFINE_integer('initial_replay_size', 20000, """Number of steps to populate the replay memory before training starts""")
tf.app.flags.DEFINE_integer('num_replay_memory', 400000, """Number of replay memory the agent uses for training""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Mini batch size""")
tf.app.flags.DEFINE_integer('target_update_interval', 10000, """The frequency with which the target network is updated""")
tf.app.flags.DEFINE_integer('action_interval', 4, """The agent sees only every 4th input""")
tf.app.flags.DEFINE_integer('train_interval', 4, """The agent selects 4 actions between successive updates""")
tf.app.flags.DEFINE_float('learning_rate', 0.00025, """Learning rate used by RMSProp""")
tf.app.flags.DEFINE_float('momentum', 0.95, """Momentum used by RMSProp""")
tf.app.flags.DEFINE_float('min_grad', 0.01, """Constant added to the squared gradient in the denominator of the RMSProp update""")
tf.app.flags.DEFINE_integer('save_interval', 300000, """The frequency with which the network is saved""")
tf.app.flags.DEFINE_integer('no_op_steps', 30, """Maximum number of "do nothing" actions to be performed by the agent at the start of an episode""")
tf.app.flags.DEFINE_boolean('load_network', False, """""")
tf.app.flags.DEFINE_integer('num_episodes_at_test', 30, """Number of episodes the agent plays at test time""")

FLAGS = tf.app.flags.FLAGS



def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FLAGS.frame_width, FLAGS.frame_height)) * 255)
    return np.reshape(processed_observation, (1, FLAGS.frame_width, FLAGS.frame_height))


def main():
    env = gym.make(FLAGS.env_name)
    agent = Agent(num_actions=env.action_space.n, config=FLAGS)

    if FLAGS.train:  # Train mode
        for _ in range(FLAGS.num_episodes):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, FLAGS.no_op_steps)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
                # env.render()
                processed_observation = preprocess(observation, last_observation)
                state = agent.run(state, action, reward, terminal, processed_observation)
    else:  # Test mode
        # env.monitor.start(ENV_NAME + '-test')
        for _ in range(FLAGS.num_episodes_at_test):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, FLAGS.no_op_steps)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action_at_test(state)
                observation, _, terminal, _ = env.step(action)
                env.render()
                processed_observation = preprocess(observation, last_observation)
                state = np.append(state[1:, :, :], processed_observation, axis=0)
        # env.monitor.close()


if __name__ == '__main__':
    main()