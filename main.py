# -*- coding: utf-8 -*-

import gym

def main():
  env = gym.make('Breakout-v0')
  env.reset()
  for _ in range(100000000):
    env.render()
    env.step(env.action_space.sample())
  pass


if __name__ == "__main__":

  main()