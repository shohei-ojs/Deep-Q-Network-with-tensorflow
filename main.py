# -*- coding: utf-8 -*-

from models.dqn import DQN


def main():
  dqn = DQN()
  dqn.train()


def make_input_data():
  pass


if __name__ == "__main__":
  main()