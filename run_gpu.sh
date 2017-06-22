#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python DQN_approx.py --num_episodes=1000 --grid_shape=10
