#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python DQN.py --num_episodes=1000 --grid_shape=20
