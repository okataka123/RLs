#!/usr/bin/env python

num_episodes = 500
max_steps = 200
gamma = 0.99
warmup = 10

e_start = 1.0
e_stop = 0.01
e_decay_rate = 0.001

memory_size = 10000
batch_size = 32
