import os
# build simulated envs
import gymnasium as gym
# main rl algo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import pygame
import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("Mps device not found")

env = gym.make('CartPole-v1' , render_mode="human")

# 4 items -> agent, action, environment, reward
# run through env 5 times
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    terminated = False
    score = 0

    while not terminated:
        env.render()
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        score += reward
env.close()

# action space has two values
# 0 -> push cart left
# 1 -> push cart right
print(env.action_space)
print(env.action_space.sample())

# observation sspace has 4 values 
# 0 -> cart position
# 1 -> cart velocity
# 2 -> pole angle
# 3 -> pole angular velocity
print(env.observation_space)
print(env.observation_space.sample())


