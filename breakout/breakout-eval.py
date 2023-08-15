import gym 
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import torch

a2c_path = os.path.join('Training', 'Saved Models', 'A2C_Breakout_model')
env = make_atari_env('Breakout-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
model = A2C.load(a2c_path, env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)