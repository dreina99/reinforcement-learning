import gym 
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("Mps device not found")

vec_env = make_atari_env('Breakout-v4', n_envs=4, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

print(vec_env.reset())
print(vec_env.action_space)
print(vec_env.observation_space)

log_path = os.path.join('Training', 'Logs')


# episodes = 5
# for episode in range(episodes):
#     obs = vec_env.reset()
#     done = False
#     score = 0

#     while not done:
#         vec_env.render()
#         action = vec_env.action_space.sample()
#         obs, reward, done, info = vec_env.step([action])
#         score += reward
#     print(f'Episode{episode}, Score: {score}')
# vec_env.close()
    

model = A2C('CnnPolicy', vec_env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=3_000_000)

a2c_path = os.path.join('Training', 'Saved Models', 'A2C_Breakout_model')
model.save(a2c_path)

