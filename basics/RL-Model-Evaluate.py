import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch

env = gym.make('CartPole-v1', render_mode="human")
env = DummyVecEnv([lambda:env])
PPO_Path= os.path.join('Training', 'SavedModels', 'PPO_Model_Cartpole')
model = PPO.load(PPO_Path, env=env)

# Evaluate env
# if model scores 200 -> environment is solved
# 1 point corresponds to every step that the cart remains upright
# res = evaluate_policy(model, env, n_eval_episodes=2, render=True)
# print(res)

# test model
episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    terminated = False
    score = 0

    while not terminated:
        env.render()
        action, _ = model.predict(obs) # using our model
        obs, reward, terminated, truncated = env.step(action)
        score += reward
    print(f"Episode: {episode} Score: {score}")
#env.close()