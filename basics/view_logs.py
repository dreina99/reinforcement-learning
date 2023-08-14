import os

log_path = os.path.join('Training', 'Logs')
training_log_path = os.path.join(log_path, 'PPO_2')

os.system(f'tensorboard --logdir={training_log_path}')
