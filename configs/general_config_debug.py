trainer_args = {
    # 'max_epochs': 5,
    'distributed_backend': None,
    # 'distributed_backend': 'ddp',
    'precision': 32,
    'amp_level': 'O2',
    # 'max_steps': 80_000,
    # 'val_check_interval': 8_000,
    'max_steps': 80,
    'val_check_interval': 40,
}
hyperparams = {
    'train_batch_size': 4,
    'shuffle_train': True,
    'train_fraction': 1.0,
    'val_batch_size': 4,
    'shuffle_val': False,
    'val_fraction': 0.00002,
    'test_batch_size': 4,
    'num_workers': 0
    # 'num_workers': 12
}

model_params = {
    'lr': 1e-3,
    'exp_factor': 0.97,
    'decay_steps_freq': trainer_args['val_check_interval']
}
PATH_TO_DATA = "/media/vlad/hdd4_tb/datasets/lyft/"
experiment_name = f"debug"
# experiment_name = f"rs50_continue_multi_mode_400k_iters"
# experiment_name = f"effnet_b3"
AGENT_CFG_PATH = "./configs/agent_motion_config.yaml"
SEED = 42