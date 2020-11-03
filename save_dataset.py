from models.baseline import LyftModel
from configs.general_config import *

import os
from datetime import datetime
from pathlib import Path
import pickle

from tqdm import tqdm
import numpy as np
from l5kit.configs import load_config_data
import l5kit
from l5kit.evaluation.csv_utils import write_pred_csv
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

pl.seed_everything(42)

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = PATH_TO_DATA


def pickle_dataset(dataset, save_path):
    dataset = []

    for data in tqdm(dataset):
        x, y, av = data['image'], data['target_positions'], data["target_availabilities"]
        dataset.append({'image': x, 'target_positions': y, "target_availabilities": av})

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    results_root = Path("./results/") / f"{experiment_name}_{timestamp}"
    checkpoints = results_root / "checkpoints"
    tensorboard_logs = results_root

    # get config
    agent_cfg = load_config_data(AGENT_CFG_PATH)

    agent_cfg['train_data_loader']['key'] = f'{os.getcwd()}/sample.zarr'
    agent_cfg['val_data_loader']['key'] = f'{os.getcwd()}/sample.zarr'
    agent_cfg['raster_params']['raster_size'] = [128, 128]

    train_cfg = agent_cfg["train_data_loader"]
    validation_cfg = agent_cfg["val_data_loader"]
    test_cfg = agent_cfg["test_data_loader"]

    print(agent_cfg)
    dm = LocalDataManager()

    rasterizer = l5kit.rasterization.build_rasterizer(agent_cfg, dm)

    # Train dataset/dataloader
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open(cache_size_bytes=int(1e9))
    train_dataset = AgentDataset(agent_cfg, train_zarr, rasterizer)
    train_dataset = torch.utils.data.Subset(train_dataset, range(0, 1000))
    # train_dataloader = DataLoader(train_dataset,
    #                               shuffle=hyperparams["shuffle_train"],
    #                               batch_size=1,
    #                               num_workers=hyperparams["num_workers"])

    dataset = []

    for data in tqdm(train_dataset):
        x, y, av = data['image'], data['target_positions'], data["target_availabilities"]
        dataset.append({'image': x, 'target_positions': y, "target_availabilities": av})

    # torch.save(dataset, 'train_data.pt')
    # loaded_dataset = torch.load('train_data.pt')
    # print(dataset[0])
    # print(loaded_dataset[0])
    # with open('train_data.pickle', 'wb') as f:
    #     pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
