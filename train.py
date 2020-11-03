from models.multimode_resnet_additional_features import LyftModelMultiModeAddFeatures
from configs import general_config_debug, general_config
from utils import save_configs, NegativeMultiLogLikelihood, prediction_multi_mode, prediction_multi_mode_add_features

import os
from datetime import datetime
from pathlib import Path
import argparse

import numpy as np
from l5kit.configs import load_config_data
import l5kit
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pprint import pprint


def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--only_submit', default=False, action='store_true')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--submission_path', default=None, type=str, help='Full path to submission.csv')
    parser.add_argument('--device_name', default="cuda:0", type=str, help='Name of device in terms of torch')

    args = parser.parse_args()

    config = general_config_debug if args.debug else general_config
    trainer_args = config.trainer_args
    hyperparams = config.hyperparams
    model_params = config.model_params

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = os.getenv('dataset_path', config.PATH_TO_DATA)
    train_location = os.getenv('dataset_path', os.getcwd())
    val_location = os.getenv('dataset_path', os.getcwd())
    test_location = os.getenv('dataset_path', config.PATH_TO_DATA)
    pl.seed_everything(config.SEED)

    print(f'Running with {args}')
    pprint(trainer_args)
    pprint(hyperparams)
    pprint(model_params)

    checkpoint_path = args.checkpoint_path
    if args.only_submit:
        assert args.submission_path is not None
        submission_path = args.submission_path
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M')
        results_folder_name = os.getenv('model_save_dir', './results/')
        results_root = Path(results_folder_name) / f"{config.experiment_name}_{timestamp}"
        checkpoints = results_root / "checkpoints"
        tensorboard_logs = results_root
        submission_path = os.path.join(results_root, "submission.csv"),
        save_config_path = os.path.join(results_root, 'config_params.json')

    # get config
    agent_cfg = load_config_data(config.AGENT_CFG_PATH)

    agent_cfg['train_data_loader']['key'] = os.path.join(train_location, agent_cfg['train_data_loader']['key'])
    agent_cfg['val_data_loader']['key'] = os.path.join(val_location, agent_cfg['val_data_loader']['key'])
    agent_cfg['test_data_loader']['key'] = os.path.join(test_location, agent_cfg['test_data_loader']['key'])

    train_cfg = agent_cfg["train_data_loader"]
    validation_cfg = agent_cfg["val_data_loader"]
    test_cfg = agent_cfg["test_data_loader"]

    print(agent_cfg)
    dm = LocalDataManager()
    train_rasterizer = l5kit.rasterization.build_rasterizer(agent_cfg, dm)
    val_rasterizer = l5kit.rasterization.build_rasterizer(agent_cfg, dm)

    criterion = NegativeMultiLogLikelihood()
    model_class = LyftModelMultiModeAddFeatures
    # model_class = LyftModelMultiModeEffNet

    if not args.only_submit:

        # Train dataset/dataloader
        train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
        train_dataset = AgentDataset(agent_cfg, train_zarr, train_rasterizer,
                                     min_frame_future=agent_cfg['model_params']['min_frame_future'],
                                     min_frame_history=agent_cfg['model_params']['min_frame_history'])

        train_dataset = torch.utils.data.Subset(train_dataset,
                                                range(0, int(len(train_dataset) * hyperparams['train_fraction'])))

        train_dataloader = DataLoader(train_dataset,
                                      shuffle=hyperparams["shuffle_train"],
                                      batch_size=hyperparams["train_batch_size"],
                                      num_workers=hyperparams["num_workers"])

        print(f'Train dataloader: {len(train_dataloader)}')
        val_zarr = ChunkedDataset(dm.require(validation_cfg["key"])).open()
        val_dataset = AgentDataset(agent_cfg, val_zarr, val_rasterizer,
                                   min_frame_future=agent_cfg['model_params']['min_frame_future'],
                                   min_frame_history=agent_cfg['model_params']['min_frame_history'])

        val_dataset = torch.utils.data.Subset(val_dataset,
                                              range(0, int(len(val_dataset) * hyperparams['val_fraction'])))

        val_dataloader = DataLoader(val_dataset,
                                    shuffle=hyperparams["shuffle_val"],
                                    batch_size=hyperparams["val_batch_size"],
                                    num_workers=hyperparams["num_workers"])
        print(f'Validation dataloader: {len(val_dataloader)}')

        model_checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(
            filepath=str(checkpoints) + '/{epoch}-{val_loss:.2f}',
            monitor='val_loss',
            verbose=True,
            save_top_k=10,
            mode='min',
            period=-1  # this is hack to save several checkpoints during one epoch after each validation
        )
        tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
            save_dir=tensorboard_logs, name='', version='tb_logs', log_graph=True
        )
        lr_logger = LearningRateLogger(logging_interval='step')

        if torch.cuda.is_available():
            trainer_args['gpus'] = 1

            if trainer_args['distributed_backend'] == 'ddp':
                trainer_args['gpus'] = -1
            else:
                trainer_args['gpus'] = 1

        print(f'Saving config to {save_config_path}')
        save_configs([agent_cfg,
                      trainer_args,
                      hyperparams,
                      model_params], save_config_path)

        trainer_args['checkpoint_callback'] = model_checkpoint
        trainer_args['logger'] = tb_logger
        trainer_args['callbacks'] = [lr_logger]
        # trainer_args['early_stop_callback'] = early_stop

        model = model_class(agent_cfg=agent_cfg, criterion=criterion, **model_params)

        if checkpoint_path:
            print('Resuming checkpoint')
            trainer_args['resume_from_checkpoint'] = checkpoint_path

        trainer = pl.Trainer(**trainer_args)
        trainer.fit(model, train_dataloader, val_dataloader)

        checkpoint_path = model_checkpoint.best_model_path

    print('Making submission')
    print(f'Loading checkpoint {checkpoint_path}')
    model = model_class.load_from_checkpoint(checkpoint_path, criterion=criterion)
    test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
    test_mask = np.load(f"{os.getenv('dataset_path', config.PATH_TO_DATA)}/scenes/mask.npz")["arr_0"]
    test_dataset = AgentDataset(agent_cfg, test_zarr, train_rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset,
                                 shuffle=False,
                                 batch_size=hyperparams["test_batch_size"],
                                 num_workers=hyperparams["num_workers"])

    device = torch.device(args.device_name)
    model.eval()
    model = model.to(device)
    # prediction_multi_mode(model, device, test_dataloader, submission_path)
    prediction_multi_mode_add_features(model, device, test_dataloader, submission_path)


if __name__ == '__main__':
    main()
