import json
from typing import List, Dict
from IPython.display import display, clear_output
from copy import deepcopy
import PIL
import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from l5kit.geometry import transform_points
from l5kit.evaluation.csv_utils import write_pred_csv
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from l5kit.evaluation.metrics import neg_multi_log_likelihood
from l5kit.evaluation import read_gt_csv
import matplotlib.pyplot as plt


def load_resnet(resnet_class):
    is_platform = os.getenv('model_save_dir', False)

    if is_platform:
        backbone = resnet_class(pretrained=False)
        state_dict = torch.load(os.path.join(os.getenv('backbone_path'), os.getenv('backbone_name')))
        backbone.load_state_dict(state_dict)
    else:
        backbone = resnet_class(pretrained=True, progress=True)

    return backbone


def save_configs(configs: List[Dict], save_path: str):
    merged_dict = []
    for c in configs:
        merged_dict.extend(c.items())
    merged_dict = dict(merged_dict)
    save_config(merged_dict, save_path)


def save_config(config: dict, save_path: str):
    with open(save_path, 'w') as fp:
        json.dump(config, fp)


def visualize_trajectory(dataset, index, title="target_positions movement with draw_trajectory"):
    data = dataset[index]

    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
    draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])

    plt.imshow(im[::-1])
    plt.show()


def visualize_gif(dataset, scene_idx, cfg):
    scene_idx = 1
    indexes = dataset.get_scene_indices(scene_idx)
    images = []

    for idx in indexes:
        data = dataset[idx]
        im = data["image"].transpose(1, 2, 0)
        im = dataset.rasterizer.to_rgb(im)

        target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
        center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
        draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])
        clear_output(wait=True)
        display(PIL.Image.fromarray(im[::-1]))


def visualize_pred(model, eval_zarr, eval_ego_dataset, eval_dataset, agent_cfg, eval_gt_path, rasterizer, limit=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gt_rows = {}
    # build a dict to retrieve future trajectories from GT
    for row in read_gt_csv(eval_gt_path):
        gt_rows[row["track_id"] + row["timestamp"]] = row["coord"]
        gt_rows[row["track_id"] + row["timestamp"] + '_avail'] = row['avail']

    i = 0
    for frame_number in range(99, len(eval_zarr.frames), 100):  # start from last frame of scene_0 and increase by 100
        agent_indices = eval_dataset.get_frame_indices(frame_number)
        if not len(agent_indices):
            continue

        # get AV point-of-view frame
        data_ego = eval_ego_dataset[frame_number]
        im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
        center = np.asarray(agent_cfg["raster_params"]["ego_center"]) * agent_cfg["raster_params"]["raster_size"]

        predicted_positions = []
        target_positions = []

        for v_index in agent_indices:
            data_agent = eval_dataset[v_index]

            out_net = model(torch.from_numpy(data_agent["image"]).unsqueeze(0).to(device))
            out_pos = out_net[0].reshape(-1, 2).detach().cpu().numpy()
            # store absolute world coordinates
            predicted_positions.append(transform_points(out_pos, data_agent["world_from_agent"]))
            # retrieve target positions from the GT and store as absolute coordinates
            track_id, timestamp = data_agent["track_id"], data_agent["timestamp"]
            target_positions.append(gt_rows[str(track_id) + str(timestamp)] + data_agent["centroid"][:2])

        # convert coordinates to AV point-of-view so we can draw them
        predicted_positions = transform_points(np.concatenate(predicted_positions), data_ego["raster_from_world"])
        target_positions = transform_points(np.concatenate(target_positions), data_ego["raster_from_world"])

        preds_third_dim = np.expand_dims(predicted_positions, axis=0)
        score = neg_multi_log_likelihood(target_positions,
                                         preds_third_dim,
                                         np.ones((preds_third_dim.shape[0],)),
                                         np.ones((preds_third_dim.shape[1],)))

        im_ego_copy = deepcopy(im_ego)
        im_ego_merged = deepcopy(im_ego)
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 7))

        draw_trajectory(im_ego_copy, target_positions, TARGET_POINTS_COLOR)
        ax1.set_title('Prediction')

        draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)
        ax2.set_title('Ground Truth')

        draw_trajectory(im_ego_merged, target_positions, TARGET_POINTS_COLOR)
        draw_trajectory(im_ego_merged, predicted_positions, PREDICTED_POINTS_COLOR)
        ax3.set_title(f'Both trajectories. neg_multi_log_likelihood: {score:.3f}')

        ax1.imshow(im_ego[::-1])
        ax2.imshow(im_ego_copy[::-1])
        ax3.imshow(im_ego_merged[::-1])
        plt.show()

        i += 1

        if i == limit:
            break


class NegativeMultiLogLikelihood(torch.nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor) -> Tensor:
        """
        Compute a negative log-likelihood for the multi-modal scenario.
        log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
        https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        https://leimao.github.io/blog/LogSumExp/
        Args:
            gt (Tensor): array of shape (bs)x(time)x(2D coords)
            pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
            confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
            avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
        Returns:
            Tensor: negative log-likelihood for this example, a single float number
        """
        assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
        batch_size, num_modes, future_len, num_coords = pred.shape

        assert gt.shape == (
            batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
        assert confidences.shape == (
            batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
        assert torch.allclose(torch.sum(confidences, dim=1),
                              confidences.new_ones((batch_size,))), f"confidences should sum to 1"
        assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
        # assert all data are valid
        assert torch.isfinite(pred).all(), "invalid value found in pred"
        assert torch.isfinite(gt).all(), "invalid value found in gt"
        assert torch.isfinite(confidences).all(), "invalid value found in confidences"
        assert torch.isfinite(avails).all(), "invalid value found in avails"

        # convert to (batch_size, num_modes, future_len, num_coords)
        gt = torch.unsqueeze(gt, 1)  # add modes
        avails = avails[:, None, :, None]  # add modes and cords

        # error (batch_size, num_modes, future_len)
        error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

        # when confidence is 0 log goes to -inf, but we're fine with it
        # add small epsilon in order to prevent -inf in log
        error = torch.log(confidences + self.eps) - 0.5 * torch.sum(error, dim=-1)  # reduce time

        # use max aggregator on modes for numerical stability
        # error (batch_size, num_modes)
        # error are negative at this point, so max() gives the minimum one
        max_value, _ = error.max(dim=1, keepdim=True)
        error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
        return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(gt: Tensor, pred: Tensor, avails: Tensor) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return NegativeMultiLogLikelihood().forward(gt, pred.unsqueeze(1), confidences, avails)


def prediction_single_mode(model: torch.nn.Module, device: torch.device,
                           test_dataloader: DataLoader, submission_path: str):
    """
    This function infer model on test_dataloader and saves predictions in submission_path variable
    """
    with torch.no_grad():
        print('Running test')
        print(f'Test size: {len(test_dataloader)}')

        # store information for evaluation
        future_coords_offsets_pd = []
        timestamps = []
        agent_ids = []
        for idx, data in enumerate(tqdm(test_dataloader)):
            inputs = torch.tensor(data["image"], device=device)

            # convert agent coordinates into world offsets
            targets = torch.tensor(data["target_positions"], device=device)
            agents_coords = model(inputs).reshape(targets.shape).cpu().numpy().copy()
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            coords_offset = []

            for agent_coords, world_from_agent, centroid in zip(agents_coords, world_from_agents, centroids):
                coords_offset.append(transform_points(agent_coords, world_from_agent) - centroid[:2])

            future_coords_offsets_pd.append(np.stack(coords_offset))
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())

        write_pred_csv(
            submission_path,
            timestamps=np.concatenate(timestamps),
            track_ids=np.concatenate(agent_ids),
            coords=np.concatenate(future_coords_offsets_pd),
        )


def prediction_multi_mode(model: torch.nn.Module, device: torch.device,
                          test_dataloader: DataLoader, submission_path: str):
    """
    This function infer model on test_dataloader and saves predictions in submission_path variable
    """
    with torch.no_grad():
        print('Running test')
        print(f'Test size: {len(test_dataloader)}')

        # store information for evaluation
        timestamps = []
        agent_ids = []
        pred_coords_list = []
        confidences_list = []

        for idx, data in enumerate(tqdm(test_dataloader)):
            inputs = torch.tensor(data["image"], device=device)

            targets = torch.tensor(data["target_positions"], device=device)
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            preds, confidences = model(inputs)
            preds = preds.cpu().numpy().copy()

            # TODO: check if we need conversion of coords to world offsests
            for batch_idx, (agent_coords, world_from_agent, centroid) in enumerate(
                    zip(preds, world_from_agents, centroids)):
                for i in range(3):
                    preds[batch_idx, i] = transform_points(agent_coords[i], world_from_agent) - centroid[:2]

            pred_coords_list.append(preds)
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())

        write_pred_csv(
            submission_path,
            timestamps=np.concatenate(timestamps),
            track_ids=np.concatenate(agent_ids),
            coords=np.concatenate(pred_coords_list),
            confs=np.concatenate(confidences_list)
        )


def prediction_multi_mode_add_features(model: torch.nn.Module, device: torch.device,
                                       test_dataloader: DataLoader, submission_path: str):
    with torch.no_grad():
        print('Running test')
        print(f'Test size: {len(test_dataloader)}')

        # store information for evaluation
        timestamps = []
        agent_ids = []
        pred_coords_list = []
        confidences_list = []

        for idx, data in enumerate(tqdm(test_dataloader)):
            inputs = torch.tensor(data["image"], device=device)
            hist_positions = torch.tensor(data["history_positions"], device=device)
            history_yaws = torch.tensor(data["history_yaws"], device=device)

            # batch_size = inputs.size(0)
            # hist_positions = hist_positions.view(batch_size, -1)
            # history_yaws = history_yaws.view(batch_size, -1)

            targets = torch.tensor(data["target_positions"], device=device)
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            preds, confidences = model(inputs, hist_positions, history_yaws)
            preds = preds.cpu().numpy().copy()

            for batch_idx, (agent_coords, world_from_agent, centroid) in enumerate(
                    zip(preds, world_from_agents, centroids)):
                for i in range(3):
                    preds[batch_idx, i] = transform_points(agent_coords[i], world_from_agent) - centroid[:2]

            pred_coords_list.append(preds)
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())

        write_pred_csv(
            submission_path,
            timestamps=np.concatenate(timestamps),
            track_ids=np.concatenate(agent_ids),
            coords=np.concatenate(pred_coords_list),
            confs=np.concatenate(confidences_list)
        )
