import numpy as np
import torch


def get_pdf(x,
            mean=1.5 * np.pi,
            std=1.0):
    std = torch.tensor([std])
    exp_num = -(torch.pow(torch.tensor([x - mean]), 2))
    exp_den = 2 * (torch.pow(std, 2))
    den = torch.sqrt(torch.tensor([2 * np.pi])) * std
    return torch.exp(exp_num / exp_den) * (1 / den)


def create_worse_direction_matrix(img_shape):
    worse_direction_matrix = torch.zeros(img_shape)
    worse_direction_matrix[0:img_shape[0] // 4, 0:img_shape[1] // 4] = (7 / 4) * np.pi
    worse_direction_matrix[0:img_shape[0] // 4, img_shape[1] // 4:img_shape[1] // 2] = (3 / 2) * np.pi
    worse_direction_matrix[0:img_shape[0] // 4, img_shape[1] // 2:3 * img_shape[1] // 4] = (3 / 2) * np.pi
    worse_direction_matrix[0:img_shape[0] // 4, 3 * img_shape[1] // 4:img_shape[1]] = (5 / 4) * np.pi
    worse_direction_matrix[img_shape[0] // 4:img_shape[0] // 2, 0:img_shape[1] // 4] = (7 / 4) * np.pi
    worse_direction_matrix[img_shape[0] // 4:img_shape[0] // 2, img_shape[1] // 4:img_shape[1] // 2] = (3 / 2) * np.pi
    worse_direction_matrix[img_shape[0] // 4:img_shape[0] // 2, img_shape[1] // 2:3 * img_shape[1] // 4] = (
                                                                                                                       3 / 2) * np.pi
    worse_direction_matrix[img_shape[0] // 4:img_shape[0] // 2, 3 * img_shape[1] // 4:img_shape[1]] = (5 / 4) * np.pi
    worse_direction_matrix[img_shape[0] // 2:3 * img_shape[0] // 4, 0:img_shape[1] // 4] = 0
    worse_direction_matrix[img_shape[0] // 2:3 * img_shape[0] // 4, img_shape[1] // 4:img_shape[1] // 2] = (
                                                                                                                       3 / 2) * np.pi
    worse_direction_matrix[img_shape[0] // 2:3 * img_shape[0] // 4, img_shape[1] // 2:3 * img_shape[1] // 4] = (
                                                                                                                           3 / 2) * np.pi
    worse_direction_matrix[img_shape[0] // 2:3 * img_shape[0] // 4, 3 * img_shape[1] // 4:img_shape[1]] = np.pi
    worse_direction_matrix[3 * img_shape[0] // 4:img_shape[0], 0:img_shape[1] // 4] = 0
    worse_direction_matrix[3 * img_shape[0] // 4:img_shape[0], img_shape[1] // 4:img_shape[1] // 2] = (3 / 2) * np.pi
    worse_direction_matrix[3 * img_shape[0] // 4:img_shape[0], img_shape[1] // 2:3 * img_shape[1] // 4] = (
                                                                                                                      3 / 2) * np.pi
    worse_direction_matrix[3 * img_shape[0] // 4:img_shape[0], 3 * img_shape[1] // 4:img_shape[1]] = np.pi
    return worse_direction_matrix


class DA(torch.nn.Module):
    def __init__(self):
        super(DA, self).__init__()

    def old_forward(self, directions, mask_features, batch_size):
        for i in range(len(directions)):
            for j in range(batch_size):
                directions_flat = directions[i][j, 1, :, :].view(-1)
                directions_flat = torch.tensor([get_pdf(dir) for dir in directions_flat])
                attention_matrix = directions_flat.view(directions[i][j, 1, :, :].shape).cuda()
                mask_features[i][j, :, :, :] = torch.mul(mask_features[i][j, :, :, :], attention_matrix)
        return mask_features

    def forward(self, directions_layers, mask_features, worse_direction_matrix, batch_size, prev_is_none):
        for num, layer in enumerate(directions_layers):
            for i in range(batch_size):
                if i == 0 and prev_is_none: continue
                # Abs between actual and worse direction
                attention_matrix = torch.abs(layer[i, 1, :, :] - worse_direction_matrix[num])
                min_matrix = attention_matrix.min().expand_as(attention_matrix)
                max_matrix = attention_matrix.max().expand_as(attention_matrix)
                # Normalize
                attention_matrix = (attention_matrix - min_matrix) / (max_matrix - min_matrix)
                mask_features[num][i, :, :, :] = torch.mul(mask_features[num][i, :, :, :], attention_matrix)
        return mask_features


def build_da():
    da = DA()
    da.training = False
    return da
