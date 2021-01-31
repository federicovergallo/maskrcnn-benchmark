import torch
import torch.nn.functional as F


class MA(torch.nn.Module):
    def __init__(self):
        super(MA, self).__init__()

    def forward(self, pwc_magnitude, mask_features, batch_size, prev_is_none):
        for i in range(len(pwc_magnitude)):
            for j in range(batch_size):
                if j == 0 and prev_is_none: continue
                magnitude = pwc_magnitude[i][j, 0, :, :].view(-1)
                scores = F.softmax(magnitude)
                attention_matrix = scores.view(pwc_magnitude[i].shape[2:])
                mask_features[i][j, :, :, :] = torch.mul(mask_features[i][j, :, :, :], attention_matrix)
        return mask_features


def build_ma():
    ma = MA()
    ma.training = False
    return ma
