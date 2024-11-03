from dataclasses import dataclass

from torch import nn


@dataclass
class ConvParameters:
    in_channels: int
    out_channels: int
    kernel_size: int


class CAFBlock(nn.Module):
    def __init__(self, video_params: ConvParameters, audio_params: ConvParameters, heads: int):
        super(CAFBlock, self).__init__()

        self.heads = heads

        self.video_conv1 = nn.Conv1d(
            in_channels=video_params.in_channels,
            out_channels=video_params.out_channels * heads,
            kernel_size=video_params.kernel_size,
        )
        self.video_conv2 = nn.Conv1d(
            in_channels=video_params.in_channels,
            out_channels=video_params.out_channels,
            kernel_size=video_params.kernel_size
        )

        self.audio_conv1 = nn.Conv2d(
            in_channels=audio_params.in_channels,
            out_channels=audio_params.out_channels,
            kernel_size=audio_params.kernel_size
        )
        self.gln1 = nn.GroupNorm(1, audio_params.out_channels)
        self.audio_conv2 = nn.Conv2d(
            in_channels=audio_params.in_channels,
            out_channels=audio_params.out_channels,
            kernel_size=audio_params.kernel_size
        )
        self.gln2 = nn.GroupNorm(1, audio_params.out_channels)

        self.relu = nn.ReLU()
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, audio_features, video_features):
        v_h = self.video_conv1(video_features)
        v_h = v_h.view(v_h.size(0), self.heads, -1, v_h.size(-1))
        v_m = v_h.mean(dim=1)
        v_attn = self.soft_max(v_m)
        v_attn = nn.functional.interpolate(v_attn, size=(audio_features.size(2)), mode="nearest")

        v_key = self.video_conv2(video_features)
        v_key = nn.functional.interpolate(v_key, size=(audio_features.size(2)), mode="nearest")

        a_val = self.audio_conv1(audio_features)
        a_gate = self.audio_conv2(audio_features)
        a_gate = self.relu(a_gate)

        f_1 = v_attn.unsqueeze(-1) * a_val
        f_2 = a_gate * v_key.unsqueeze(-1)

        return f_1 + f_2
