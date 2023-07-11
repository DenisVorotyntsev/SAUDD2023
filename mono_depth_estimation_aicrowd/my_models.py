# see README for details
import sys

sys.path.append("./mono_depth_estimation_aicrowd/MIM-Depth-Estimation")
from models.model import GLPDepth, Decoder

# to include dinov2
sys.path.append("./mono_depth_estimation_aicrowd/dinov2")

from collections import OrderedDict

import torch
import torch.nn as nn
from mmcv.cnn import build_upsample_layer

from mono_depth_estimation_aicrowd.constants import MULT_X, MULT_Y, VIT_PATCH_SIZE


class MiMDecoder(Decoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, conv_feats):
        """
        Forward pass of the MiM decoder. In this implementation linear upscale layers are removed.

        Args:
            conv_feats (list[Tensor]): List of convolutional features.

        Returns:
            Tensor: The output tensor.
        """
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)
        return out


def load_mim(decoder_input_size: int):
    """
    Load the MiM model.

    Args:
        decoder_input_size (int): The input size of the decoder.

    Returns:
        Args: The arguments of the MiM model.
        GLPDepth: The loaded MiM model.
    """

    class Args:
        def __init__(self):
            self.ckpt_dir = (
                "./mono_depth_estimation_aicrowd/models/kitti_swin_base.ckpt"
            )
            self.pretrained = ""

            self.gpu_or_cpu = "cpu"

            self.max_depth = 100.0
            self.max_depth_eval = 100.0
            self.min_depth_eval = 0

            self.do_kb_crop = 1
            self.backbone = "swin_base_v2"
            self.depths = [2, 2, 18, 2]
            self.num_filters = [32, 32, 32]
            self.deconv_kernels = [2, 2, 2]
            self.window_size = [22, 22, 22, 11]
            self.pretrain_window_size = [12, 12, 12, 6]
            self.use_shift = [True, True, False, False]
            self.flip_test = True
            self.shift_window_test = True
            self.shift_size = 16
            self.drop_path_rate = 0.3
            self.use_checkpoint = False
            self.num_deconv = 3
            self.do_evaluate = True

            self.frozen_stages = -1

    mim_args = Args()
    model = GLPDepth(args=mim_args)

    decoder_output_size = decoder_input_size // 8
    print("mim in/out", decoder_input_size, decoder_output_size)

    model.decoder = MiMDecoder(
        in_channels=decoder_input_size, out_channels=decoder_output_size, args=mim_args
    )
    model_weight = torch.load(mim_args.ckpt_dir)

    if "module" in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    return mim_args, model


class DinoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # load backbone - dino model
        path_to_dino = "./mono_depth_estimation_aicrowd/models/dinov2_vitl14.pt"
        self.dino_model = torch.load(path_to_dino)
        self.patch_size = VIT_PATCH_SIZE  # =self.dino_model.patch_embed.patch_size[0]
        self.dino_embed_dim = self.dino_model.patch_embed.embed_dim

        self.multx = MULT_X
        self.multy = MULT_Y

        # max value based on targets distribution
        # this param should be finetuned
        self.max_depth = 100.0

        # load head - head of MIM model
        mim_args, model = load_mim(decoder_input_size=self.dino_embed_dim)
        self.decoder = model.decoder
        self.last_layer_depth = model.last_layer_depth

        # additional upscale layers
        self.additional_decoder_block = [
            build_upsample_layer(
                dict(type="deconv"),
                in_channels=mim_args.num_filters[-1],
                out_channels=mim_args.num_filters[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mim_args.num_filters[-1]),
            nn.ReLU(inplace=True),
        ]
        self.additional_decoder_block = nn.Sequential(*self.additional_decoder_block)

    def forward(self, img):
        """
        Forward pass of the DinoModel.

        Args:
            img (Tensor): The input image tensor.

        Returns:
            Tensor: The output depth tensor.
        """
        # backbone
        x = self.dino_model(img, is_training=True)[
            "x_norm_patchtokens"
        ]  # set to True to get full dict
        x = x.reshape(
            -1, self.multx, self.multy, self.dino_embed_dim
        )  # output: B H W C
        x = torch.permute(x, (0, 3, 1, 2))  # B H W C -> B,C,H,W

        # head
        x = self.decoder.deconv_layers(x)
        x = self.additional_decoder_block(x)
        x = self.decoder.conv_layers(x)

        # final output
        out_depth = self.last_layer_depth(x)
        out_depth = torch.sigmoid(out_depth) * self.max_depth
        return out_depth
