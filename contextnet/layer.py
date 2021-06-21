# Copyright (c) 2021, Sangchun Ha. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import Tensor
from typing import Tuple
from contextnet.module import Swish
import torch.nn as nn


class SELayer(nn.Module):
    r"""
    Squeeze-and-excitation module.

    Args:
        dim (int): Dimension to be used for two fully connected (FC) layers

    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``

    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim: int) -> None:
        super(SELayer, self).__init__()
        assert dim % 8 == 0, 'Dimension should be divisible by 8.'

        self.dim = dim
        self.sequential = nn.Sequential(
            nn.Linear(dim, dim // 8),
            Swish(),
            nn.Linear(dim // 8, dim),
        )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for SE Layer.

        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        residual = inputs
        seq_lengths = inputs.size(2)

        inputs = inputs.sum(dim=2) / input_lengths.unsqueeze(1)
        output = self.sequential(inputs)

        output = output.sigmoid().unsqueeze(2)
        output = output.repeat(1, 1, seq_lengths)

        return output * residual


class ConvLayer(nn.Module):
    r"""
    When the stride is 1, it pads the input so the output has the shape as the input.
    And when the stride is 2, it does not pad the input.

    Args:
        in_channels (int): Input channel in convolutional layer
        out_channels (int): Output channel in convolutional layer
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        stride(int, optional): Value of stride (default : 1)
        padding (int, optional): Value of padding (default: 0)
        activation (bool, optional): Flag indication use activation function or not (default : True)
        groups(int, optional): Value of groups (default : 1)
        bias (bool, optional): Flag indication use bias or not (default : True)

    Inputs: inputs, input_lengths
        - **inputs**: Input of convolution layer `FloatTensor` of size ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``

    Returns: output, output_lengths
        - **output**: Output of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        - **output_lengths**: The length of output tensor. ``(batch)``
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 5,
            stride: int = 1,
            padding: int = 0,
            activation: bool = True,
            groups: int = 1,
            bias: bool = True,
    ):
        super(ConvLayer, self).__init__()
        assert kernel_size == 5, "The convolution layer in the ContextNet model has 5 kernels."

        if stride == 1:
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=1,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=bias,
            )
        elif stride == 2:
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=1,
                padding=padding,
                groups=groups,
                bias=bias,
            )

        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.activation = activation

        if self.activation:
            self.swish = Swish()

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for convolution layer.

        Args:
            **inputs** (torch.FloatTensor): Input of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Output of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        outputs, output_lengths = self.conv(inputs), self._get_sequence_lengths(input_lengths)
        outputs = self.batch_norm(outputs)

        if self.activation:
            outputs = self.swish(outputs)

        return outputs, output_lengths

    def _get_sequence_lengths(self, seq_lengths):
        return (
                (seq_lengths + 2 * self.conv.padding[0]
                 - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1
        )
