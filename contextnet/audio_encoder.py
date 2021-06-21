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

from typing import Tuple
from torch import Tensor
from contextnet.convolution import ConvBlock
import torch.nn as nn


class AudioEncoder(nn.Module):
    r"""
    Audio encoder goes through 23 convolution blocks to convert to higher feature values.

    Args:
        input_dim (int, optional): Dimension of input vector (default : 80)
        num_layers (int, optional): The number of convolution layers (default : 5)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        num_channels (int, optional): The number of channels in the convolution filter (default: 256)
        output_dim (int, optional): Dimension of encoder output vector (default: 640)

    Inputs: inputs, input_lengths
        - **inputs**: Parsed audio of batch size number `FloatTensor` of size ``(batch, seq_length, dimension)``
        - **input_lengths**: Tensor representing the sequence length of the input ``(batch)``

    Returns: output, output_lengths
        - **output**: Tensor of encoder output `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        - **output_lengths**: Tensor representing the length of the encoder output ``(batch)``
    """
    def __init__(
            self,
            input_dim: int = 80,
            num_layers: int = 5,
            kernel_size: int = 5,
            num_channels: int = 256,
            output_dim: int = 640,
    ) -> None:
        super(AudioEncoder, self).__init__()
        self.blocks = ConvBlock.make_conv_blocks(input_dim, num_layers, kernel_size, num_channels, output_dim)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for audio encoder.

        Args:
            **inputs** (torch.FloatTensor): Parsed audio of batch size number `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            **input_lengths** (torch.LongTensor): Tensor representing the sequence length of the input
                `LongTensor` of size ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Tensor of encoder output `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            **output_lengths** (torch.LongTensor): Tensor representing the length of the encoder output
                `LongTensor` of size ``(batch)``
        """
        output = inputs.transpose(1, 2)
        output_lengths = input_lengths

        for block in self.blocks:
            output, output_lengths = block(output, output_lengths)

        return output.transpose(1, 2), output_lengths
