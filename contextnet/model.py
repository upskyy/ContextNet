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
from contextnet.audio_encoder import AudioEncoder
from contextnet.label_encoder import LabelEncoder
import torch
import torch.nn as nn


class ContextNet(nn.Module):
    r"""
    ContextNet has CNN-RNN-transducer architecture and features a fully convolutional encoder that incorporates
    global context information into convolution layers by adding squeeze-and-excitation modules.
    Also, ContextNet supports three size models: small, medium, and large.
    ContextNet uses the global parameter alpha to control the scaling of the model
    by changing the number of channels in the convolution filter.

    Args:
        num_vocabs (int): The number of vocabulary
        model_size (str, optional): Size of the model['small', 'medium', 'large'] (default : 'medium')
        input_dim (int, optional): Dimension of input vector (default : 80)
        encoder_num_layers (int, optional): The number of convolutional layers (default : 5)
        decoder_num_layers (int, optional): The number of rnn layers (default : 1)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        num_channels (int, optional): The number of channels in the convolution filter (default: 256)
        hidden_dim (int, optional): The number of features in the decoder hidden state (default : 2048)
        encoder_output_dim (int, optional): Dimension of encoder output vector (default: 640)
        decoder_output_dim (int, optional): Dimension of decoder output vector (default: 640)
        dropout (float, optional): Dropout probability of decoder (default: 0.3)
        rnn_type (str, optional): Type of RNN cell (default: lstm)
        sos_id (int, optional): Index of the start of sentence (default: 1)

    Inputs: inputs, input_lengths, targets, target_lengths
        - **inputs** (torch.FloatTensor): Parsed audio of batch size number `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        - **input_lengths** (torch.LongTensor): Tensor representing the sequence length of the input `LongTensor` of size
            ``(batch)``
        - **targets** (torch.LongTensor): Tensor representing the target `LongTensor` of size
            ``(batch, seq_length)``
        - **target_lengths** (torch.LongTensor): Tensor representing the target length `LongTensor` of size
            ``(batch)``

    Returns: output
        - **output** (torch.FloatTensor): Result of model predictions
    """
    supported_models = {
        'small': 0.5,
        'medium': 1,
        'large': 2,
    }

    def __init__(
            self,
            num_vocabs: int,
            model_size: str = 'medium',
            input_dim: int = 80,
            encoder_num_layers: int = 5,
            decoder_num_layers: int = 1,
            kernel_size: int = 5,
            num_channels: int = 256,
            hidden_dim: int = 2048,
            encoder_output_dim: int = 640,
            decoder_output_dim: int = 640,
            dropout: float = 0.3,
            rnn_type: str = 'lstm',
            sos_id: int = 1,
    ) -> None:
        super(ContextNet, self).__init__()
        assert model_size in ('small', 'medium', 'large'), f'{model_size} is not supported.'

        alpha = self.supported_models[model_size]

        num_channels = int(num_channels * alpha)
        encoder_output_dim = int(encoder_output_dim * alpha)

        self.encoder = AudioEncoder(
            input_dim=input_dim,
            num_layers=encoder_num_layers,
            kernel_size=kernel_size,
            num_channels=num_channels,
            output_dim=encoder_output_dim,
        )
        self.decoder = LabelEncoder(
            num_vocabs=num_vocabs,
            output_dim=decoder_output_dim,
            hidden_dim=hidden_dim,
            num_layers=decoder_num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            sos_id=sos_id,
        )
        self.joint = JointNet(num_vocabs, encoder_output_dim + decoder_output_dim)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ) -> Tensor:
        r"""
        Forward propagate a `inputs` for label encoder.

        Args:
            **inputs** (torch.FloatTensor): Parsed audio of batch size number `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            **input_lengths** (torch.LongTensor): Tensor representing the sequence length of the input
                `LongTensor` of size ``(batch)``
            **targets** (torch.LongTensor): Tensor representing the target `LongTensor` of size
                ``(batch, seq_length)``
            **target_lengths** (torch.LongTensor): Tensor representing the target length `LongTensor` of size
                ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Result of model predictions
        """
        encoder_output, encoder_output_lengths = self.encoder(inputs, input_lengths)

        self.decoder.rnn.flatten_parameters()
        decoder_output, _ = self.decoder(targets, target_lengths)

        output = self.joint(encoder_output, decoder_output)

        return output

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_lengths: int) -> Tensor:
        r"""
        Decode `encoder_output`.

        Args:
            **encoder_output** (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            **max_lengths** (int): Max decoding time step

        Returns:
            **decode_output** (torch.LongTensor): Result of model predictions
        """
        token_list = list()
        hidden_states = None

        token = torch.LongTensor([[self.decoder.sos_id]])
        if torch.cuda.is_available():
            token = token.cuda()

        for i in range(max_lengths):
            decoder_output, hidden_states = self.decoder(token, hidden_states=hidden_states)
            output = self.joint(encoder_output[i].view(-1), decoder_output.view(-1))
            prediction_token = output.topk(1)[1]
            token = prediction_token.unsqueeze(1)  # (1, 1)
            prediction_token = int(prediction_token.item())
            token_list.append(prediction_token)

        return torch.LongTensor(token_list)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        r"""
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.

        Args:
            **inputs** (torch.FloatTensor): Parsed audio of batch size number `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            **input_lengths** (torch.LongTensor): Tensor representing the sequence length of the input
                `LongTensor` of size ``(batch)``

        Returns:
            **outputs** (torch.FloatTensor): Result of model predictions.
        """
        outputs = list()

        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)

        max_lengths = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            output = self.decode(encoder_output, max_lengths)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)

        return outputs  # (B, T)


class JointNet(nn.Module):
    r"""
    Joint `encoder_output` and `decoder_output`.

    Args:
        num_vocabs (int): The number of vocabulary
        output_dim (int): Encoder output dimension plus Decoder output dimension

    Inputs: encoder_output, decoder_output
        - **encoder_output** (torch.FloatTensor): A output sequence of encoder `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        - **decoder_output** (torch.FloatTensor): A output sequence of decoder `FloatTensor` of size
            ``(batch, seq_length, dimension)``

    Returns: output
        - **output** (torch.FloatTensor): Result of joint `encoder_output` and `decoder_output`
    """
    def __init__(
            self,
            num_vocabs: int,
            output_dim: int,
    ) -> None:
        super(JointNet, self).__init__()
        self.fc = nn.Linear(output_dim, num_vocabs)

    def forward(
            self,
            encoder_output: Tensor,
            decoder_output: Tensor,
    ) -> Tensor:
        assert encoder_output.dim() == decoder_output.dim()

        if encoder_output.dim() == 3 and decoder_output.dim() == 3:  # Train
            seq_lengths = encoder_output.size(1)
            target_lengths = decoder_output.size(1)

            encoder_output = encoder_output.unsqueeze(2)
            decoder_output = decoder_output.unsqueeze(1)

            encoder_output = encoder_output.repeat(1, 1, target_lengths, 1)
            decoder_output = decoder_output.repeat(1, seq_lengths, 1, 1)

        output = torch.cat((encoder_output, decoder_output), dim=-1)
        output = self.fc(output).log_softmax(dim=-1)

        return output