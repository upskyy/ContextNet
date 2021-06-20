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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn


class LabelEncoder(nn.Module):
    r"""
    Label encoder goes through a one-layered lstm model to convert to higher feature values.

    Args:
        num_vocabs (int): The number of vocabulary
        output_dim (int, optional): Dimension of decoder output vector (default: 640)
        hidden_dim (int, optional): The number of features in the decoder hidden state (default : 2048)
        num_layers (int, optional): The number of rnn layers (default : 1)
        dropout (float, optional): Dropout probability of decoder (default: 0.3)
        rnn_type (str, optional): Type of RNN cell (default: lstm)
        sos_id (int, optional): Index of the start of sentence (default: 1)

    Inputs: inputs, input_lengths, hidden_states
        - **inputs**: Tensor representing the target `LongTensor` of size ``(batch, seq_length)``
        - **input_lengths**: Tensor representing the target length `LongTensor` of size ``(batch)``
        - **hidden_states**: A previous hidden state of decoder `FloatTensor` of size
            ``(batch, seq_length, dimension)``

    Returns: outputs, hidden_states
        - **outputs**: A output sequence of decoder `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        - **hidden_states**: A hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
    """
    supported_rnns = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU
    }

    def __init__(
            self,
            num_vocabs: int,
            output_dim: int = 640,
            hidden_dim: int = 2048,
            num_layers: int = 1,
            dropout: float = 0.3,
            rnn_type: str = 'lstm',
            sos_id: int = 1,
    ) -> None:
        super(LabelEncoder, self).__init__()
        self.sos_id = sos_id
        self.embedding = nn.Embedding(num_vocabs, hidden_dim)
        self.rnn = self.supported_rnns[rnn_type](hidden_dim, hidden_dim, num_layers, True, True, dropout, False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor = None,
            hidden_states: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for label encoder.

        Args:
            **inputs** (torch.LongTensor): Tensor representing the target `LongTensor` of size
                ``(batch, seq_length)``
            **input_lengths** (torch.LongTensor): Tensor representing the target length `LongTensor` of size
                ``(batch)``
            **hidden_states** (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            **outputs** (torch.FloatTensor): A output sequence of decoder `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            **hidden_states** (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        embedded = self.embedding(inputs)

        if input_lengths is not None:
            embedded = pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True)
            rnn_output, hidden = self.rnn(embedded, hidden_states)
            rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)

        else:
            rnn_output, hidden_states = self.rnn(embedded, hidden_states)

        output = self.fc(rnn_output)

        return output, hidden_states