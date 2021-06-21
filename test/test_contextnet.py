import unittest

from contextnet.model import ContextNet
import torch


class TestContextNet(unittest.TestCase):
    def test_forward(self):
        batch_size = 3
        seq_length = 500
        input_size = 80

        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        model = ContextNet(
            model_size='small',
            num_vocabs=10,
        ).to(device)

        inputs = torch.FloatTensor(batch_size, seq_length, input_size).to(device)
        input_lengths = torch.IntTensor([500, 450, 350]).to(device)
        targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                    [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                    [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
        target_lengths = torch.LongTensor([9, 8, 7]).to(device)

        outputs = model(inputs, input_lengths, targets, target_lengths)

        print(outputs.size())  # torch.Size([3, 59, 9, 10])

    def test_recognize(self):
        batch_size = 3
        seq_length = 500
        input_size = 80

        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        model = ContextNet(
            model_size='large',
            num_vocabs=10,
        ).to(device)

        inputs = torch.FloatTensor(batch_size, seq_length, input_size).to(device)
        input_lengths = torch.IntTensor([500, 450, 350]).to(device)

        outputs = model.recognize(inputs, input_lengths)

        print(outputs.size())  # torch.Size([3, 59])


if __name__ == '__main__':
    unittest.main()
