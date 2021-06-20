# ContextNet
**ContextNet** has CNN-RNN-transducer architecture and features a fully convolutional encoder that incorporates global context information into convolution layers by adding squeeze-and-excitation modules.  
Also, ContextNet supports three size models: small, medium, and large.
ContextNet uses the global parameter alpha to control the scaling of the model by changing the number of channels in the convolution filter.

  
This repository contains only model code, but you can train with contextnet at [openspeech](https://github.com/sooftware/openspeech).

## Model Architecuture 
- **Configuration of the ContextNet encoder**  
  
![image](https://user-images.githubusercontent.com/54731898/122670308-4b497080-d1fc-11eb-93ae-cd2bd179440c.png)  
If you choose the model size among small, medium, and large, the number of channels in the convolution filter is set using the global parameter alpha. If the stride of a convolution block is 2, its last conv layer has a stride of two while the rest of the conv layers has a stride of one.  
- **A convolution block architecuture**  
  
![image](https://user-images.githubusercontent.com/54731898/122670336-864ba400-d1fc-11eb-985e-e40e20339a68.png)  
  
ContextNet has 23 convolution blocks C0, .... ,C22. All convolution blocks have five layers of convolution except C0 and C22 which only have one layer of convolution each. A skip connection with projection is applied on the output of the squeeze-and-excitation(SE) block.  
- **1D Squeeze-and-excitation module(SE)**    
![image](https://user-images.githubusercontent.com/54731898/122670784-abd9ad00-d1fe-11eb-8be1-c1aa8f97a7bf.png)  
  
Average pooling is applied to condense the convolution result into a 1D vector and then followed two fully connected (FC) layers with activation functions. The output goes through a Sigmoid function to be mapped to (0, 1) and then tiled and applied on the convolution output using pointwise multiplications.  


**Please check [the paper](https://arxiv.org/abs/2005.03191) for more details.**

## Usage
```python
from contextnet.model import ContextNet
import torch

BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE, NUM_VOCABS = 3, 500, 80, 10

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

model = ContextNet(num_vocabs=10)

inputs = torch.FloatTensor(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE).to(device)
input_lengths = torch.IntTensor([500, 450, 350])
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8, 7])

outputs = model(inputs, input_lengths, targets, target_lengths)

```

## Reference
- [ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context](https://arxiv.org/abs/2005.03191)  
- [ishine/ContextNet](https://github.com/ishine/ContextNet)  

## License
```
Copyright 2021 Sangchun Ha.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```  
