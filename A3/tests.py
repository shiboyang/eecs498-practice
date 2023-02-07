# @Time    : 2023/1/31 下午4:33
# @Author  : Boyang
# @Site    : 
# @File    : tests.py
# @Software: PyCharm

from fully_connected_networks import FullyConnectedNet
import torch
import eecs598
from eecs598 import reset_seed, Solver

from convolutional_networks import DeepConvNet
from fully_connected_networks import sgd_momentum
reset_seed(0)

data_dict = eecs598.data.preprocess_cifar10(cuda=True, dtype=torch.float64, flatten=False)
# Try training a deep convolutional net with different weight initialization methods
num_train = 10000
small_data = {
  'X_train': data_dict['X_train'][:num_train],
  'y_train': data_dict['y_train'][:num_train],
  'X_val': data_dict['X_val'],
  'y_val': data_dict['y_val'],
}
input_dims = data_dict['X_train'].shape[1:]

weight_scales = ['kaiming', 1e-1, 1e-2, 1e-3]

solvers = []
for weight_scale in weight_scales:
  print('Solver with weight scale: ', weight_scale)
  model = DeepConvNet(input_dims=input_dims, num_classes=10,
                      num_filters=([8] * 10) + ([32] * 10) + ([128] * 10),
                      max_pools=[9, 19],
                      weight_scale=weight_scale,
                      reg=1e-5,
                      dtype=torch.float32,
                      device='cuda'
                      )

  solver = Solver(model, small_data,
                  num_epochs=1, batch_size=128,
                  update_rule=sgd_momentum,
                  optim_config={
                    'learning_rate': 2e-3,
                  },
                  print_every=20, device='cuda')
  solver.train()
  solvers.append(solver)