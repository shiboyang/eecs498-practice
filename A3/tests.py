# @Time    : 2023/1/31 下午4:33
# @Author  : Boyang
# @Site    : 
# @File    : tests.py
# @Software: PyCharm

from fully_connected_networks import FullyConnectedNet
import torch
import eecs598
from eecs598 import reset_seed, Solver

reset_seed(0)
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = torch.randn(N, D, dtype=torch.float64, device='cuda')
y = torch.randint(C, size=(N,), dtype=torch.int64, device='cuda')

for reg in [0, 3.14]:
    print('Running check with reg = ', reg)
    model = FullyConnectedNet(
        [H1, H2],
        input_dim=D,
        num_classes=C,
        reg=reg,
        weight_scale=5e-2,
        dtype=torch.float64,
        device='cuda'
    )

    loss, grads = model.loss(X, y)
    print('Initial loss: ', loss.item())

    for name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        grad_num = eecs598.grad.compute_numeric_gradient(f, model.params[name])
        print('%s relative error: %.2e' % (name, eecs598.grad.rel_error(grad_num, grads[name])))
