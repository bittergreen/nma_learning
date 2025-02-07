import torch

from spikingjelly.activation_based import neuron, functional, layer


net_s = neuron.IFNode(step_mode='s')
T = 4
N = 1
C = 3
H = 8
W = 8

x_seq = torch.rand([T, N, C, H, W])
y_seq = functional.multi_step_forward(x_seq, net_s)

net_s.reset()
net_m = layer.MultiStepContainer(net_s)
z_seq = net_m(x_seq)
