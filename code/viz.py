import torch
from torchviz import make_dot

from conv_autoencoder import StridedConvAutoencoder


model = StridedConvAutoencoder()
x = torch.randn(1, 1, 8192, 8192)
y = model(x)

g = make_dot(y)
# g = make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
# g = make_dot(y, params=dict(model.named_parameters()))
# g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))

#  Save image
g.view() #  Generate  Digraph.gv.pdf, And automatically open
# g.render(filename='graph', view=False)  #  Save as  graph.pdf, Parameters view Indicates whether to open pdf
