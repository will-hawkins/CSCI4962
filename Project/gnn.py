import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
import numpy as np



class GNN(torch.nn.Module):
  	
	def __init__(self,num_features,hidden,layers, out_features):
		super(GNN,self).__init__()
		seed = np.random.randint(0,high=999999,dtype=int)
		torch.manual_seed(seed)
		self.conv1 = GCNConv(num_features,hidden)
		self.conv2 = GCNConv(hidden,hidden)
		#self.conv3 = GCNConv(hidden,hidden)
		self.out_features = out_features
		self.end_layer = nn.Linear(hidden, out_features)
	def reset_parameters(self):
		self.conv1.reset_parameters()
		self.conv2.reset_parameters()
		self.end_layer.reset_parameters()
		#self.conv3.reset_parameters()

	def forward(self,torchG):

		x, edge_index = torchG.x, torchG.edge_index
		x = x.float()

		x = self.conv1(x,edge_index)
		x = F.relu(x)
		x = F.dropout(x,training=self.training)
		x = self.conv2(x,edge_index)
		x = F.relu(x)
		x = self.end_layer(x)
		#x = F.dropout(x,training=self.training)
		#x = self.conv3(x,edge_index)

		return F.log_softmax(x, dim=1)

		#return F.relu(x)
		#return F.relu(x,dim=1)
