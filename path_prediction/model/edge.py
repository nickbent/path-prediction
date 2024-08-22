import torch.nn as nn
import torch

from path_prediction.model import linear_sequential

class EdgeEmbedding(nn.Module):


    def __init__(self, num_edges, num_in_feats, out_embedding_size, hidden_sizes, relu_last = False, layer_norm = True):
        super(EdgeEmbedding, self).__init__()
        self.num_edges = num_edges
        self.num_in_feats = num_in_feats
        self.out_embedding_size = out_embedding_size
        
    
        sizes = [self.num_edges+self.num_in_feats] + hidden_sizes + [out_embedding_size]
        self.embedding = linear_sequential(sizes, relu_last=relu_last, layer_norm=layer_norm, layer_norm_last=False)

    def forward(self, edges, edge_feats):

        one_hot = nn.functional.one_hot(edges, num_classes= self.num_edges)
        input_embedding = torch.cat([one_hot, edge_feats], dim = -1)
        return self.embedding(input_embedding)

class EdgeEmbeddingProjection(nn.Module):
    def __init__(self, num_edges, in_embedding_dim, out_dimension = 20):
        super(EdgeEmbeddingProjection, self).__init__()
        self.num_edges = num_edges
        self.in_embedding_dim = in_embedding_dim
        self.out_dimension = out_dimension


        self.linear = nn.Sequential(nn.Linear(in_embedding_dim, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, out_dimension))
        
    def forward(self, edge_embedding):
        projection = self.linear(edge_embedding)

        return projection