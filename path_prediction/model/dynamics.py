import torch.nn as nn
import torch

from path_prediction.model import linear_sequential

class PathPrediction(nn.Module):
    '''
        路径选择模块
    '''
    def __init__(self, projection_dim, path_embedding_dim, hidden_sizes, relu_last = True, layer_norm = False):
        super(PathPrediction, self).__init__()
        self.projection_dim = projection_dim
        self.path_embedding_dim = path_embedding_dim

        sizes = [projection_dim*2] + hidden_sizes + [path_embedding_dim]
        self.linear = linear_sequential(sizes, relu_last=relu_last, layer_norm=layer_norm)

    def forward(self, embed_1, embed_2):

        embed = torch.concat((embed_1, embed_2), dim=1)
        return self.linear(embed)
    



class DUEPrediction(nn.Module):

    def __init__(self, projection_dim, path_embedding_dim, hidden_sizes, relu_last = True, layer_norm = False):
        super(DUEPrediction, self).__init__()
        self.projection_dim = projection_dim
        self.path_embedding_dim = path_embedding_dim

        sizes = [projection_dim] + hidden_sizes + [path_embedding_dim]
        self.linear = linear_sequential(sizes, relu_last=relu_last, layer_norm=layer_norm)


    def forward(self, embed):

        return self.linear(embed)
    
    

    
class ForwardPrediction(nn.Module):
    def __init__(self,
                 in_size,
                 out_embedding_size, 
                 hidden_size,
                 num_layers,
                 window_width,
                 edge_num,
                 horizon,
                 hidden_sizes,
                 relu_last = True, 
                 layer_norm = False,
                 ):
        super(ForwardPrediction, self).__init__()
        self.in_size = in_size
        self.out_embedding_size = out_embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.window_width = window_width
        self.edge_num = edge_num

        self.gru = nn.GRU(input_size=in_size * self.edge_num,
                          hidden_size=hidden_sizes[0],
                          num_layers=num_layers,
                          batch_first=True)


        sizes = hidden_sizes + [self.out_embedding_size*self.edge_num]
        self.linear = linear_sequential(sizes, relu_last=relu_last, layer_norm=layer_norm)


    def forward(self, seq):
        seq = seq.reshape(-1, self.window_width, self.in_size * self.edge_num)
        all_out, _ = self.gru(seq)
        out = all_out[:, -1, :]
        pred_out = self.pred(out)
        return pred_out.reshape(-1, self.out_embedding_size)