import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import dgl.nn.pytorch as dglnn
import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair
import dgl.function as fn

from marl_dta.model import linear_sequential

class InnerPathModel(nn.Module):
    def __init__(self,
                 seq_dim,
                 hidden_size,
                 num_layers,
                 seq_max_len,
                 edge_num):
        super(InnerPathModel, self).__init__()
        self.in_size = seq_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_max_len = seq_max_len
        self.edge_num = edge_num

        self.gru = nn.GRU(input_size=seq_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bias=True,
                          bidirectional=True)

    def forward(self, seq, lengths):
        package = pack_padded_sequence(seq,lengths,batch_first=True,enforce_sorted=False)
        results, _ = self.gru(package)
        outputs, lens = pad_packed_sequence(results,
                                            batch_first=True,total_length=seq.shape[1])
        return outputs

class MessageFunc(nn.Module):
    '''
        次序聚合中的消息传递函数的构造
    '''
    def __init__(self, orderInfo):
        super(MessageFunc, self).__init__()
        self.orderInfo = orderInfo

    def getMessageFun(self, feat_src, orderInfo):
        unbind_feat_src = torch.unbind(feat_src)
        unbind_orderInfo = torch.unbind(orderInfo)
        messageList = list(
            map(lambda x: torch.index_select(input=x[0], dim=0, index=x[1]),
                tuple(zip(unbind_feat_src, unbind_orderInfo))))
        mailboxInfo = torch.stack(messageList).view(-1, feat_src.shape[2])
        return mailboxInfo

    def forward(self, edges):
        feat_src = edges.src['embedding']  # 根据有链接的边，获得path表示
        mask_node_feat = self.getMessageFun(feat_src=feat_src,
                                            orderInfo=self.orderInfo)
        return {'m': mask_node_feat * edges.data['_edge_weight']}


class InterPathModel(nn.Module):
    '''
        根据次序聚合节点信息
    '''
    def __init__(self):
        super(InterPathModel, self).__init__()
        self._allow_zero_in_degree = True

    def forward(self, graph, feat, orderInfo):
        with graph.local_scope():
            graph.apply_edges(fn.copy_u(u='predFlow',out='_edge_weight'))
            aggregate_fn = MessageFunc(orderInfo=orderInfo)
            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            return rst.squeeze(dim=1)

class PathEncoder(nn.Module):
    def __init__(self, num_paths, window_width, hidden_sizes, out_dimension = 32, relu_last = False, layer_norm = False):
        super(PathEncoder, self).__init__()
        self.window_width = window_width
        self.num_paths = num_paths
        self.out_dimension = out_dimension

        sizes = [num_paths] + hidden_sizes + [out_dimension]
        self.linear = linear_sequential(sizes, relu_last=relu_last, layer_norm=layer_norm)
        
    def forward(self, actions):
        actions = actions.reshape(-1, self.num_paths)
        embedding = self.linear(actions)

        return embedding.reshape(-1, self.window_width*self.out_dimension)


class GATConv(nn.Module):

    def __init__(self, in_feats, out_feats, num_heads, seq_max_len):
        super(GATConv, self).__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats


        self.gatconv = dglnn.GATConv(in_feats=in_feats,
                                     out_feats=out_feats,
                                     num_heads=num_heads,
                                     allow_zero_in_degree=True)



    def forward(self, graph):
        g_sub = dgl.metapath_reachable_graph(graph, ['select-', 'select+'])
        routeLearningEmbed = g_sub.ndata['embedding'].to(torch.float32)

        # 1.获取路径选择表示
        g_sub = dgl.add_self_loop(g_sub)
        gatEmb = self.gatconv(g_sub, routeLearningEmbed)
        return gatEmb


class PathLearningProjection(nn.Module):
    def __init__(self, in_feats, num_heads,seq_max_len, projection_dim, num_paths, hidden_sizes , relu_last = True, layer_norm = False):
        super(PathLearningProjection, self).__init__()
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.seq_max_len = seq_max_len
        self.num_paths = num_paths

        sizes = [num_paths*in_feats * num_heads* seq_max_len] + hidden_sizes + [projection_dim]
        self.linear = linear_sequential(sizes, relu_last=relu_last, layer_norm=layer_norm)

    def forward(self, gatEmbed):

        embed = gatEmbed.view(-1, self.num_paths*self.in_feats * self.num_heads *self.seq_max_len)
        return self.linear(embed)
    
    