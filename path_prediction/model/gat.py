from torch import nn
import torch
import dgl.nn.pytorch as dglnn
import dgl
from path_prediction.model.edge import EdgeEmbedding
from path_prediction.model.path import InnerPathModel

class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()

        self.num_paths = 24
        self.window_width = 1
        self.num_edges = 23
        self.num_feats = 15


        self.embedding = nn.Sequential(nn.Linear(self.num_feats*self.num_edges, 200),
                                        nn.ReLU(),
                                       nn.Linear(200, 100),
                                        nn.ReLU(),
                                       nn.Linear(100, 20))

        self.action_pred = nn.Sequential(nn.Linear(40, 40),
                                         nn.ReLU(),
                                         nn.Linear(40,self.num_paths),
                                         nn.ReLU())


    def forward(self, online_graph, target_graph):

        features = online_graph.nodes["segment"].data["feature"]
        online_embedding = self.embedding(features.view(-1, 23*15))

        features = target_graph.nodes["segment"].data["feature"]
        target_embedding = self.embedding(features.view(-1, 23*15))

        input = torch.cat((online_embedding, target_embedding), dim = 1)
        out = self.action_pred(input)
        return out


class EdgeEmbeddingModel(nn.Module):
    def __init__(self):
        super(EdgeEmbeddingModel, self).__init__()

        self.num_paths = 24
        self.window_width = 1
        self.num_edges = 23
        self.num_feats = 15
        self.window_width = 1
        self.num_heads = 2
        self.edge_embedding_dim = 25
        self.out_feats = 100

        self.one_hot_embedding = True


        if self.one_hot_embedding:
            self.embedding = EdgeEmbedding(self.num_edges+1,
                                self.num_feats,
                                self.edge_embedding_dim,
                                [50,50],
                                layer_norm = False)
        else:
            self.embedding = nn.Sequential(nn.Linear(self.num_feats, 20),
                                            nn.ReLU(),
                                          nn.Linear(20, 20),
                                            nn.ReLU(),
                                          nn.Linear(20, self.edge_embedding_dim))




        self.gatconv_1 = dglnn.GATConv(in_feats=self.edge_embedding_dim,
                                     out_feats=self.out_feats,
                                     num_heads=self.num_heads,
                                     allow_zero_in_degree=True)


        self.gatconv_2 = dglnn.GATConv(in_feats=self.num_heads*self.out_feats,
                                     out_feats=self.edge_embedding_dim,
                                     num_heads=self.num_heads,
                                     allow_zero_in_degree=True)


        in_size = self.num_edges*self.edge_embedding_dim*self.num_heads
        self.action_pred = nn.Sequential(nn.Linear(in_size*2, 500),
                                         nn.ReLU(),
                                         nn.Linear(500, 100),
                                         nn.ReLU(),
                                         nn.Linear(100, self.num_paths),)


        self.traffic_pred = nn.Sequential(nn.Linear(in_size, 300),
                                         nn.ReLU(),
                                         nn.Linear(300, 100),
                                         nn.ReLU(),
                                         nn.Linear(100, self.num_edges),)



    def embed_edges(self, batch_graph):
        edge_feats = batch_graph.nodes['segment'].data["feature"]
        if self.one_hot_embedding:
            edge_ids = batch_graph.nodes['segment'].data["id"]
            edge_embeddings = self.embedding(edge_ids.squeeze(), edge_feats)
        else:
            edge_embeddings = self.embedding(edge_feats)
        return edge_embeddings


    def forward(self, online_graph, target_graph):

        online_edge_embedding = self.embed_edges(online_graph)
        target_edge_embedding = self.embed_edges(target_graph)

        o_g_sub = dgl.metapath_reachable_graph(online_graph, ['connect-', 'connect+'])
        o_g_sub = dgl.add_self_loop(o_g_sub)
        t_g_sub = dgl.metapath_reachable_graph(target_graph, ['connect-', 'connect+'])
        t_g_sub = dgl.add_self_loop(t_g_sub)
        online_gat_emb = self.gatconv_1(o_g_sub, online_edge_embedding.view(-1, self.edge_embedding_dim))
        target_gat_emb = self.gatconv_1(t_g_sub, target_edge_embedding.view(-1, self.edge_embedding_dim))
        online_gat_emb = self.gatconv_2(o_g_sub, online_gat_emb.view(-1, self.num_heads*self.out_feats) )
        target_gat_emb = self.gatconv_2(o_g_sub, target_gat_emb.view(-1, self.num_heads*self.out_feats) )



        online_gat_emb = online_gat_emb.view(-1, self.num_edges*self.edge_embedding_dim*self.num_heads)
        target_gat_emb = target_gat_emb.view(-1, self.num_edges*self.edge_embedding_dim*self.num_heads)


        input = torch.cat((online_gat_emb, target_gat_emb), dim = 1)
        routes = self.action_pred(input)
        traff = self.traffic_pred(online_gat_emb)
        return traff, routes


class PathModel(nn.Module):
    def __init__(self):
        super(PathModel, self).__init__()

        self.num_paths = 24
        self.window_width = 1
        self.num_edges = 23
        self.num_feats = 15
        self.window_width = 1
        self.num_heads = 2
        self.edge_embedding_dim = 30
        self.out_feats = 5
        self.inner_path_hidden_size = 5
        self.inner_path_num_layers = 2
        self.seq_max_len = 7
        self.path_projection_dim = 5

        self.one_hot_embedding = True


        if self.one_hot_embedding:
            self.embedding = EdgeEmbedding(self.num_edges+1,
                                self.num_feats,
                                self.edge_embedding_dim,
                                [50,50,20],
                                layer_norm = True)
        else:
            self.embedding = nn.Sequential(nn.Linear(self.num_feats, 20),
                                            nn.ReLU(),
                                          nn.Linear(20, 20),
                                            nn.ReLU(),
                                          nn.Linear(20, self.edge_embedding_dim))



        self.inner_path_model = InnerPathModel(seq_dim=self.edge_embedding_dim,
                                             hidden_size=self.inner_path_hidden_size,
                                             num_layers=self.inner_path_num_layers,
                                             seq_max_len=self.seq_max_len,
                                             edge_num=self.num_edges)


        self.gat_conv = dglnn.GATConv(in_feats=self.inner_path_hidden_size*2,#self.seq_max_len
                                     out_feats=self.out_feats,
                                     num_heads=self.num_heads,
                                     allow_zero_in_degree=True)


        in_size = self.num_paths*self.out_feats*self.num_heads*self.seq_max_len
        self.action_pred = nn.Sequential(nn.Linear(in_size*2, 1000),
                                         nn.ReLU(),
                                         nn.Linear(1000, 500),
                                         nn.ReLU(),
                                        nn.Linear(500, 150),
                                         nn.ReLU(),
                                          nn.Linear(150, 60),
                                         nn.ReLU(),
                                         nn.Linear(60,self.num_paths),
                                         nn.ReLU())


    def embed_paths(self, batch_graph):
        edgeIdOfPath = batch_graph.nodes['path'].data['segmentId']
        edgeIdOfPath = edgeIdOfPath.squeeze(dim=-1).to(torch.int64)
        lengths = torch.count_nonzero(edgeIdOfPath, dim=1).float()
        pathedgeFeat = batch_graph.nodes['path'].data['pathSegmentFeat']
        pathEdgeEmbedding = self.embedding(edgeIdOfPath, pathedgeFeat)
        path_embedding = self.inner_path_model(pathEdgeEmbedding, lengths)
        g_sub = dgl.metapath_reachable_graph(batch_graph, ['select-', 'select+'])
        g_sub = dgl.add_self_loop(g_sub)

        path_embedding = self.gat_conv(g_sub, path_embedding)


        return path_embedding



    def forward(self, online_graph, target_graph):

        online_path_embeddings = self.embed_paths(online_graph)
        target_path_embeddings = self.embed_paths(target_graph)


        input = torch.cat((target_path_embeddings.view(-1, self.seq_max_len*self.num_paths*self.out_feats*self.num_heads),
                           online_path_embeddings.view(-1, self.seq_max_len*self.num_paths*self.out_feats*self.num_heads)), dim = 1)
        out = self.action_pred(input)
        return out


