import torch
import torch.nn as nn
import dgl

from path_prediction.model.edge import EdgeEmbedding
from path_prediction.model.path import InnerPathModel, InterPathModel, GATConv, PathLearningProjection, TemporalProjection, PathProjection
from path_prediction.model.dynamics import PathPrediction, DUEPrediction, ForwardPrediction

class DynamicsTraffModel(nn.Module):
    def __init__(self, 
                #  seq_max_len, 
                #  edge_num, 
                #  num_paths, 
                #  window_width, 
                #  horizon,
                #  edge_feats,  
                #  embedding_dim = 48,
                #  projection_dim = 8,
                #  inner_path_hidden_size = 400,
                #  inner_path_num_layers = 2,
                #  gat_conv_out_feats = 40,
                #  num_heads = 2, 
                #  route_learning_projection_dim = 500,
                #  forward_prediction_hidden_size = 1200, 
                #  project = True
                 ):
        super(DynamicsTraffModel, self).__init__()
        # self.seq_max_len = seq_max_len
        # self.edge_num = edge_num
        # self.num_paths = num_paths
        # self.window_width = window_width
        # self.horizon = horizon
        # self.embedding_dim = embedding_dim
        # self.edge_feats = edge_feats
        # self.projection_dim = projection_dim
        # self.inner_path_hidden_size = inner_path_hidden_size
        # self.inner_path_num_layers = inner_path_num_layers
        # self.gat_conv_out_feats = gat_conv_out_feats
        # self.num_heads = num_heads
        # self.route_learning_projection_dim = route_learning_projection_dim
        # self.forward_prediction_hidden_size = forward_prediction_hidden_size
        # self.project = project

        # self._init()
        # self._init_backward()
        # self._init_forward()
        
        # self.projection = EdgeEmbeddingProjection(self.edge_num, 
        #                                           self.embedding_dim, 
        #                                           self.projection_dim)
        # self.linear_projection = nn.Sequential(
        #                                        nn.Linear(self.projection_dim, self.projection_dim))


    def _init(self):


        self.embedding = EdgeEmbedding(self.edge_num+1, self.edge_feats, self.embedding_dim, self.edge_embedding_hidden_sizes, layer_norm=self.edge_embedding_layer_norm)


        self.InnerPathModel = InnerPathModel(seq_dim=self.embedding_dim,
                                             hidden_size=self.inner_path_hidden_size,
                                             num_layers=self.inner_path_num_layers,
                                             seq_max_len=self.seq_max_len,
                                             edge_num=self.edge_num)


        self.interPathModel = InterPathModel()


    def _init_backward(self):


        self.gat_conv = GATConv(in_feats=self.inner_path_hidden_size * 2,
                                out_feats=self.gat_conv_out_feats,
                                hidden_size = self.gat_conv_hidden_size,
                                num_heads=self.num_heads,
                                seq_max_len=self.seq_max_len)
        
        self.route_projection = PathLearningProjection(in_feats = self.gat_conv_out_feats,
                                                  num_heads=self.num_heads, 
                                                  seq_max_len=self.seq_max_len, 
                                                  num_paths= self.num_paths, 
                                                  projection_dim=self.route_learning_projection_dim,
                                                  hidden_sizes=self.path_learning_hidden_sizes)

        self.temporal_projection = TemporalProjection(in_size = self.route_learning_projection_dim,
                                                      num_paths = self.num_paths,
                                                      window_width = self.window_width,
                                                      hidden_size = self.gru_hidden_size,
                                                      num_layers = self.gru_num_layers)                          

        self.action_prediction = PathPrediction(
                                                  projection_dim=self.route_learning_projection_dim,#self.gru_hidden_size, 
                                                  path_embedding_dim=self.projection_dim,#self.num_paths,
                                                  hidden_sizes=self.path_prediction_hidden_sizes,
                                                  layer_norm = self.prediction_layer_norm ) 
        # IF NUM PATHS IS BIG THEN MAYBE BETTER TO DO ACTION ENCODING
        #self.action_encoder = ActionEncoder(self.num_paths, self.window_width)
        self.due_prediction = DUEPrediction(
                                            projection_dim=self.route_learning_projection_dim,#self.gru_hidden_size, 
                                            path_embedding_dim=self.projection_dim,#self.num_paths,
                                            hidden_sizes=self.due_prediction_hidden_sizes,
                                            layer_norm = self.prediction_layer_norm)
        
        self.action_encoding =  PathProjection(num_paths = self.num_paths,
                                               projection_dim = self.projection_dim )


    def _init_forward(self):


        self.forward_prediction = ForwardPrediction(
                                                    in_size = self.inner_path_hidden_size * 2, 
                                                    out_embedding_size = self.embedding_dim,
                                                    hidden_size = self.forward_prediction_hidden_size,
                                                    num_layers = self.num_forward_gru_layers, 
                                                    window_width=self.window_width, 
                                                    edge_num=self.edge_num, 
                                                    horizon = self.horizon)

    def embed_paths(self, batch_graph):
        edgeIdOfPath = batch_graph.nodes['path'].data['segmentId']
        edgeIdOfPath = edgeIdOfPath.squeeze(dim=-1).to(torch.int64)
        lengths = torch.count_nonzero(edgeIdOfPath, dim=1).float()
        pathedgeFeat = batch_graph.nodes['path'].data['pathSegmentFeat']
        pathEdgeEmbedding = self.embedding(edgeIdOfPath, pathedgeFeat)
        path_embedding = self.InnerPathModel(pathEdgeEmbedding, lengths)
        #NEED TO CONSIDER WHERE I WANT THE PREDICTION, SHOULD I PREDIT THE PATH EMBEDDINGS?
        #SHOULD I PREDICT THE ACTUAL EDGE COUNTS?
        #TRY BOTH ?
        #DOES THE PROJECTION AFTER THE PREDICTION ALLOW TO REMOVE INFROMATION FROM
        #THE EMBEDDINGS IN THE FORWARD PREDICTION SO THAT WE DONT PENALIZE IT
        #FOR LEARNING IMPORTANT INFORMATION???>


        path2edge = self.inter_path_embedding(batch_graph, path_embedding)

        return path2edge


    def embed_edges(self, batch_graph):
        edge_feats = batch_graph.nodes['segment'].data["feature"]
        edge_ids = batch_graph.nodes['segment'].data["id"]
        edge_embeddings = self.embedding(edge_ids.squeeze(), edge_feats)
        return edge_embeddings


    def inter_path_embedding(self, batch_graph, path_embedding):
        
        batch_graph.nodes['path'].data['embedding'] = path_embedding
        batch_graph.nodes['path'].data['predFlow'] = batch_graph.nodes['path'].data['pathNum']
        G_predict = dgl.metapath_reachable_graph(batch_graph, ['pass+'])
        orderInfo = batch_graph.edges['pass+'].data['orderInfo']
        edge_feats = batch_graph.nodes['segment'].data['feature'].float()


        # 2. inter-path embedding
        path2edge = self.interPathModel(graph=G_predict,
                                        feat=(path_embedding, edge_feats),
                                        orderInfo=orderInfo)


        return path2edge


    def embed_paths_from_pred(self, pred_graph, edge_embeddings_pred):


        path_edge_ids = pred_graph.nodes['path'].data['segmentId'].squeeze(dim=-1).to(torch.int64)
        lengths = torch.count_nonzero(path_edge_ids, dim=1).float()
        path_embeddingds_pred = self._get_inner_path_from_edge_embeddings( path_edge_ids, edge_embeddings_pred)

        path_embedding = self.InnerPathModel(path_embeddingds_pred, lengths)
        path2edge = self.inter_path_embedding(pred_graph, path_embedding)
        
        return path2edge



    def forward_dynamics(self, path2edge):


        # #forward dynamics
        node_embeddings_pred = self.forward_prediction(path2edge)

        return node_embeddings_pred


    def _get_inner_path_from_edge_embeddings(self, path_edge_ids, edge_embeddings):


        path_input_embeds_pred = []
        for edges in path_edge_ids:
            path_input_embeds_pred.append(torch.stack([edge_embeddings[i-1] for i in edges])) #IS THIS THE RIGHT THING
        path_input_embeds_pred = torch.stack(path_input_embeds_pred)
        return path_input_embeds_pred

    def _get_final_states_from_online(self, online_graph, batch_size):
        unbatched = dgl.unbatch(online_graph)
        last_state = []
        for i in range(batch_size):
            last_state.append(unbatched[int((i+1)*self.window_width-1)])
        return dgl.batch(last_state)


    # def backwards_dynamics(self, batch_graph, target_graph):
    #     path_embedding = self.gat_conv(batch_graph)
    #     path_embedding_target = self.gat_conv(target_graph) # DO I WANT TO COMPARE THE EMBEDDINGS INCLUING HORIZON OR JUST THE FINAL STATE IN THE HORIZON
    #     action = self.action_prediction(path_embedding, path_embedding_target)
    #     return action


    def forward(self, online_graph, target_graphs):

        batch_size = int(len(online_graph.batch_num_nodes("segment"))/self.window_width)

        
        projection_pred = []
        due = []
        due_pred = []
        actions = []
        actions_pred = []
        path2edge = self.embed_paths(online_graph)
        edge_embeddings_pred = self.forward_dynamics(path2edge)
        target_edge_embeddings = self.embed_edges(dgl.batch(target_graphs))
        if self.project:
            target_edge_embeddings = self.projection(target_edge_embeddings)

        final_state_graph = self._get_final_states_from_online(online_graph, batch_size)
        for i, target_graph in enumerate(target_graphs):
            path2edge_pred = self.embed_paths_from_pred(target_graph, edge_embeddings_pred)
            path2edge_pred = torch.cat([path2edge[self.edge_num*batch_size:], path2edge_pred])
            edge_embeddings_pred = self.forward_dynamics(path2edge)
            if self.project:
                edge_embeddings_pred = self.linear_projection(self.projection(edge_embeddings_pred))

            projection_pred.append(edge_embeddings_pred)

            if i ==0:
                path_target_gat_embedding = self.gat_conv(target_graph) #DOUBLE CHECK THIS
            path_final_state_gat_embedding = self.gat_conv(final_state_graph)

            due_pred.append(self.due_prediction(path_final_state_gat_embedding))
            due.append(final_state_graph.nodes['path'].data['cost'].reshape(batch_size, self.num_paths))
            actions.append(final_state_graph.nodes['path'].data['pathNum'].reshape(batch_size, self.num_paths))
            actions_pred.append(self.action_prediction(path_final_state_gat_embedding, path_target_gat_embedding))

            final_state_graph = target_graph
            path_target_gat_embedding = path_final_state_gat_embedding


        return target_edge_embeddings, torch.cat(projection_pred), torch.cat(actions), torch.cat(actions_pred), torch.cat(due), torch.cat(due_pred)


class PredPathModel(DynamicsTraffModel):
    def __init__(self, 
                 seq_max_len, 
                 edge_num, 
                 num_paths, 
                 window_width, 
                 horizon,
                 edge_feats,  
                 embedding_dim,
                 edge_embedding_hidden_sizes,
                 edge_embedding_layer_norm, 
                 projection_dim,
                 inner_path_hidden_size ,
                 inner_path_num_layers,
                 gat_conv_out_feats ,
                 gat_conv_hidden_size,
                 num_heads, 
                 route_learning_projection_dim ,
                 path_learning_hidden_sizes,
                 gru_hidden_size,
                 gru_num_layers,
                 path_prediction_hidden_sizes,
                 due_prediction_hidden_sizes,
                 prediction_layer_norm,  
                 forward_prediction_hidden_size, 
                 project = True
                 ):
        super(PredPathModel, self).__init__()
        self.seq_max_len = seq_max_len
        self.edge_num = edge_num
        self.num_paths = num_paths
        self.window_width = window_width
        self.horizon = horizon
        self.edge_embedding_hidden_sizes = edge_embedding_hidden_sizes
        self.embedding_dim = embedding_dim
        self.edge_embedding_layer_norm = edge_embedding_layer_norm
        self.edge_feats = edge_feats
        self.projection_dim = projection_dim
        self.inner_path_hidden_size = inner_path_hidden_size
        self.inner_path_num_layers = inner_path_num_layers
        self.gat_conv_out_feats = gat_conv_out_feats
        self.gat_conv_hidden_size = gat_conv_hidden_size
        self.num_heads = num_heads
        self.route_learning_projection_dim = route_learning_projection_dim
        self.path_learning_hidden_sizes = path_learning_hidden_sizes
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.forward_prediction_hidden_size = forward_prediction_hidden_size
        self.path_prediction_hidden_sizes = path_prediction_hidden_sizes
        self.due_prediction_hidden_sizes = due_prediction_hidden_sizes
        self.prediction_layer_norm = prediction_layer_norm
        self.project = project

        self._init()
        self._init_backward()

    def forward(self, online_graph, target_graph):

        batch_size = int(len(online_graph.batch_num_nodes("segment"))/self.window_width)

        _ = self.embed_paths(online_graph)
        _ = self.embed_paths(target_graph)
        path_online_gat_embedding = self.gat_conv(online_graph)
        path_target_gat_embedding = self.gat_conv(target_graph)
        # ADD NOISE?
        path_online_gat_projection = self.route_projection(path_online_gat_embedding)
        path_target_gat_projection = self.route_projection(path_target_gat_embedding)
        # path_target_gat_projection = torch.cat([path_online_gat_projection[batch_size*self.num_paths:,:], path_target_gat_projection])

        # path_online_projection = self.temporal_projection(path_online_gat_projection)
        # path_target_projection = self.temporal_projection(path_target_gat_projection)

        action_prediction = self.action_prediction(path_online_gat_projection, path_target_gat_projection)
        cost_prediction = self.due_prediction(path_online_gat_projection)

        actions = target_graph.nodes['path'].data['pathNum'].reshape(batch_size,  self.num_paths)
        action_encoding = self.action_encoding(actions)

        return action_prediction, cost_prediction, action_encoding
    




traff_pred_hidden = [1200, 5500, 5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 900, 600]            

class PredTraffModel(DynamicsTraffModel):
    def __init__(self, seq_max_len, 
                 edge_num, 
                 num_paths, 
                 window_width, 
                 horizon,
                 edge_feats,  
                 embedding_dim,
                 projection_dim,
                 inner_path_hidden_size,
                 inner_path_num_layers,
                 gat_conv_out_feats,
                 num_heads, 
                 route_learning_projection_dim,
                 forward_prediction_hidden_size, 
                 project
                 ):
        super(PredTraffModel, self).__init__()
        self.seq_max_len = seq_max_len
        self.edge_num = edge_num
        self.num_paths = num_paths
        self.window_width = window_width
        self.horizon = horizon
        self.embedding_dim = embedding_dim
        self.edge_feats = edge_feats
        self.projection_dim = projection_dim
        self.inner_path_hidden_size = inner_path_hidden_size
        self.inner_path_num_layers = inner_path_num_layers
        self.gat_conv_out_feats = gat_conv_out_feats
        self.num_heads = num_heads
        self.route_learning_projection_dim = route_learning_projection_dim
        self.forward_prediction_hidden_size = forward_prediction_hidden_size
        self.project = project

        self._init()
        self._init_forward()

    def forward(self, online_graph):

        batch_size = int(len(online_graph.batch_num_nodes("segment"))/self.window_width)

    
        path2edge = self.embed_paths(online_graph)
        edge_embeddings_pred = self.forward_dynamics(path2edge)

        return edge_embeddings_pred