import torch.nn as nn
import torch
import lightning as L


class TraffDynamics(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cos = nn.CosineSimilarity(dim = 1)
        self.mse = nn.MSELoss(reduce = False)
    def training_step(self, data, batch_idx):
        # training_step defines the train loop.

        online_graph = data[0]
        target_graphs = data[1]
        projection, projection_pred, actions, actions_pred, due, due_pred = self.model(online_graph, target_graphs)
        forward_loss, backward_loss, due_loss = self.compute_loss(projection, projection_pred, actions, actions_pred, due, due_pred)

        values = {"forward_loss": forward_loss, "backward_loss": backward_loss, "due_loss":due_loss}  # add more items if needed
        self.log_dict(values)
        return forward_loss# + backward_loss+due_loss
    
    def compute_loss(self, projection, projection_pred, actions, actions_pred, due, due_pred):
        forward_loss = self.cos(projection, projection_pred).mean()
        backward_loss = self.mse(actions, actions_pred).sum(dim = 1).mean()
        due_loss = self.mse(due, due_pred).sum(dim = 1).mean()

        return forward_loss, backward_loss, due_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        return optimizer

class TraffPred(L.LightningModule):
    def __init__(self, model, lr = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.mse = nn.MSELoss(reduce = False)

    def training_step(self, data, batch_idx):
        # training_step defines the train loop.

        online_graph = data[0]
        target_graphs = data[1]
        edge_embeddings_pred = self.model(online_graph)
        flow = target_graphs[0].nodes["segment"].data["feature"].T[0]
        forward_loss = self.mse(edge_embeddings_pred, flow).sum(dim = 1).mean()



        values = {"forward_loss": forward_loss}  # add more items if needed
        self.log_dict(values)
        return forward_loss# + backward_loss+due_loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class PathPred(L.LightningModule):
    def __init__(self, model, lr = 1e-3, loss = "mse"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.cosine = nn.CosineSimilarity(dim=1)
        self.mse = nn.MSELoss(reduce = False)
        self.loss = loss

    def training_step(self, data, batch_idx):
        # training_step defines the train loop.

        online_graph = data[0]
        target_graphs = data[1]
        batch_size = int(len(online_graph.batch_num_nodes("segment"))/self.model.window_width)
        action_pred = self.model(online_graph, target_graphs[0])
        actions = online_graph.nodes['path'].data['pathNum'].reshape(batch_size, self.model.num_paths)

        if self.loss == "mse":
            backward_loss = self.mse(actions, action_pred).sum(dim = 1).mean()
        elif self.loss == "cosine":
            backward_loss = self.cosine(actions, action_pred).mean()




        values = {"backward_loss": backward_loss}  # add more items if needed
        self.log_dict(values)
        return backward_loss# + backward_loss+due_loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer