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
        action_pred, cost_pred, action_encoding = self.model(online_graph, target_graphs[0])
        cost = target_graphs[0].nodes['path'].data['cost'].reshape(batch_size, self.model.num_paths)

        if self.loss == "mse":
            backward_loss = self.mse(action_encoding, action_pred).sum(dim = 1).mean()
        elif self.loss == "cosine":
            backward_loss = self.cosine(action_encoding, action_pred).mean()

        #due_loss = self.mse(cost, cost_pred).sum(dim = 1).mean()



        values = {"backward_loss": backward_loss}#, "due_loss":due_loss}  # add more items if needed
        self.log_dict(values)
        return backward_loss# + backward_loss+due_loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class SimpleLightning(L.LightningModule):
    def __init__(self, model, lr, lam):
        super().__init__()

        self.model = model
        self.lr = lr
        self.lam = lam
        self.mse = nn.MSELoss(reduce = False)

    def _step(self, data):

        online_graph = data[0]
        target_graphs = data[1]
        batch_size = int(len(online_graph.batch_num_nodes("segment"))/self.model.window_width)


        traff_pred, routes_pred = self.model(online_graph, target_graphs[0])
        routes = online_graph.nodes['path'].data['pathNum'].reshape(batch_size, self.model.num_paths)
        traff = target_graphs[0].nodes['segment'].data["feature"][:,0].reshape(batch_size, self.model.num_edges)

        backward_loss = self.mse(routes, routes_pred).sum(dim = 1).mean()
        forward_loss = self.mse(traff, traff_pred).sum(dim = 1).mean()
        return forward_loss, backward_loss, batch_size


    def training_step(self, data, batch_idx):
        # training_step defines the train loop.


        forward_loss, backward_loss, batch_size = self._step(data)
        values = {"train_backward_loss": backward_loss, "train_traff_loss": forward_loss}#, "due_loss":due_loss}  # add more items if needed
        self.log_dict(values)
        return forward_loss+self.lam*backward_loss


    def validation_step(self, data, batch_idx, dataloader_idx):
        # training_step defines the train loop.


        forward_loss, backward_loss, batch_size = self._step(data)

        values = {f"val_backward_loss_{dataloader_idx}": backward_loss, f"val_traff_loss_{dataloader_idx}": forward_loss}#, "due_loss":due_loss}  # add more items if needed
        self.log_dict(values, batch_size = batch_size)
        return backward_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer