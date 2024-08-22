import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import os
from pathlib import Path
from abc import ABC, abstractmethod



def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

class File(ABC):


    @classmethod
    @abstractmethod
    def open(self):
        pass

    @classmethod
    @abstractmethod
    def save(self):
        pass

class Labels(File):

    @classmethod
    def open(self, path):
        return np.load(path)['arr_0']

    @classmethod
    def save(self, labels, path):
        np.savez(path, labels)

class HTG(File):

    hetero_prefix = "Hetero_"
    hetero_suffix = ".bin"

    @classmethod
    def open(self, hetero_graph_dir):


        return [ self.open_hetero_file(graph_path) for graph_path in hetero_graph_dir.iterdir()]

    @classmethod
    def open_hetero_file(self, hetero_graph_path):

        return dgl.load_graphs(str(hetero_graph_path), [0])[0][0]


    @classmethod
    def save(self, hetero_graph, path):
        dgl.save_graphs(path, [hetero_graph])

class DataDir:

    scenario_string = "dta"
    labels_string = "labels.npz"
    hetero_path = "HTGWithFeat"
    hetero_suffix = ".bin"

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)
        self.scenarios = [d for d in self.data_dir.iterdir() if self.scenario_string in d.name]
        self.scenarios.sort()
        self.iterations = [ sorted([ iteration for iteration in scenario.iterdir() if has_numbers(iteration.name) and iteration.is_dir() ]) for scenario in self.scenarios]


    def find_files_in_iterations(self, file_name):

        files = []
        for iteration_scenario in self.iterations:
            scenario_files = []
            for iteration in iteration_scenario:
                for file in iteration.iterdir():
                    if file_name in file.name:
                        scenario_files.append(file)
            if scenario_files:
                files.append(scenario_files)

        return files



def create_datasets(data_dir: DataDir, window_width, horizon, train_percentage = 0.8):
    htg_files = data_dir.find_files_in_iterations(data_dir.hetero_path)
    htg = [[ HTG.open(htg_iteration) for htg_iteration in htg_scenario] for htg_scenario in htg_files]

    train_size = int(len(htg)*train_percentage)
    
    train_htg = [h for h in htg[:train_size]]
    test_htg = [h for h in htg[train_size:]]

    return DatasetHTG(train_htg, window_width, horizon), DatasetHTG(test_htg, window_width, horizon)



class DatasetHTG(Dataset):
    def __init__(self,
                 htg,
                 window_width,
                 horizon
                 ):
        
        self.htg_x, self.htg_y = self._create_x_y_htg(htg, window_width, horizon)
        self.window_width = window_width
        self.horizon = horizon


    def _create_x_y_htg(self, htg, window_width, horizon):

        htg_x = []
        htg_y = []
        for htg_scenario in htg:
            for htg_iteration in htg_scenario:
                for t in range(len(htg_iteration)-window_width-horizon):
                    x = []
                    y = []
                    for tt in range(t, t+window_width):
                        x.append(htg_iteration[tt])
                    for tt in range(t+window_width, t+window_width+horizon):
                        y.append(htg_iteration[tt])
                    htg_x.append(dgl.batch(x))
                    htg_y.append(dgl.batch(y))

        
        return htg_x, htg_y

    def __getitem__(self, index):
        return [self.htg_x[index], self.htg_y[index]]

    def __len__(self):
        return len(self.htg_x)

def traffNet_collect_fn(samples):
    htg_x, htg_y = map(list, zip(*samples))
    batch_x = []
    batch_y = [[] for i in range(len(dgl.unbatch(htg_y[0])))]
    for h_x, h_y in zip(htg_x, htg_y):
        batch_x.append(h_x)
        for i, y in enumerate(dgl.unbatch(h_y)):
            batch_y[i].append(y)
    return dgl.batch(batch_x), [dgl.batch(y) for y in batch_y]

