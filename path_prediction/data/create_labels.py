import pandas as pd
import numpy as np
import os


EDGES_DEPARTED_FILE = "edges_departed.csv"
EDGES_COUNT_FILE = "edges_count.csv"
EDGES_LEFT_FILE = "edges_left.csv"
EDGES_ARRIVED_FILE = "edges_arrived.csv"
EDGES_COUNT_EXIT_FILE = "edges_count_exit.csv"
EDGES_OCCUPANCY_FILE = "edges_occupancy.csv"
EDGES_DENSITY_FILE = "edges_density.csv"
EDGES_TRAVEL_TIME_FILE = "edges_travel_time.csv"
EDGES_COUNT_INSIDE_FILE = "edges_count_inside.csv"
EDGES_SPEED_FILE = "edges_speed.csv"

EDGES_VARIABLES = [EDGES_DEPARTED_FILE, EDGES_COUNT_FILE, EDGES_LEFT_FILE, EDGES_ARRIVED_FILE, 
                    EDGES_COUNT_EXIT_FILE, EDGES_OCCUPANCY_FILE, EDGES_DENSITY_FILE, 
                    EDGES_TRAVEL_TIME_FILE, EDGES_COUNT_INSIDE_FILE, EDGES_SPEED_FILE]


LABEL_FILE = "labels.npz"

def get_edge_variables(iteration_dir, edge_variables = EDGES_VARIABLES):

    return np.stack([ pd.read_csv(os.path.join(iteration_dir, edges_file)).values for edges_file in edge_variables], axis = -1)

def getGraphLabels(edges_variables, window_width):

    end_timestamps = edges_variables.shape[0] - window_width
    edgeNum = edges_variables.shape[1]
    num_edge_variables = edges_variables.shape[-1]
    labels = np.zeros((edgeNum, edges_variables.shape[-1]))
    for t in range(end_timestamps):
        label = edges_variables[t + window_width]
        labels = np.vstack((labels, label))

    labels = labels.reshape(-1, num_edge_variables, edgeNum)
    label2Mins = labels[1:]
    return label2Mins


def save_graph_labels(iterations_dir, window_width):


    flow = get_edge_variables(iterations_dir)
    label2Mins = getGraphLabels(flow, window_width)
    np.savez(os.path.join(iterations_dir, LABEL_FILE), label2Mins)