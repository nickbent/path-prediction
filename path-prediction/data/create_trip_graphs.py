import os
import json

import pandas as pd
import numpy as np
import torch
import dgl


from sklearn import preprocessing
from dgl import save_graphs, load_graphs

from marl_dta.data import  json_loads_tuple_keys, load_dict
from marl_dta.data.convert_sumo_data import *

NODE_DICT_FILENAME = "edges.json"
PATH_DICT_FILENAME = "paths.json"
OD_DICT_FILENAME = "od.json"
OD_PATH_FILENAME = "od2path.csv"
PATH_SEGMENT_FILENAME = "path2segment.csv"
NEIGHBOUR_FILENAME = "neighbour.csv"
STATIC_INFO_FILE = "static_info.csv"
HETERO_PATH = "HtgNoFeat/Hetero.bin"

OD_PATH_HEADERS = ["connection_from","connection_to"]
PATH_AREA_HEADERS = OD_PATH_HEADERS + ["orderInfo"]




def getOdPathSegmentRelation(save_dir):

    segmentNodeDict = load_dict(os.path.join(save_dir, NODE_DICT_FILENAME), load_tuples = False)
    pathNodeDict = load_dict(os.path.join(save_dir, PATH_DICT_FILENAME))
    odNodeDict = load_dict(os.path.join(save_dir, OD_DICT_FILENAME))


    od2path = []
    path2area = []
    for path in list(pathNodeDict.keys()):
        od = (path[0][0],path[-1][-1]) #THIS MIGHT NEED TO BE CHANGED DEPENDING ON THE NAMES OF THE PATHS
        od2path.append((odNodeDict[od],pathNodeDict[path]))
        for order,p in enumerate(path):
            path2area.append((pathNodeDict[path],segmentNodeDict[p],order))

    od2path_array = np.array(od2path)
    path2area_array = np.array(path2area)

    df_od2path = pd.DataFrame(od2path_array)
    df_od2path.to_csv(os.path.join(save_dir, OD_PATH_FILENAME),header=OD_PATH_HEADERS,index=None)

    df_path2area = pd.DataFrame(path2area_array)
    df_path2area.to_csv(os.path.join(save_dir, PATH_SEGMENT_FILENAME),header=PATH_AREA_HEADERS,index=None)

def getHeteroRelation(relationfile):
    df = pd.read_csv(relationfile)
    connection_from = df['connection_from']
    connection_to = df['connection_to']

    from_tensor = torch.tensor(connection_from.tolist())
    to_tensor = torch.tensor(connection_to.tolist())
    return from_tensor, to_tensor


def createTrafficHeteroGraph(od_path_from, od_path_to, path_segment_from, path_segment_to, segment_segment_from, segment_segment_to):
    G = dgl.heterograph({
        ('od', 'select+', 'path'): (od_path_from, od_path_to),
        ('path', 'select-', 'od'): (od_path_to, od_path_from),
        ('path', 'pass+', 'segment'): (path_segment_from, path_segment_to),
        ('segment', 'pass-', 'path'): (path_segment_to, path_segment_from),
        ('segment', 'connect+', 'segment'): (segment_segment_from, segment_segment_to),
        ('segment', 'connect-', 'segment'): (segment_segment_to, segment_segment_from)
    })
    return G

def getHeteroTripGraph(save_dir):
    od_path_from,od_path_to = getHeteroRelation(os.path.join(save_dir, OD_PATH_FILENAME))
    path_segment_from, path_segment_to = getHeteroRelation(os.path.join(save_dir, PATH_SEGMENT_FILENAME))
    segment_segment_from,segment_segment_to = getHeteroRelation(os.path.join(save_dir, NEIGHBOUR_FILENAME))
    G = createTrafficHeteroGraph(od_path_from,od_path_to,path_segment_from,path_segment_to,segment_segment_from,segment_segment_to)
    save_graphs(os.path.join(save_dir, HETERO_PATH), [G])



def expendDyInitFeature(x, max_size):
    padding = torch.zeros(((max_size - x.shape[0]), x.shape[1]))
    x_new = torch.vstack((x, padding))
    return x_new

def expendDyInitFeaturePadInf(x, max_size):
    padding = torch.ones(((max_size - x.shape[0]), x.shape[1])) * torch.tensor(float('inf'))
    x_new = torch.vstack((x, padding))
    return x_new

def getPathFeat(numpathnode, PathNodeDict_reverse, max_size, segment_feats, segmentNodeDict):
    path_feat_Tensor = torch.zeros(1, max_size, segment_feats.shape[1])
    for i in range(numpathnode):
        path_t = PathNodeDict_reverse[i]
        first_line = torch.zeros(1, segment_feats.shape[1])
        for segment in path_t:
            first_line = torch.vstack((first_line, segment_feats[segmentNodeDict[segment]]))
        path_feature = first_line[1:]

        expendFeature = expendDyInitFeature(path_feature, max_size)
        path_feat_Tensor = torch.vstack((path_feat_Tensor, expendFeature.unsqueeze(dim=0)))
    return path_feat_Tensor[1:]


def getPathFeatPadWithInf(numpathnode, PathNodeDict_reverse, max_size, segment_feats, segmentNodeDict):
    path_feat_Tensor = torch.zeros(1, max_size, segment_feats.shape[1])
    for i in range(numpathnode):
        path_t = PathNodeDict_reverse[i]
        first_line = torch.zeros(1, segment_feats.shape[1])
        for segment in path_t:
            first_line = torch.vstack((first_line, segment_feats[segmentNodeDict[segment]]))
        path_feature = first_line[1:]

        expendFeature = expendDyInitFeaturePadInf(path_feature, max_size)
        path_feat_Tensor = torch.vstack((path_feat_Tensor, expendFeature.unsqueeze(dim=0)))
    return path_feat_Tensor[1:]


def getSeqMaxSize(pathNodeDict):
    allPath = list(pathNodeDict.keys())
    seqMaxSize = max([len(path) for path in allPath])

    return seqMaxSize

def getOd2Path_t(t, total_path_count_t, allOd_dict, allPath_dict):
    od2path = []
    path_t = list(total_path_count_t[t].keys())
    for path in path_t:
        od = (path[0][0], path[-1][-1]) # MIGHT NEED TO CHANGE THIS IF OD DICT IS DIFFERENT
        od2path.append([allOd_dict[od], allPath_dict[path]]) 
    od2path_array = np.array(od2path)
    return od2path_array



def embed_feats(iteration_dir, scenario_dir, net_dir):

    df_flow = pd.read_csv(os.path.join(iteration_dir, EDGES_COUNT_FILE))
    df_avgspeed = pd.read_csv(os.path.join(iteration_dir, EDGES_SPEED_FILE))
    df_edges_departed = pd.read_csv(os.path.join(iteration_dir, EDGES_DEPARTED_FILE))
    df_edges_start = pd.read_csv(os.path.join(iteration_dir, EDGES_COUNT_FILE))
    df_edges_left = pd.read_csv(os.path.join(iteration_dir, EDGES_LEFT_FILE))
    df_edges_arrived = pd.read_csv(os.path.join(iteration_dir, EDGES_ARRIVED_FILE))
    df_edges_count_exit = pd.read_csv(os.path.join(iteration_dir, EDGES_COUNT_EXIT_FILE))
    df_edges_occupancy = pd.read_csv(os.path.join(iteration_dir, EDGES_OCCUPANCY_FILE))
    df_edges_density = pd.read_csv(os.path.join(iteration_dir, EDGES_DENSITY_FILE))
    df_edges_travel_time = pd.read_csv(os.path.join(iteration_dir, EDGES_TRAVEL_TIME_FILE))
    df_edges_count_inside = pd.read_csv(os.path.join(iteration_dir, EDGES_COUNT_INSIDE_FILE))



    df_limitSpeed = pd.read_csv(os.path.join(scenario_dir, SPEED_LIMIT_FILE))

    if len(df_limitSpeed) < len(df_edges_travel_time):
        for i in range(len(df_limitSpeed), len(df_edges_travel_time)):
            df_limitSpeed.loc[i] = df_limitSpeed.loc[i-1]

    df_staticInfo = pd.read_csv(os.path.join(net_dir, STATIC_INFO_FILE))[['segment_numLanes', 'segment_length']]
    df_path2areaOrder = pd.read_csv(os.path.join(net_dir, PATH_SEGMENT_FILE))

    allTimestamps = len(df_flow)


    total_flow = np.array(df_flow).flatten('C').reshape(-1, 1)
    total_avgspeed = np.array(df_avgspeed).flatten('C').reshape(-1, 1)
    total_edges_departed = np.array(df_edges_departed).flatten('C').reshape(-1, 1)
    total_edges_left = np.array(df_edges_left).flatten('C').reshape(-1, 1)
    total_edges_start = np.array(df_edges_start).flatten('C').reshape(-1, 1)
    total_edges_arrived = np.array(df_edges_arrived).flatten('C').reshape(-1, 1)
    total_edges_count_exit = np.array(df_edges_count_exit).flatten('C').reshape(-1, 1)
    total_edges_occupancy = np.array(df_edges_occupancy).flatten('C').reshape(-1, 1)
    total_edges_density = np.array(df_edges_density).flatten('C').reshape(-1, 1)
    total_edges_travel_time = np.array(df_edges_travel_time).flatten('C').reshape(-1, 1)
    total_edges_count_inside = np.array(df_edges_count_inside).flatten('C').reshape(-1, 1)


    total_limitspeed = np.array(df_limitSpeed).flatten('C').reshape(-1, 1)
    total_staticInfo = np.tile(np.array(df_staticInfo), (allTimestamps, 1))

    totalsegmentFeat = np.concatenate((total_flow, total_avgspeed, total_edges_departed, 
                                    total_edges_start, total_limitspeed, total_edges_arrived, total_edges_count_exit, 
                                    total_edges_left, total_edges_occupancy, total_edges_occupancy, total_edges_density, 
                                    total_edges_travel_time, total_edges_count_inside, total_staticInfo), axis=1)
    max_min_scaler = preprocessing.MinMaxScaler()
    minmax_feat = max_min_scaler.fit_transform(totalsegmentFeat)

    with open(os.path.join(iteration_dir, PATH_COUNT_FILE), 'r') as f:
        total_path_count_t = eval(f.readlines()[0])

    segmentNodeDict = load_dict(os.path.join(net_dir, NODE_DICT_FILENAME), load_tuples = False)

    with open(os.path.join(iteration_dir, OD_COUNT_FILE), 'r') as f:
        total_od_count_t = eval(f.readlines()[0])

    with open(os.path.join(iteration_dir, PATH_COST_FILE), 'r') as f:
        path_cost_t = eval(f.readlines()[0])

    pathNodeDict = load_dict(os.path.join(net_dir, PATH_DICT_FILENAME))
    PathNodeDict_reverse = {v: k for k, v in pathNodeDict.items()}
    pathLen = {index: len(path) for path, index in pathNodeDict.items()}
    allOd_dict = load_dict(os.path.join(net_dir, OD_DICT_FILENAME))
    allOd_dict_reversed = {v: k for k, v in allOd_dict.items()}

    seqmaxsize = getSeqMaxSize(pathNodeDict)

    with open(os.path.join(scenario_dir, JAM_NODE_INFO_FILE), "r") as f:
        jam_segment_info_dict = json.load(f)


    for t in range(allTimestamps):
        g_t = dgl.load_graphs(os.path.join(net_dir, HETERO_PATH), [0])[0][0]
        segment_num = g_t.num_nodes('segment')
        # segment feature
        flow = torch.tensor(np.array(df_flow.iloc[t]).reshape(segment_num, 1).astype(float))
        limitspeed = torch.tensor(np.array(df_limitSpeed.iloc[t]).reshape(segment_num, 1).astype(float))
        avgspeed = torch.tensor(np.array(df_avgspeed.iloc[t]).reshape(segment_num, 1).astype(float))
        staticInfo = torch.tensor(np.array(df_staticInfo).reshape(segment_num, 2))
        segmentInitFeat = torch.cat([flow, avgspeed, limitspeed, staticInfo], dim=1)

        
        g_t.nodes['segment'].data['feature'] = torch.tensor(minmax_feat[t * segment_num:(t + 1) * segment_num]).to(torch.float32)
        
        g_t.nodes['segment'].data['id'] = torch.arange(1, segment_num + 1).reshape(-1, 1)

        
        segmentId = g_t.nodes['segment'].data['id']
        num_node_path = g_t.num_nodes("path")

        
        segmentIdOfPath = getPathFeat(numpathnode=num_node_path,
                                        PathNodeDict_reverse=PathNodeDict_reverse,
                                        max_size=seqmaxsize,
                                        segment_feats=segmentId,
                                        segmentNodeDict=segmentNodeDict)
        g_t.nodes['path'].data['segmentId'] = segmentIdOfPath

        
        pathInitFeats = getPathFeat(numpathnode=num_node_path,
                                    PathNodeDict_reverse=PathNodeDict_reverse,
                                    max_size=seqmaxsize,
                                    segment_feats=segmentInitFeat,
                                    segmentNodeDict=segmentNodeDict)

        pathInitFeatsInf = getPathFeatPadWithInf(numpathnode=num_node_path,
                                                    PathNodeDict_reverse=PathNodeDict_reverse,
                                                    max_size=seqmaxsize,
                                                    segment_feats=segmentInitFeat,
                                                    segmentNodeDict=segmentNodeDict)

        segmentMinMaxScalerFeat = g_t.nodes['segment'].data['feature']
        pathFeat = getPathFeat(numpathnode=num_node_path,
                                PathNodeDict_reverse=PathNodeDict_reverse,
                                max_size=seqmaxsize,
                                segment_feats=segmentMinMaxScalerFeat,
                                segmentNodeDict=segmentNodeDict)
        g_t.nodes['path'].data['pathSegmentFeat'] = pathFeat

        
        pathAvgSpeedLaneLimSpeed = pathInitFeats.index_select(dim=2, index=torch.LongTensor([1, 2, 3]))
        pathAvgSpeedLaneLimSpeedSum = torch.sum(pathAvgSpeedLaneLimSpeed, dim=1)
        pathLenTensor = torch.tensor(list(pathLen.values())).reshape(num_node_path, 1)
        averagepathLaneLimSpeed = pathAvgSpeedLaneLimSpeedSum / pathLenTensor
        
        pathAvgSpeedLaneLimSpeedLaneNum = pathInitFeats.index_select(dim=2, index=torch.LongTensor([1, 2, 3]))
        pathSpeedLaneLaneNumLimSpeedSumMax = torch.max(pathAvgSpeedLaneLimSpeedLaneNum, dim=1)[0]
        
        pathLength = pathInitFeats.index_select(dim=2, index=torch.LongTensor([4]))
        pathSumLength = torch.sum(pathLength, dim=1)
        
        pathLaneNumPadInf = pathInitFeatsInf.index_select(dim=2, index=torch.LongTensor([1, 2, 3]))
        pathMinavgSpeedLaneNumLimSpeedMin = torch.min(pathLaneNumPadInf, dim=1)[0]

        onePathFeature = torch.cat(
            [averagepathLaneLimSpeed, pathSpeedLaneLaneNumLimSpeedSumMax, pathMinavgSpeedLaneNumLimSpeedMin,
                pathSumLength],
            dim=1)
        onePathFeat = max_min_scaler.fit_transform(np.array(onePathFeature.float()))
        g_t.nodes['path'].data['onePathFeat'] = torch.tensor(onePathFeat)

        
        num_node_od = g_t.num_nodes("od")
        od_count_t = total_od_count_t[t]
        if od_count_t != {}:
            od2path_t_array = getOd2Path_t(t, total_path_count_t, allOd_dict, pathNodeDict)
            od_id = od2path_t_array[0:, 0]
            odNum = torch.zeros(1, 1)
            for i in range(num_node_od):
                if i in od_id:
                    odNum = torch.vstack((odNum, torch.tensor(od_count_t[allOd_dict_reversed[i]])))
                else:
                    odNum = torch.vstack((odNum, torch.tensor([0.])))
            odNum = odNum[1:]
            g_t.nodes['od'].data['odNum'] = odNum
        else:
            g_t.nodes['od'].data['odNum'] = torch.zeros((num_node_od, 1))

        
        g_t.edges['pass+'].data['orderInfo'] = torch.tensor(df_path2areaOrder['orderInfo']).reshape(-1, 1)

        
        jam_segment_list = []
        if t in jam_segment_info_dict:
            # what if
            for i in range(segment_num):
                if i in jam_segment_info_dict[t]:
                    jam_segment_list.append(1)
                else:
                    jam_segment_list.append(0)
        else:
            # no whatif
            jam_segment_list = [0 for _ in range(segment_num)]

        g_t.nodes['segment'].data['isWhatif'] = torch.tensor(jam_segment_list).reshape(-1, 1)

        path_count_t = total_path_count_t[t]
        if path_count_t != {}:
            od2path_t_array = getOd2Path_t(t, total_path_count_t, allOd_dict, pathNodeDict)
            path_id = od2path_t_array[0:, 1]
            pathNum = torch.zeros(1, 1)
            for i in range(num_node_path):
                if i in path_id:
                    pathNum = torch.vstack((pathNum, torch.tensor(path_count_t[PathNodeDict_reverse[i]])))
                else:
                    pathNum = torch.vstack((pathNum, torch.tensor([0.])))
            pathNum = pathNum[1:]
            g_t.nodes['path'].data['pathNum'] = pathNum
        else:
            g_t.nodes['path'].data['pathNum'] = torch.zeros((num_node_path, 1))

        cost_t = path_cost_t[t]
        if cost_t != {}:
            od2path_t_array = getOd2Path_t(t, path_cost_t, allOd_dict, pathNodeDict)
            path_id = od2path_t_array[0:, 1]
            cost = torch.zeros(1, 1)
            for i in range(num_node_path):
                if i in path_id:
                    cost = torch.vstack((cost, torch.tensor(cost_t[PathNodeDict_reverse[i]])))
                else:
                    cost = torch.vstack((cost, torch.tensor([-1])))
            cost = cost[1:]
            g_t.nodes['path'].data['cost'] = cost
        else:
            g_t.nodes['path'].data['cost'] = -1*torch.ones((num_node_path, 1))


        hetero_num_file = f"HTGWithFeat/Hetero_{t}.bin"
        dgl.save_graphs(os.path.join(iteration_dir, hetero_num_file), [g_t])
        print(f'sumovs:{t} ,is over')





def get_scenario_num(scenario):
    pass

def save_static_data(scenarios, save_dir, iteration_step_size):
    pass


def save_iteration_trip_graph(iteration, save_dir):
    pass


def save_scenario_trip_graph(scenario, save_dir, iteration_step_size):
    iterations = os.listdir(scenario)


    for iteration in iterations[::iteration_step_size]:
        save_iteration_trip_graph(iteration, save_dir)

    save_iteration_trip_graph(iterations[-1])


def create_data(data_dir, save_dir, iteration_step_size):


    scenarios = os.listdir(data_dir)
    save_static_data(scenarios, save_dir, iteration_step_size)

    for scenario in scenarios:
        ix = get_scenario_num(scenario)
        scenario_dir = os.path.join(save_dir, ix)
        save_scenario_trip_graph(scenario, scenario_dir, iteration_step_size)

