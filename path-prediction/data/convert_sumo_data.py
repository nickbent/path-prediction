import os
import json
import copy

from math import ceil, floor

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path


ROUTES_SUFFIX = "vehroute"
SUMMARY_STRING = "summary"
SPEED_CHANGE_FILE = 'input_additional.add.xml'
DUMP_STRING = "dump"

SPEED_LIMIT_FILE = "speed_limit.csv"
PATH_COUNT_FILE = "total_path_count.txt"
OD_COUNT_FILE = "total_od_count.txt"
PATH_COST_FILE = "total_path_cost.txt"


NODE_DICT_FILENAME = "edges.json"
PATH_DICT_FILENAME = "paths.json"

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
JAM_PATH_INFO_FILE = "jam_path_info_dict.json"
JAM_NODE_INFO_FILE = "jam_node_info_dict.json"
PATH_SEGMENT_FILE = "path2segment.csv"
JAM_INFLUENCE_FILE = "jam_info.json"

SAVE_DIR_BASE = "/home/chocolatethunder/Documents/phd/dta/marl_dta/marl_dta/data/data/Nguyen"
NET_FILE = "/home/chocolatethunder/Documents/phd/dta/sumo-rl/sumo_rl/nets/Nguyen/nguyentl.net.xml"
BASE_DIR_NUM = "dta01"
edges_attributes = {"departed", "entered", "left", "arrived", "occupancy", "density", "traveltime", "speed"}


from marl_dta.data import json_dumps_tuple_keys, load_dict, is_numbers


def get_times(vehicle):
    return [float(vehicle.attrib["depart"])]+[float(t) for t in vehicle[0].attrib["exitTimes"].split(" ")]

def get_edges_path_od(vehicle):
    edges = vehicle[0].attrib["edges"].split(" ")
    path = tuple(edges)
    od = (vehicle.attrib["fromTaz"], vehicle.attrib["toTaz"])
    return edges, path, od

def gen_time_indices(time1, time2, max_index, time_interval=60):
    lower_bound = min(floor(time1/time_interval), max_index)
    upper_bound = min(ceil(time2/time_interval)+1, max_index)
    return [i for i in range(lower_bound, upper_bound)]

def add_to_dict(d, key, indices, start):
    if key not in d:
        d[key] = start
    d[key][indices] +=1

def add_info_iteration(iteration_dir, paths, od):
    files = os.listdir(iteration_dir)
    routes_file = [f for f in files if ROUTES_SUFFIX in f][0]
    routes = ET.parse(os.path.join(iteration_dir, routes_file)).getroot()

    for vehicle in routes:
        _, path_id, od_id = get_edges_path_od(vehicle)
        paths.add(path_id)
        od.add(od_id)


def get_paths_ods(results_dir):
    scenarios = os.listdir(results_dir)
    paths = set()
    od = set()

    for scenario in scenarios:
        iterations = os.listdir(os.path.join(results_dir, scenario))

        iterations = [i for i in iterations if is_numbers(i)]

        for iteration in iterations:
            iteration_dir = os.path.join(os.path.join(results_dir, scenario), iteration)
            add_info_iteration(iteration_dir, paths, od)


    paths = {path:ix for ix, path in enumerate(paths)}
    od = {o:ix for ix, o in enumerate(od)}

    with open('paths.json', "w") as f:
        json.dump(json_dumps_tuple_keys(paths), f)


    with open('od.json', "w") as f:
        json.dump(json_dumps_tuple_keys(od), f)

    return paths, od



def save_static_info(net):

    edges = [el for el in net if el.tag == 'edge' and  'function' not in el.attrib ]

    num_lanes = []
    segment_speed = []
    segment_length = []
    for edge in edges:
        num_lanes.append(len(edge))
        segment_speed.append(edge[0].attrib["speed"])
        segment_length.append(edge[0].attrib["length"])

    edges_dict = set(edge.attrib["id"] for edge in edges)
    edges_dict = {edge:ix for ix, edge in enumerate(list(edges_dict))}

    connections = [el for el in net if el.tag == 'connection' ]
    connections_set = []

    for connection in connections:
        if connection.attrib["from"] not in edges_dict or connection.attrib["to"] not in edges_dict:
            continue

        connections_set.append((edges_dict[connection.attrib["from"]], edges_dict[connection.attrib["to"]]))
    connections_set = set(connections_set)

    fro = []
    to = []

    for f,t in list(connections_set):
        fro.append(f)
        to.append(t)

    df = pd.DataFrame(columns = ["connection_from", "connection_to"])
    df["connection_from"] = fro
    df["connection_to"] = to

    with open('edges.json', "w") as f:
        json.dump(edges_dict, f)

    df.to_csv("neighbour.csv", index = False)
    static_info  = {"segment_id":edges_dict.keys(), "segment_numLanes":num_lanes, "segment_speed":segment_speed, "segment_length":segment_length}
    pd.DataFrame(static_info).to_csv("static_info.csv", index = False)




def get_indices(x: list, value: int) -> list:
    indices = list()
    non_indices = list()
    for i in range(len(x)):
        if x[i] == value:
            indices.append(i)
        else:
            non_indices.append(i)
    return np.array(indices), np.array(non_indices)

def get_closest(array, values):
    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")
    
    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1
    
    return array[idxs]


def convert_cost(cost, all_times_min):


    path_cost_t = [ { path:[] for path in cost.keys() } for t in all_times_min]


    for path, costs in cost.items():
        path_cost = [[] for t in all_times_min]
        curr_ix = 0
        costs = dict(sorted(costs.items())) 
        for t, c in costs.items():
            ix = floor(t/60)
            path_cost[ix].append(c)
            if ix > curr_ix:
                new_val = path_cost[curr_ix]
                if new_val:
                    new_val = np.mean(new_val)
                path_cost[curr_ix] = new_val
                curr_ix = ix
        new_val = path_cost[ix]
        if new_val:
            new_val = np.mean(new_val)
        path_cost[ix] = new_val
        indices, non_indices = get_indices(path_cost, [])
        closest_ix = get_closest(non_indices, indices)
        for c_ix, ix in zip(closest_ix, indices):
            path_cost[ix] = path_cost[c_ix]
        
        for t, c in enumerate(path_cost):
            path_cost_t[t][path] = c

    return path_cost_t

def convert_to_total_counts(d, all_times_min):
    total_count = []
    for i in range(len(all_times_min)):
        count = {}
        for k,v in d.items():
            if v[i] >0:
                count[k] = v[i]
        total_count.append(count)
    return total_count


def get_total_counts(summary, routes, paths):

    all_times = [float(s.attrib["time"]) for s in summary]
    all_times_min = np.zeros(ceil(len(all_times)/60))
    max_index = len(all_times_min)-1

    edges = {}
    paths = { path_id: np.copy(all_times_min) for path_id in paths}
    od = {}
    cost = {}


    for vehicle in routes:
        times = get_times(vehicle)
        edges_id, path_id, od_id = get_edges_path_od(vehicle)
        edges_time_indices = [gen_time_indices(time1, time2, max_index) for time1, time2 in zip(times[:-1], times[1:])]
        ix = [edges_time_indices[0]]
        for e in edges_time_indices[1:]:
            ix.append(e[1:])
        edges_time_indices = ix
        path_time_indices = gen_time_indices(times[0], times[-1], max_index)
        od_time_indices = path_time_indices

        for ix, edge_id in zip( edges_time_indices, edges_id):
            add_to_dict(edges, edge_id, ix, np.copy(all_times_min))

        add_to_dict(paths, path_id, path_time_indices, np.copy(all_times_min))

        if path_id not in cost:
            cost[path_id] = {}
        cost[path_id][times[0]] = times[-1]-times[0]

        add_to_dict(od, od_id, od_time_indices,np.zeros(ceil(len(all_times)/60)))


    total_path_count_t = convert_to_total_counts(paths, all_times_min)
    total_od_count_t = convert_to_total_counts(od, all_times_min)
    cost = convert_cost(cost, all_times_min)

    return total_path_count_t, total_od_count_t, cost


def get_edge_variables(dump, edges):

    edges_entered = {}
    for edge in edges:
        edges_entered[edge.attrib["id"]] = np.zeros(len(dump))


    edges_departed = copy.deepcopy(edges_entered)
    edges_count_start = copy.deepcopy(edges_entered)
    edges_left = copy.deepcopy(edges_entered)
    edges_arrived = copy.deepcopy(edges_entered)
    edges_count_exit = copy.deepcopy(edges_entered)
    edges_occupancy = copy.deepcopy(edges_entered)
    edges_density = copy.deepcopy(edges_entered)
    edges_travel_time = copy.deepcopy(edges_entered)
    edges_count_inside = copy.deepcopy(edges_entered)
    edges_speed = copy.deepcopy(edges_entered)


    for ix, step in enumerate(dump):
        for e in step:
            if not edges_attributes.issubset(set(e.attrib.keys())):
                continue
            edges_departed[e.attrib["id"]][ix] = float(e.attrib["departed"])
            edges_entered[e.attrib["id"]][ix] = float(e.attrib["entered"])
            edges_count_start[e.attrib["id"]][ix] = float(e.attrib["departed"])+float(e.attrib["entered"])

            edges_left[e.attrib["id"]][ix] = float(e.attrib["left"])
            edges_arrived[e.attrib["id"]][ix] = float(e.attrib["arrived"])
            edges_count_exit[e.attrib["id"]][ix] = float(e.attrib["left"])+float(e.attrib["arrived"])
            edges_occupancy[e.attrib["id"]][ix] = float(e.attrib["occupancy"])
            edges_density[e.attrib["id"]][ix] = float(e.attrib["density"])
            edges_travel_time[e.attrib["id"]][ix] = float(e.attrib["traveltime"])
            edges_speed[e.attrib["id"]][ix] = float(e.attrib["speed"])
            if ix >0:
                c = edges_count_inside[e.attrib["id"]][ix-1]
            else:
                c =0
            
            edges_count_inside[e.attrib["id"]][ix] = c+ edges_count_start[e.attrib["id"]][ix]-edges_count_exit[e.attrib["id"]][ix]

    return edges_departed, edges_count_start, edges_left, edges_arrived, edges_count_exit, edges_occupancy, edges_density, edges_travel_time, edges_count_inside, edges_speed




def find_edge_from_lanes(edges, lanes):

    for edge in edges:
        if set([lane.attrib["id"] for lane in edge] ) == set(lanes):
            return edge.attrib["id"]


def get_speed_limit(speed_change, edges, dump):

    speed_limits = []
    for step in speed_change:
        start_time = step[0].attrib["time"]
        end_time = step[1].attrib["time"]
        speed = step[0].attrib["speed"]
        lanes = step.attrib["lanes"].split()
        edge_id = find_edge_from_lanes(edges, lanes)
        speed_limits.append({"edge": edge_id, "start_time":start_time, "end_time":end_time, "speed":speed})

    speed_limits_all = {}
    time_step = 60

    for edge in edges:
        speed_limit = np.ones(len(dump))*float(edge[0].attrib["speed"])
        for speed in speed_limits:
            if speed["edge"] == edge.attrib["id"]:
                ix_start = int(float(speed["start_time"])/time_step)
                ix_end = int(float(speed["end_time"])/time_step)
                speed_limit[ix_start:ix_end] = float(speed["speed"])
        speed_limits_all[edge.attrib["id"]] = speed_limit
    return speed_limits_all


def find_paths(edge_id, edges_path):
    return [path for path in edges_path if edge_id in path]


def convert_influ_to_node_path(jam_info, edges, paths):

    jam_segment_info_dict = {}
    jam_path_info_dict = {}
    time_step = 60


    for jam in jam_info:
        time_stamp = int(float(jam["time"])/time_step)
        if time_stamp not in jam_segment_info_dict:
            jam_segment_info_dict[time_stamp] = []
        jam_segment_info_dict[time_stamp].append(edges[jam["edge"]])

        for path in jam["paths"]:
            if time_stamp not in jam_path_info_dict:
                jam_path_info_dict[time_stamp] = []
            jam_path_info_dict[time_stamp].append(paths[path])

    return jam_path_info_dict, jam_segment_info_dict


def get_jam_influ(base_converged, converged, paths):

    relevant_attribs = ["traveltime", "density", "occupancy", "speed"]
    tolerance = 0.1
    jam_influ = []
    for base_step, step in zip(base_converged, converged):
        base_edges_step = { base_edge.attrib["id"] : base_edge for base_edge in base_step}
        edges_step = {edge.attrib["id"] : edge for edge in step}
        b_k = set(base_edges_step.keys())
        e_k = set(edges_step.keys())
        overlap_edges = list(b_k&e_k)
        non_overlap_edges = list(b_k^e_k)
        for key in overlap_edges:
            edge = edges_step[key]
            base_edge = base_edges_step[key]
            for attribs in relevant_attribs:
                if attribs not in edge.attrib or attribs not in base_edge.attrib:
                    continue
                if 1- tolerance > (float(edge.attrib[attribs])/float(base_edge.attrib[attribs])) or float(edge.attrib[attribs])/float(base_edge.attrib[attribs])  > 1+tolerance:
                    affected_paths = find_paths(edge.attrib["id"], paths)
                    jam_influ.append({"time": step.attrib["begin"], "edge": edge.attrib["id"], "paths":affected_paths, "effect": float(edge.attrib[attribs])/float(base_edge.attrib[attribs])})
                    break
        for key in non_overlap_edges:
            if key in edges_step:
                edge = edges_step[key]
                affected_paths = find_paths(edge.attrib["id"], paths)
            else:
                edge = base_edges_step[key]
                affected_paths = find_paths(edge.attrib["id"], paths)
            affected_paths = find_paths(edge.attrib["id"], paths)
            jam_influ.append({"time": step.attrib["begin"], "edge": key, "paths":affected_paths, "effect": 1})

    return jam_influ



def get_file_in_dir(dir, string):
    files = os.listdir(dir)
    file = [file for file in files if string in file][0]
    return os.path.join(dir, file)

def get_file_in_save_dir(save_dir,file):
    return os.path.join(save_dir, file)
    

def save_jam_data(scenario_dir, base_dir, scenario_save_dir, converged_iteration, edges, edges_dict, paths):
    speed_change =  ET.parse(get_file_in_dir(scenario_dir, SPEED_CHANGE_FILE)).getroot()

    converged = ET.parse(get_file_in_dir(os.path.join(base_dir,converged_iteration), DUMP_STRING)).getroot()
    base_converged = ET.parse(get_file_in_dir(os.path.join(scenario_dir,converged_iteration), DUMP_STRING)).getroot()
    speed_limits_all = get_speed_limit(speed_change, edges, converged)
    jam_influ  = get_jam_influ(base_converged, converged, paths)
    jam_path_info_dict, jam_node_info_dict = convert_influ_to_node_path(jam_influ, edges_dict, paths)

    pd.DataFrame(speed_limits_all).to_csv(os.path.join(scenario_save_dir, SPEED_LIMIT_FILE), index = False)
    with open(os.path.join(scenario_save_dir, JAM_PATH_INFO_FILE), "w") as f:
        json.dump(jam_path_info_dict, f)
    with open(os.path.join(scenario_save_dir, JAM_NODE_INFO_FILE), "w") as f:
        json.dump(jam_node_info_dict, f)


def save_iteration_data(iteration_dir, edges, paths, save_dir):
    routes = ET.parse(get_file_in_dir(iteration_dir, ROUTES_SUFFIX)).getroot()
    summary = ET.parse(get_file_in_dir(iteration_dir, SUMMARY_STRING)).getroot()
    dump = ET.parse(get_file_in_dir(iteration_dir, DUMP_STRING)).getroot()



    edges_departed, edges_count_start, edges_left, edges_arrived, edges_count_exit, edges_occupancy, edges_density, edges_travel_time, edges_count_inside, edges_speed = get_edge_variables(dump, edges)
    total_path_count_t, total_od_count_t, path_cost_t = get_total_counts(summary, routes, paths)


    pd.DataFrame(edges_departed).to_csv(os.path.join(save_dir,EDGES_DEPARTED_FILE), index = False)
    pd.DataFrame(edges_count_start).to_csv(os.path.join(save_dir,EDGES_COUNT_FILE), index = False)
    pd.DataFrame(edges_left).to_csv(os.path.join(save_dir,EDGES_LEFT_FILE), index = False)
    pd.DataFrame(edges_arrived).to_csv(os.path.join(save_dir,EDGES_ARRIVED_FILE), index = False)
    pd.DataFrame(edges_count_exit).to_csv(os.path.join(save_dir,EDGES_COUNT_EXIT_FILE), index = False)
    pd.DataFrame(edges_occupancy).to_csv(os.path.join(save_dir,EDGES_OCCUPANCY_FILE), index = False)
    pd.DataFrame(edges_density).to_csv(os.path.join(save_dir,EDGES_DENSITY_FILE), index = False)
    pd.DataFrame(edges_travel_time).to_csv(os.path.join(save_dir,EDGES_TRAVEL_TIME_FILE), index = False)
    pd.DataFrame(edges_count_inside).to_csv(os.path.join(save_dir,EDGES_COUNT_INSIDE_FILE), index = False)
    pd.DataFrame(edges_speed).to_csv(os.path.join(save_dir,EDGES_SPEED_FILE), index = False)


    with open(os.path.join(save_dir,PATH_COUNT_FILE), "w") as f:
        f.write(str(total_path_count_t))


    with open(os.path.join(save_dir,OD_COUNT_FILE), "w") as f:
        f.write(str(total_od_count_t))

    with open(os.path.join(save_dir,PATH_COST_FILE), "w") as f:
        f.write(str(path_cost_t))

def save_scenario_data(edges, edges_dict, paths_dict, scenario_dir, scenario_save_dir, base_dir):



    iteration_dirs = os.listdir(scenario_dir)
    iteration_dirs = [i for i in iteration_dirs if is_numbers(i)]
    iteration_dirs.sort()

    for iteration in iteration_dirs:
        iteration_dir = os.path.join(scenario_dir, iteration)
        save_iteration_dir = os.path.join(scenario_save_dir, iteration)
        Path(save_iteration_dir).mkdir(parents=True, exist_ok=True)
        save_iteration_data(iteration_dir, edges, paths_dict, save_iteration_dir)

    if scenario_dir != base_dir:
        save_jam_data(scenario_dir, base_dir, scenario_save_dir, iteration, edges, edges_dict, paths_dict)
    else:
        converged = ET.parse(get_file_in_dir(os.path.join(scenario_dir, iteration), DUMP_STRING)).getroot()
        speed_limits_all = get_speed_limit([], edges, converged)
        pd.DataFrame(speed_limits_all).to_csv(os.path.join(scenario_save_dir, SPEED_LIMIT_FILE), index = False)