import os
import xml.etree.ElementTree as ET


from collections import namedtuple
from copy import deepcopy

from path_prediction.data.dataset import File



JamItem = namedtuple("jam_item", ["id", "lanes"])
JamTime = namedtuple("jam_time", ["time", "speed"])
JamInstance = namedtuple("jam_instance", ["jam_item", "jam_start", "jam_end"])


class JamInfoFile(File):

    base_jam_file = "/home/chocolatethunder/Documents/phd/dta/sumo-rl/sumo_rl/nets/Nguyen/input_additional.add.base.xml"
    jam_info_file_name = "input_additional.add.xml"
        
    @staticmethod
    def open(path):
        return ET.parse(path)
    
    @staticmethod
    def save(jam_info, path):

        jam_info.write(path)

    @staticmethod
    def get_jam_path(results_dir):
        return os.path.join(results_dir, JamInfoFile.jam_info_file_name)


class Jam:

    id = "vss"
    end_speed = "-1"
        
    def __init__(self, num_accidents_per_scenario,
                 edges,
                 time_interval_length, 
                 allowable_time_intervals,
                 allowable_traffic_jam_time_percent,
                 end_time, 
                 min_jam_percentage, 
                 max_jam_percentage):
        
        self.time_interval_length = time_interval_length
        self.final_allowable_traffic_time_interval = int(end_time*allowable_traffic_jam_time_percent/time_interval)
        self.final_allowable_traffic_time = final_allowable_traffic_time_interval*time_interval

        self.edges = edges
        self.starting_intervals = random.choices(range(self.final_allowable_traffic_time_interval), k=num_accidents_per_scenario)
        self.length_of_jams = random.choices(allowable_time_intervals, k=num_accidents_per_scenario)
        self.edges_jam = random.choices(edges, k=num_accidents_per_scenario)
        self.edges_change = [random.uniform(min_jam_percentage, max_jam_percentage) for i in edges_jam]

        self.create_jams()



    def create_jam(self,  edge, starting_interval, length_of_jam, percent_change_in_speed):

        lanes = [ lane.attrib["id"] for lane in edge]

        jam_item = JamItem(self.id, " ".join(lanes))

        speed = float(edge[0].attrib["speed"])
        start_time = int(self.time_interval_length*starting_interval)
        speed_change = speed*percent_change_in_speed
        speed_change = "{:.2f}".format(speed_change)
        jam_start = JamTime(str(start_time), speed_change)

        end_time = int(start_time+self.time_interval_length*length_of_jam)
        jam_end = JamTime(str(end_time), str(self.end_speed))

        return JamInstance(jam_item, jam_start, jam_end)


    def create_jams(self):

        jams = []
        for edge, starting_interval, length_of_jam, percent_change_in_speed in zip(self.edges, self.starting_intervals, self.length_of_jams, self.edges_change):
            jams.append(self.create_jam(edge, starting_interval, length_of_jam, percent_change_in_speed))

        self.jams = jams



    def create_jam_root(self):

        tree = JamInfoFile.open(JamInfoFile.base_jam_file)
        base_jam = tree.getroot()



        for ix, jam in enumerate(self.jams):
            jam_element = deepcopy(base_jam[0])
            self.modify_jam_element(jam_element, jam)
            if ix != 0:
                base_jam.append(jam_element)
            else:
                base_jam[0] = jam_element

        return tree, base_jam



    def modify_jam_element(self, jam_element, jam):
        
        jam_element.attrib = jam.jam_item._asdict()
        jam_element[0].attrib = jam.jam_start._asdict()
        jam_element[1].attrib = jam.jam_end._asdict()


ALLOWABLE_TRAFFIC_JAM_TIME_PERCENT = 3/4
time_interval = float(base_converged[0].attrib["end"]) - float(base_converged[0].attrib["begin"])
end_time = float(base_converged[-1].attrib["end"])
final_allowable_traffic_time_interval = int(end_time*ALLOWABLE_TRAFFIC_JAM_TIME_PERCENT/time_interval)
final_allowable_traffic_time = final_allowable_traffic_time_interval*time_interval
jam = Jam(num_accidents_per_scenario, edges, time_interval, allowable_time_intervals, ALLOWABLE_TRAFFIC_JAM_TIME_PERCENT, end_time, min_jam_percentage, max_jam_percentage)
tree, root = jam.create_jam_root()
JamInfoFile.save(tree, JamInfoFile.get_jam_path("/home/chocolatethunder/Documents/phd/dta/sumo-rl/sumo_rl/nets/Nguyen/results/dta10"))