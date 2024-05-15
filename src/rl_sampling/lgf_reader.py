import pandas as pd
import os
import re
from collections import defaultdict
from pathlib import Path
from shapely import box


class LgfReader():
    """
    lgf file reader. It only read wire section 
    process polygon boolean operation for calculating area
    """
    
    def __init__(self, fname: str):
        """
        init function

        Parameters:
            fname (str) : lgf file name with full path
        """
        self.fname = fname
        self.net_layer = defaultdict(list)
        self.net =set()
        self.layer =set()
        data = self.read_files()
        wire_db = self.read_wire(data)
        
    def read_files(self):
        """
        read lgf file
        
        Returns
        -------
        list
            list of lines
        """
        if Path(self.fname).exists():
            data = []
            with open(self.fname, "r") as f:
                for line in f.readlines():
                    if line.startswith("Wire "):
                        data.append(line)
            return data            
        else:
            print("check your file path. it is not exist")        
            return []

    def read_wire(self, data: list) -> dict:
        """
        read wire session

        Parameters:
        data (list): list of lines with wire information
        
        Return:
        
        """      
        #Wire net=nc1 layer=wirepoly rect=8090:755:8230:757 payload=PL_Gate invisible=1
        pattern = re.compile(r"Wire\s+net=(.*)\s+layer=(.*)\s+rect=(.*)\s+payload=(.*)\s+.*")
        for item in data:
            matched = pattern.match(item)
            if matched:
                #print(matched.groups())                   
                net, layer, rect, payload = matched.groups()
                self.net_layer[(net, layer)].append(tuple([int(it) for it in rect.split(":")]))
                self.layer.add(layer)
                self.net.add(net)
            else:
                print("regular expression for Wire session need to be updated")

        return None

    def get_area(self, layer, net=None) -> area:int:
        """
        generate layer union area. if net is not defined sum for all the net
        
        parameters:
        layer (str): 
        """
        
  




if __name__ == "__main__":
    fname = "/nfs/site/disks/ad_wa_skim501/ldtco_rl_sampling/data/k0mfvz203at12b1x5_r0.lgf"
    reader = LgfReader(fname)