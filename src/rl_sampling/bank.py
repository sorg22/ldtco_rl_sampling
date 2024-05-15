import os
from typing import NewType
from collections import defaultdict

class Bank():
    def __init__(self):
        self.db = {} #key: node1_node2_length Value:{data:key_id}
        self.db_count = defaultdict(int)
    
    def add_edges(self, node0, node1, data):
        pass

    def _add_edge(self, node0, node1, data):
        l = len(data[0])
        if node0 > node1:
            node0, node1 = node1, node0
            data = tuple([item[::-1] for item in data])
        k = f"{node0}_{node1}_{l}"
                 
        if k not in self.db:
            self.db[k] = {data: f"{k}_0"}
            self.db_count[k] += 1
            rdata = tuple([item[::-1] for item in data])
            #if node0 == node1 and data != rdata:
            #    self.db[k].update({rdata: f"{k}_1"})
            #    self.db_count[k] += 1
        else:
            if data not in self.db[k]:
                self.db[k][data] = f"{k}_{self.db_count[k]}"
                self.db_count[k] += 1            
                rdata = tuple([item[::-1] for item in data])
            #    if node0 == node1 and data != rdata:
            #        self.db[k][rdata] = f"{k}_{self.db_count[k]}"
            #        self.db_count[k] += 1 
    def add_edge(self, node0, node1, data):
        l = len(data[0])
        rdata = tuple([item[::-1] for item in data])
        if data > rdata:
            node0, node1 = node1, node0
            data, rdata = rdata, data
        k = f"{node0}_{node1}_{l}"
                 
        if k not in self.db:
            self.db[k] = {data: f"{k}_0"}
            self.db_count[k] += 1
            #rdata = tuple([item[::-1] for item in data])
            #if node0 == node1 and data != rdata:
            #    self.db[k].update({rdata: f"{k}_1"})
            #    self.db_count[k] += 1
        else:
            if data not in self.db[k]:
                self.db[k][data] = f"{k}_{self.db_count[k]}"
                self.db_count[k] += 1            
            #    rdata = tuple([item[::-1] for item in data])
            #    if node0 == node1 and data != rdata:
            #        self.db[k][rdata] = f"{k}_{self.db_count[k]}"
            #        self.db_count[k] += 1
                        
    def get_id(self, data):
        l = len(data[0])
        rdata = tuple([item[::-1] for item in data])
        if data > rdata:
            data, rdata = rdata, data
        k = f"{data[0][0]}_{data[0][-1]}_{l}"
        if k not in self.db:
            return None
        else:
            return self.db[k].get(data, None)
                
    def get_seq(self, data):
        pass
    
    def remove_edge(self):
        pass
        
    def __len__(self):
        tot = 0
        for v in self.db_count.values():
            tot += v
        return tot    

if __name__ == "__main__":
    #n2n = Net2Node("../data/test_encode_pos.csv", "../data/test_spec.csv")
    #all_placement = n2n.find_all_placement()
    #n2n.save_into_file(all_placement, "new_placement.dat")
    pass