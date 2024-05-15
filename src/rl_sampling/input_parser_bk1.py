from functools import reduce
from pathlib import Path
import re, random, copy
from collections import defaultdict
import pandas as pd
from graphrp.generate_graph import GraphGen
import networkx as nx
import matplotlib.pyplot as plt



#import rtree as rt 

diffusion_stack = [0, 1, 3, 2, 4, 5, 7, 6]
#skip_net = ["!float", "!ssb", "ssb", "vcc", "vss", "vssx"]
skip_net = ["vcc", "vss", "vssx"]


class Net2Node():
    """convert string netlist input to graph node"""
    def __init__(self, fname, fnode):
        self.skip, self.dummy_gate = self.extract_skip_net(fnode)
        self.data, self.labels = self.process_data(fname)
        self.reduced_all_data = self.reduce_data()
        self.G, self.edges = self.build_graph(0)
        print ("org edges: ", self.edges)
        self.generate_euler_graph()
        
    def generate_euler_graph(self):
        g = self.G.copy()
        edges = copy.deepcopy(self.edges)
        print("start : ", g.edges("A"))
        all_paths = []
        #start node always self.dummy_gates
        res = [self.dummy_gate]
        def gen(res): 
            if len(edges) == 0:
                all_paths.append(res[:])
                return 
            
            while(edges):
                curr= res[-1]
                neighbor = list(g[curr])
                random.shuffle(neighbor)
                #reachable = nx.algorithms.descendants(g, curr)
                found_valid_next_node = False
                try_idx = 0
                while not found_valid_next_node and len(neighbor) > try_idx:
                    try_node = neighbor[try_idx]
                    node_pair_key = tuple(sorted([curr, try_node]))
                    all_candidate_edges = edges[node_pair_key]
                    g.remove_edge(curr, try_node, all_candidate_edges[0][0])
                    print(nx.algorithms.descendants(g, curr))
                    print(nx.algorithms.descendants(g, try_node))
                    connection_curr = curr == self.dummy_gate or self.dummy_gate in nx.algorithms.descendants(g, curr) or g.degree(curr)==0
                    connection_next = try_node == self.dummy_gate or self.dummy_gate in nx.algorithms.descendants(g, try_node)
                    
                    if connection_curr and connection_next:
                        found_valid_next_node = True
                        res.append(try_node)
                        #remove edges form all_candidate_edges
                        edges[node_pair_key].pop(0)
                        if len(edges[node_pair_key]) == 0:
                            edges.pop(node_pair_key)
                    else:
                        #recover edge if it is not valid edge selection
                        g.add_edge(curr, try_node, label=all_candidate_edges[0][0])
                    try_idx += 1
                print("%s %s is valid transition"%(node_pair_key))    
            print(res)
            print(all_paths)
            print(len(all_paths))
            print()
                

        gen(res)
                        
            
            
        

            

    def reduce_data(self):
        reduced_all_data = []
        for item in self.data:
            reduced_data = []
            m, n = len(item), len(item[0])
            for col in range(n):
                align = True
                for row in range(1, m):
                    if item[0][col] != item[row][col]:
                        align = False
                        break
                if align:
                    reduced_data.append((item[0][col], col))    
            reduced_all_data.append(reduced_data)
        return reduced_all_data    
            
    def build_graph(self, idx):
        data = self.data[idx]
        r_data = self.reduced_all_data[idx]
        for d in data:
            print(d)
        G = nx.MultiGraph() 
        for n_idx in range(len(r_data)-1):
            G.add_node(r_data[n_idx][0])
            G.add_node(r_data[n_idx+1][0])
            #G.add_edge(r_data[n_idx][0],r_data[n_idx+1][0], label="%s_%s"%(r_data[n_idx][1],r_data[n_idx+1][1]))
            G.add_edge(r_data[n_idx][0],r_data[n_idx+1][0], label=(r_data[n_idx][1],r_data[n_idx+1][1]))
            
        #nx.draw(G, with_labels=True)
        #plt.show() 
        labels = nx.get_edge_attributes(G, "label")
        #print("label: ", labels)
        print("edges: ", G.edges)
        print("nodes: ", G.nodes)
        print("edges connected to A: ", G.edges[('A', 'm', 0)])
        edges = defaultdict(list)
        for k, v in labels.items():
            edges[tuple(sorted(k[:2]))].append((k[2], v))
        for k in edges:
            random.shuffle(edges[k]) 
        return G, edges

    def extract_skip_net(self, file_name):
        dummy_gate = ""
        fin = Path(file_name)
        skip = []
        if not fin.is_file():
            print("input node name file is not exist")
        with open(file_name, "r") as f:    
            for line in f:
                line = line.strip().replace("'","").split()
                print(line)
                if line[1] in skip_net:
                    skip.append(line[0])
                if line[1] == "!float":
                    dummy_gate = line[0]    
        return skip, dummy_gate            

    def process_data(self, file_name):
        fin = Path(file_name)
        if not fin.is_file():
            print("input file is not exist")
        with open(file_name, "r") as f:
            chunk = None
            header = f.readline()
            #print("file header: ", header)
            all_placement = []
            all_label = []
            for placement, line in enumerate(f):
                line = line.strip()
                net, chunk, label = line.split(",")
                chunk, label = int(chunk), int(label)
                net = list(net)
                if not len(net)%chunk == 0:
                    print("Wrong ascii code length. line len: %s chuck: %s"%(len(net), chunk))
                    return None

                n2d = defaultdict(list)
                stacks = []
                for i in range(0, len(line), chunk):
                    if i + chunk < len(line):
                        stacks.append(self.dummy_gate + line[i:i+chunk] + self.dummy_gate)
                        
                all_placement.append(stacks)
                all_label.append(label)
                
        return  all_placement, all_label

if __name__ == "__main__":
    n2n = Net2Node("../data/test_encode_pos.csv", "../data/test_spec.csv")
    #print(n2n.reduced_all_data)
    #for org, data in zip(n2n.data, n2n.reduced_all_data):
    #    print(data)
    #    for d in org:
    #        print(d)
        #for item in data:
        #    print (item)
        #print()    