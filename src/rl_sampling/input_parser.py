from functools import reduce
from pathlib import Path
import re, random, copy, pickle
from collections import defaultdict
import pandas as pd
from graphrp.generate_graph import GraphGen
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from .bank import Bank
#from graph_mutation import GraphMutation

#import rtree as rt 

diffusion_stack = [0, 1, 3, 2, 4, 5, 7, 6]
#skip_net = ["!float", "!ssb", "ssb", "vcc", "vss", "vssx"]
skip_net = ["vcc", "vss", "vssx"]


class Net2Node():
    """convert string netlist input to graph node"""
    def __init__(self, fname, fnode):
        self.bank = Bank()
        self.skip, self.dummy_gate, self.vcc_net, self.vss_net = self.extract_skip_net(fnode)
        if fname:
            self.data, self.labels = self.process_data(fname)
            self.p_data, self.p_labels = self.prep_placement(fname)
        #self.data, self.labels = self.process_data(fname)
        self.reduced_all_data = self.reduce_data()
        #self.all_placement = self.find_all_placement()
        #self.save_into_file(self.all_placement, "new_placement.dat")

    def codes_2_placement(self, codes):
        res_all = []
        for code in codes:
            l = len(code[0])
            res = ""
            for sec in code:
                res += sec
            res += ",%s,2"%l
            res_all.append(res)    
        return res_all    
            
    def save_into_file(self, placements, fname):
        with open(fname, "w") as f:
            f.write("code,part_length,routability\n")
            for idx in placements:
                for placement in idx:

                    l = len(placement[0])
                    for p in placement:
                        f.write("%s"%p)
                    f.write(",%s,2"%(l))
                    f.write("\n") 

    def find_all_placement(self, unique_idx=None):
        all_placement = set()
        if unique_idx == None:
            unique_idx = range(len(self.p_data))
        for idx in unique_idx:
        #for idx in range(1):
            G, edge_list_org, edge_labels = self.build_graph(idx = idx)
            all_paths = self.generate_euler_graph(G, edge_list_org)
            all_path_convert = self.build_full_placement(all_paths, edge_labels, idx=idx)   
                    
            n0 = len(all_placement)
            all_placement.update(all_path_convert)
            n1 = len(all_placement)
            #print(n0, n1)
            #if n1 > n0:
            #    print("idx: %s --> %s\tnew: %s"%(idx, n1, n1 - n0))
        return all_placement    

    def get_graph(self, edge_labels):
        vk = {v:k for k, v in edge_labels.items()}
        graph = []
        for k in sorted(vk.keys()):
            graph.append([k[2], k[3], vk[k][2]])
        return graph
    #add mutation in here    
    def _find_idx_placement(self, unique_idx=None, max_samples= 1e10):
        all_placement_with_idx = {}
        if unique_idx == None:
            unique_idx = range(len(self.p_data))
        for idx in unique_idx:
            G, edge_list_org, edge_labels = self.build_graph(idx = idx)
            #mutations
            graph = self.get_graph(edge_labels)
            mg = GraphMutation(graph)
            #generate mutation based data in here
            all_paths = mg.xxx(graph)
            #all_paths = self.generate_euler_graph(G, edge_list_org, max_samples)
            all_path_convert = list(self.build_full_placement(all_paths, edge_labels, idx=idx))   
            random.shuffle(all_path_convert)
            all_placement_with_idx[idx] = all_path_convert
        return all_placement_with_idx

    def find_idx_placement(self, unique_idx=None, max_samples= 1e10, graph_mutation=True):
        graph_mutation = False
        all_placement_with_idx = {}
        if unique_idx == None:
            unique_idx = range(len(self.p_data))
        for idx in unique_idx:
            G, edge_list_org, edge_labels = self.build_graph(idx = idx)
            if graph_mutation:
                all_paths = self.mutate_euler_graph(G, edge_list_org, max_samples)
            else:
                all_paths = self.generate_euler_graph(G, edge_list_org, max_samples)
                
            all_path_convert = list(self.build_full_placement(all_paths, edge_labels, idx=idx))   
            random.shuffle(all_path_convert)
            all_placement_with_idx[idx] = all_path_convert
        return all_placement_with_idx

    def build_full_placement(self, all_path, edge_labels, idx):
        #add inverse direction info
        all_path_convert = []
        template = self.p_data[idx].copy()
        #for item in template:
        #    print(item)
        row = len(template)
        ref = {}
        for k, v in edge_labels.items():
            sorted_key = sorted(k[0:2]) 
            is_sorted = True if tuple(sorted_key[0:2]) == v[2:4] else False
            if is_sorted:
                ref[tuple(sorted_key + [k[2]])] = v[0:2]
                if sorted_key[0] != sorted_key[1]:
                    ref[tuple(sorted_key[::-1] + [k[2]])] = v[1::-1]
            else:
                ref[tuple(sorted_key + [k[2]])] = v[1::-1]
                ref[tuple(sorted_key[::-1] + [k[2]])] = v[0:2]
        
        for path in all_path:
            inversion = [i for i in range(len(path)) if path[i][0]==path[i][1]]
            inv_n = len(inversion)
            if inv_n == 0:
                res = ["" for _ in range(row)]
                #original path
                for item in path:
                    start, stop = ref[tuple(item)]
                    step = 1 if stop > start else -1
                    for r in range(row):
                        res[r] = res[r] + "".join(template[r][start:stop:step])
                res = [item[1:] for item in res]        
                all_path_convert.append(tuple(res))
            else:
                #take care of inversion if there is self edge
                for ca in product(*[[False, True] for _ in range(inv_n)]):
                    res = ["" for _ in range(row)]
                    inv_cond = {k:v for k, v in zip(inversion, ca)}
                    for n, item in enumerate(path):
                        if n in inversion and inv_cond[n]:
                            stop, start = ref[tuple(item)]
                        else:
                            start, stop = ref[tuple(item)]
                        step = 1 if stop > start else -1
                        for r in range(row):
                            res[r] = res[r] + "".join(template[r][start:stop:step])
                    res = [item[1:] for item in res]        
                    all_path_convert.append(tuple(res))
                
        return tuple(all_path_convert) 

    def generate_euler_graph(self, G, edge_list_org, max_samples=1e9, rand=True):
        g = G.copy()
        edge_list = copy.deepcopy(edge_list_org)
        all_result = []
        all_path_result = []
        all_failed_path = []
        #start node always self.dummy_gates
        res = [self.dummy_gate]
        res_path = []
        def gen(): 
            curr_node = res[-1]
            if len(all_result) >= max_samples:
                return 
            # need to check total res size == total edge num + 1
            if len(edge_list) == 0:
                all_result.append(res[:])
                all_path_result.append(res_path[:])
                return 
            curr_edge = copy.deepcopy(edge_list[curr_node])    
            if rand:
                random.shuffle(curr_edge)
            # if current node does not have edges stop
            if len(curr_edge) == 0:
                all_failed_path.append(res[:])
                return
                #Find connected neighbor nodes
                
            neighbor = list(g[curr_node])
            #Check transition to dummy node is allowed
            block_dummy = False
            if curr_node != self.dummy_gate and self.dummy_gate in neighbor:
                #count non-self edge number.
                self_edge, single_edge = 0, 0
                for d_edge in edge_list[self.dummy_gate]:
                    if d_edge[0:2] == [self.dummy_gate, self.dummy_gate]:
                        self_edge +=1
                    elif self.dummy_gate in d_edge:
                        single_edge += 1
                if neighbor != [self.dummy_gate] and single_edge == 1:
                    block_dummy = True             
            
            #Explore all possible next node. if block_dummy, don't move to dummy 
            for edge in curr_edge:
                next_node = edge[0] if edge[0] != curr_node else edge[1]
                if next_node == self.dummy_gate and block_dummy:
                    continue
                #Remove this edge and move to next node
                g.remove_edge(*edge)
                res.append(next_node)
                res_path.append([curr_node, next_node, edge[2]])
                edge_list[curr_node].remove(edge)
                if curr_node != next_node:
                    edge_list[next_node].remove(edge)
                if len(edge_list[curr_node]) == 0:
                    edge_list.pop(curr_node)
                if len(edge_list[next_node]) == 0:
                    edge_list.pop(next_node)
                gen()
                #Back racking. Recover removed edge for next node try
                g.add_edge(*edge)
                res.pop()
                res_path.pop()
                edge_list[curr_node].append(edge)
                if curr_node != next_node: 
                    edge_list[next_node].append(edge)

        gen()
        return all_path_result

    def _reduce_data(self):
        reduced_all_data = []
        for item in self.p_data:
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

    def reduce_data(self):
        reduced_all_data = []
        
        for item in self.p_data:
            reduced_data = []
            m, n = len(item), len(item[0])
            #In case any diffusion have more than 2 times seen aligned
            #additional_break = defaultdict(list)
            for col in range(n):
                align = False
                col_val = [row[col] for row in item]
                #when all column value is same in shared diffusion
                if col%2 == 1 and len(set(col_val)) == 1:
                    align = True
                #when all gate is dummy (clar diffusion break)
                elif col%2 == 0 and set(col_val) == {self.dummy_gate}:
                    align = True
                #VCC and VSS based island    
                elif col%2 == 1 and set(col_val) == {self.vss_net, self.vcc_net}:
                    align = True
                #elif col%2 == 1:
                #    additional_break[tuple(col_val)].append(col)    
                if align:
                    reduced_data.append((item[0][col], col))    
            #new_break = [(k, v) for k, v in additional_break.items() if len(v) > 1]        
            #if new_break:
            #    print("FOUND ADDITIONAL BREAK !!!!!", new_break)
            reduced_all_data.append(reduced_data)
        return reduced_all_data            

    def build_edge_bank(self, idx):
        data = self.p_data[idx]
        r_data = self.reduced_all_data[idx]
        k = []
        for n_idx in range(len(r_data)-1):
            start = r_data[n_idx][1]
            end = r_data[n_idx+1][1]
            edge_val = tuple([d[start:end+1] for d in data])
            self.bank.add_edge(r_data[n_idx][0], r_data[n_idx+1][0], edge_val)
            k.append(self.bank.get_id(edge_val))
        #print("idx: %s -->"%idx, sorted(k))    
        return k
        
    def build_all_edge_bank(self):
        unique_idx = []
        ks_2_idx = defaultdict(list)
        ks = set()
        for idx in range(len(self.p_data)):
            old_count = len(self.bank)
            k = self.build_edge_bank(idx)
            sorted_k = tuple(sorted(k))
            if sorted_k not in ks:
                unique_idx.append(idx)
            ks_2_idx[sorted_k].append(idx)
            ks.add(sorted_k)
            new_count = len(self.bank)
            if new_count > old_count:
                print("idx %s: %s %s find %s new key"%(idx, old_count, 
                                                      new_count, new_count - old_count))
        return ks, unique_idx, ks_2_idx
    
    def build_graph(self, idx):
        data = self.p_data[idx]
        r_data = self.reduced_all_data[idx]
        G = nx.MultiGraph() 
        for n_idx in range(len(r_data)-1):
            G.add_node(r_data[n_idx][0])
            G.add_node(r_data[n_idx+1][0])
            #G.add_edge(r_data[n_idx][0],r_data[n_idx+1][0], label="%s_%s"%(r_data[n_idx][1],r_data[n_idx+1][1]))
            G.add_edge(r_data[n_idx][0],r_data[n_idx+1][0], label=(r_data[n_idx][1],r_data[n_idx+1][1],
                                                                   r_data[n_idx][0],r_data[n_idx+1][0]))
            
        #nx.draw(G, with_labels=True)
        #plt.show() 
        labels = nx.get_edge_attributes(G, "label")
        #print("label: ", labels)
        #print("edges: ", G.edges)
        #print("nodes: ", G.nodes)
        edge_list = defaultdict(list)    
        for k, v in labels.items():
            if sorted(k[:2]) + [k[2]] not in edge_list[k[0]]:
                edge_list[k[0]].append(sorted(k[:2]) + [k[2]])
            if sorted(k[:2]) + [k[2]] not in edge_list[k[1]]:
                edge_list[k[1]].append(sorted(k[:2]) + [k[2]])
        #for k in edge_list:
        #    random.shuffle(edge_list[k]) 
        return G, edge_list, labels

    def extract_skip_net(self, file_name):
        dummy_gate = ""
        vcc_net = ""
        vss_net = ""
        fin = Path(file_name)
        skip = []
        if not fin.is_file():
            print("input node name file is not exist")
        with open(file_name, "r") as f:    
            for line in f:
                line = line.strip().replace("'","").split()
                if line[1] in skip_net:
                    skip.append(line[0])
                if line[1] == "!float":
                    dummy_gate = line[0]    
                elif line[1] == "vcc":
                    vcc_net = line[0]
                elif line[1] == "vssx":
                    vss_net = line[0]
                    
        return skip, dummy_gate, vcc_net, vss_net            

    def process_data_from_code(self, codes):
        all_placement = []
        for placement, code in enumerate(codes):
            n2d = defaultdict(list)
            nstack = 0
            for net in code:
                for j, ch in enumerate(net):
                    if ch in self.skip:
                        continue
                    x = j // 2
                    y = diffusion_stack[nstack] 
                    term = "s" if j % 2 == 0 else "g"
                    n2d[ch].append((x, y, term, None))
                nstack += 1    
            all_placement.append((placement, n2d))
        return  all_placement

    def prep_placement(self, file_name):
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
                nstack = 0
                for i in range(0, len(net), chunk):
                    for j, ch in enumerate(net[i:i+chunk]):
                        if ch in self.skip:
                            continue
                        x = j // 2
                        y = diffusion_stack[nstack] 
                        term = "s" if j % 2 == 0 else "g"
                        n2d[ch].append((x, y, term, None))
                        
                    nstack += 1    
                all_placement.append((placement, n2d))
                all_label.append(label)
        return  all_placement, all_label


if __name__ == "__main__":
    n2n = Net2Node("../data/test_encode_pos.csv", "../data/test_spec.csv")
    #bank = n2n.build_edge_bank(0)
    #print("Bank: ", bank, len(n2n.bank))
    bank, unique_idx, ks_2_idx = n2n.build_all_edge_bank()
    for b, i in zip(bank, unique_idx):
        print(i, b)
    print("unique_idx: ", len(unique_idx))
    #all_placement = n2n.find_all_placement(unique_idx)
    all_placement = n2n.find_idx_placement(unique_idx)
    
    n2n.save_into_file(all_placement, "new_placement_pos.dat")