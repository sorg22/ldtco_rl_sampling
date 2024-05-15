from inspect import classify_class_attrs
from nis import match
import numpy as np
from input_parser import Net2Node
from collections import defaultdict
from itertools import product, chain
from copy import copy
import networkx as nx


p_pair = [1, 2]
n_pair = [0, 3]

def get_explicit_path(item):
    path = []
    prev = "A"
    for it in item:
        start = prev
        end = it[1] if prev == it[0] else it[0] 
        path.append([start, end, it[2]])
        prev = end
    return path
    
def save_into_file(placements, fname):
    with open(fname, "w") as f:
        f.write("code,part_length,routability\n")
        for placement in placements:
            l = len(placement[0])
            for p in placement:
                f.write("%s"%p)
            f.write(",%s,2"%(l))
            f.write("\n")

def merge_into_file(placements, ref_fname, fname):

    with open(ref_fname, "r") as f:
        chunk = None
        header = f.readline()
        all_ref_placement = dict()
        for placement, line in enumerate(f):
            line = line.strip()
            net, chunk, label = line.split(",")
            chunk, label = int(chunk), int(label)
            if not len(net)%chunk == 0:
                print("Wrong ascii code length. line len: %s chuck: %s"%(len(net), chunk))
                return None
            data = []
            for i in range(int(len(net)/chunk)):    
                data.append(net[i*chunk:(i+1)*chunk])
            all_ref_placement[tuple(data)] = label    


    with open(fname, "w") as f:
        f.write("code,part_length,routability\n")
        for placement, label in all_ref_placement.items():
            l = len(placement[0])
            for p in placement:
                f.write("%s"%p)
            f.write(",%s,%s"%(l, label))
            f.write("\n")
        for placement in placements:
            if placement in all_ref_placement:
                continue
            l = len(placement[0])
            for p in placement:
                f.write("%s"%p)
            f.write(",%s,2"%(l))
            f.write("\n")


class SwapRow():
    def __init__(self, fname, spec):
        self.n2n = Net2Node(fname, spec)
        self.bank, self.unique_idx, self.ks_2_idx = self.n2n.build_all_edge_bank() 
        self.idx_new_placement, self.new_placement = self.gen_new_placement()
        merge_into_file(self.new_placement, fname, "test_encode_swap.csv") 
        #save_into_file(self.new_placement, "new_swap_placement.dat") 

    def same_graph(self, ref, placements):
        g_ref = self.build_graphs(ref)
        for p in placements:
            g_p = self.build_graphs(p)
            #res =  nx.vf2pp_is_isomorphic(g_ref, g_p, node_label=None)
            #res =  nx.is_isomorphic(g_ref, g_p, node_match=None)
            res =  nx.is_isomorphic(g_ref, g_p)
            if not res:
                print("Failed !!!!!")
        
    def gen_new_placement(self):
        idx_new_placement = defaultdict(list)
        for idx in self.unique_idx:
            G, edge_list_org, edge_labels = self.n2n.build_graph(idx = idx)
            path, swaped_paths = self.swap_columns(edge_labels)
            path_convert = list(self.build_full_placement([path], edge_labels, idx=idx)) 
            swaped_paths_converted = list(self.build_full_placement(swaped_paths, edge_labels, idx=idx, ref_placement=path_convert))
            #graph_isomorphic_testing
            self.same_graph(path_convert[0], swaped_paths_converted)

            ref_area = self.calc_conjestion(path_convert[0])
            ref_align = self.calc_align(path_convert[0])
            for item in swaped_paths_converted:
                area = self.calc_conjestion(item)
                align = self.calc_align(item)
                if ref_area * 1.0 >= area and ref_align <= align:
                    print(idx, ref_area, area, ref_align, align)
                    idx_new_placement[idx].append(item)
        new_placement = set((tuple(it) for item in idx_new_placement.values() for it in item))
        return idx_new_placement, new_placement
        
        
        
        
    def swap_columns(self, edge_labels):
        path = []
        label_edges = {v:k for k, v in edge_labels.items()}
        for k in sorted(label_edges):
            path.append([k[2], k[3], label_edges[k][2]])
        swap_candidate = defaultdict(list) 
        for k, v in edge_labels.items():
            swap_candidate[(k[0], k[1], v[1] - v[0])].append(k)
        swap_candidate  = {k:v for k, v in swap_candidate.items() if len(v) > 1}   
        #Swap_mapping
        swap_sample = defaultdict(list)
        def swap(data):
            ans = []
            idx = list(range(len(data)))
            def rec(i):
                if i == len(data):
                    ans.append(idx[:])
                for j in range(i, len(data)):
                    idx[i], idx[j] = idx[j], idx[i]
                    rec(i+1)
                    idx[i], idx[j] = idx[j], idx[i]
            rec(0)
            swapped = []
            for item in ans:
                swapped.append([data[i] for i in item])
            return swapped
                    
            
        #swapped = swap(list(swap_candidate.values())[0])
        swapped = {k:swap(v) for k, v in swap_candidate.items()}
        doe = []
        for item in product(*list(swapped.values())):
            item = chain.from_iterable(item)
            doe.append(list(item))

        #doe_ref = doe.pop(0)
        sorted_path = [sorted(item[0:2])+[item[2]] for item in path]
        all_ref = []
        for d in doe:
            ref = []
            for item in d:
                ref.append(sorted_path.index(list(item)))
            all_ref.append(ref[:])
        all_sorted_path = []    
        ref = all_ref.pop(0)
        for item in all_ref:
            p = copy(sorted_path)
            for n, idx in enumerate(item):
                if idx != ref[n]:
                    p[idx] = sorted_path[ref[n]]
            all_sorted_path.append(p)
        all_sorted_path = [sorted_path] + all_sorted_path
        all_sorted_path = [get_explicit_path(item) for item in all_sorted_path]
        return path, all_sorted_path

    def calc_align(self, template):        
        match_count = 0
        #Assume nppn two row case. In bigger cell need to handle multiple row case
        for n in range(len(template[0])):
            if template[n_pair[0]][n] == template[n_pair[1]][n]: 
                match_count += 1
            if template[p_pair[0]][n] == template[p_pair[1]][n]: 
                match_count += 1
        return match_count
        
    def calc_conjestion(self, template):
        #key: net name, value: [xmin, ymin, xmax, ymax] for BB box
        min_max = dict()
        for r in range(len(template)):
            for c in range(len(template[0])):
                net = template[r][c]
                if net in ["A", "m", "q", "r"]:
                    continue
                if net in min_max:
                    xmin, ymin, xmax, ymax = min_max[net][0], min_max[net][1], min_max[net][2], min_max[net][3]
                    min_max[net] = [min(xmin, r), min(ymin, c), max(xmax, r), max(ymax, c)]
                else:    
                    min_max[net] = [r, c, r, c]
        area = 0
        for _, (xmin, ymin, xmax, ymax) in min_max.items():
            area += (xmax-xmin) * (ymax - ymin)
        return area
        
            
    def build_full_placement(self, all_path, edge_labels, idx, ref_placement=None):
        #add inverse direction info
        all_path_convert = []
        template = self.n2n.p_data[idx].copy()
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
            res = ["" for _ in range(row)]
            #original path
            for item in path:
                start, stop = ref[tuple(item)]
                step = 1 if stop > start else -1
                for r in range(row):
                    res[r] = res[r] + "".join(template[r][start:stop:step])
            res = [item[1:] for item in res]        
            all_path_convert.append(res)
        
        if ref_placement:
            for item in all_path_convert:
                item[2] = copy(ref_placement[0][2])
                item[3] = copy(ref_placement[0][3])
                
        return tuple(all_path_convert) 

    def build_graphs(self, path_convert):
        G = nx.Graph()
        for c in range(1, len(path_convert[0]), 2):
            for r in range(len(path_convert)):
                dev_name = "d_%s_%s"%(r, c)
                #G.add_edge(path_convert[r][c], dev_name)
                #G.add_edge(path_convert[r][c-1], dev_name)
                #G.add_edge(path_convert[r][c+1], dev_name)
                #G-left_shared
                if path_convert[r][c] != "A":
                    G.add_edge(path_convert[r][c], path_convert[r][c-1])
                    #G-right_shared
                    G.add_edge(path_convert[r][c], path_convert[r][c+1])
                    #Left-Right diffusion connect
                    G.add_edge(path_convert[r][c-1], path_convert[r][c+1])

        if False:
            pos = nx.spring_layout(G)        
            nx.draw(G, with_labels=True, pos=pos)    
            plt.show()
        return G


if __name__ == "__main__":
    fname = "../data/test_encode.csv"
    spec = "../data/test_spec.csv"
    swap = SwapRow(fname, spec)
    