from tkinter import W
import networkx as nx
from input_parser import Net2Node
import matplotlib.pyplot as plt


class GraphBuilder():
    def __init__(self, fname, spec):
        self.n2n = Net2Node(fname, spec)
        self.bank, self.unique_idx, self.ks_2_idx = self.n2n.build_all_edge_bank() 
        self.gen_new_placement()

    def build_full_placement(self, all_path, edge_labels, idx):
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
        return tuple(all_path_convert)

    def gen_new_placement(self):
        for idx in self.unique_idx:
            G, edge_list_org, edge_labels = self.n2n.build_graph(idx = idx)
            label_edges = {v:k for k, v in edge_labels.items()}
            path = []
            for k in sorted(label_edges):
                path.append([k[2], k[3], label_edges[k][2]])
            path_convert = list(self.build_full_placement([path], edge_labels, idx=idx))
            graphs = self.build_graphs(path_convert[0])
            print(path)

    def build_graphs(self, path_convert):
        G = nx.Graph()
        for c in range(1, len(path_convert[0]), 2):
            for r in range(len(path_convert)):
                dev_name = "d_%s_%s"%(r, c)
                #G.add_edge(path_convert[r][c], dev_name)
                G.add_edge(path_convert[r][c-1], dev_name)
                G.add_edge(path_convert[r][c+1], dev_name)

        if True:
            pos = nx.spring_layout(G)        
            nx.draw(G, with_labels=True, pos=pos)    
            plt.show()
                
                
    

        
if __name__ == "__main__":
    fname = "../data/test_encode.csv"
    spec = "../data/test_spec.csv"
    swap = GraphBuilder(fname, spec)       
        