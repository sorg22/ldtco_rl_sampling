from pathlib import Path
import re
from collections import defaultdict
import pandas as pd
from graphrp.generate_graph import GraphGen
#import rtree as rt 

diffusion_stack = [0, 1, 3, 2, 4, 5, 7, 6]
#skip_net = ["!float", "!ssb", "ssb", "vcc", "vss", "vssx"]
skip_net = ["vcc", "vss", "vssx"]


class Net2Node():
    """convert string netlist input to graph node"""
    def __init__(self, fname, fnode, spec="spec.json"):
        self.skip = self.extract_skip_net(fnode)
        self.data, self.labels = None, None
        if fname:
            self.data, self.labels = self.process_data(fname)

    def extract_skip_net(self, file_name):
        fin = Path(file_name)
        skip = []
        if not fin.is_file():
            print("input node name file is not exist")
        with open(file_name, "r") as f:    
            for line in f:
                line = line.strip().replace("'","").split()
                if line[1] in skip_net:
                    skip.append(line[0])
        return skip            

    def process_data_from_code(self, code, labels, chunk):
        all_placement = []
        all_label = []
        placement = 0
        for net, label in zip(code, labels):
        #for placement, line in enumerate(f):
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
            placement += 1
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
    fname = "test.dat"
    #n2n = Net2Node("data/test.dat", "/nfs/site/disks/ad_wa_skim501/GNN_data/test1/out.spec")
    #n2n = Net2Node("test_encode[CJprnF\[Ar[X^`_ParmM`a`NmrA^Br^_Aa_b^dcUmrAcbrbk[CI\qnEp[Aq[W_`^Oaqma`q`amqA^Bq^_Aa^b_dcqmdAcbqbk\pq\KD]pGcq]Y_e^SfqmfeqefmqAlgqghAf^g_ihqmiAjhqjo\prpLD]\Hcr]Z^e_TfrmQefeRmrAlgrghAf_g^ihVmrAjhrjo_pos.csv", "test_spec.csv")
    if 0:
        n2n = Net2Node('test_encode_pos.csv', "test_spec.csv")
        graph_gen = GraphGen(20, 6)
        all_nx_graphs = graph_gen.ascii_2_graphs(n2n.data)
        print(all_nx_graphs)
    if 1:
        n2n = Net2Node(None, "test_spec.csv")
        data, labels = n2n.process_data_from_code(['[CJprnF\[Ar[X^`_ParmM`a`NmrA^Br^_Aa_b^dcUmrAcbrbk[CI\qnEp[Aq[W_`^Oaqma`q`amqA^Bq^_Aa^b_dcqmdAcbqbk\pq\KD]pGcq]Y_e^SfqmfeqefmqAlgqghAf^g_ihqmiAjhqjo\prpLD]\Hcr]Z^e_TfrmQefeRmrAlgrghAf_g^ihVmrAjhrjo'],
                                [1,], 49)
        graph_gen = GraphGen(20, 6)
        all_nx_graphs = graph_gen.ascii_2_graphs(data)
        print(all_nx_graphs)
