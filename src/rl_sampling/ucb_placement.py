import graphrp.main as main
from .input_parser import Net2Node
from collections import defaultdict
import numpy as np
import math
from .euler_trie import get_pattern_from_edge_label

class UCBPL():
    def __init__(self, fname, spec, model_file, sample_num=1000, res_file="res.dat"):
        self.prouter = main.PRouter()
        self.prouter.load_model(model_file)
        
        self.n2n = Net2Node(fname, spec)
        self.bank, self.unique_idx, self.ks_2_idx = self.n2n.build_all_edge_bank()
        self.routable = {}
        self.mutation_bank = {}
        print(self.bank)
        print(self.unique_idx)
        print(self.ks_2_idx)
        self.sample_by_UCB(num=sample_num)

        if 0:
            self.all_placement = self.generate_placement()
            self.all_placement = list(self.all_placement)
            self.all_data = self.n2n.process_data_from_code(self.all_placement)
            self.run_model(res_file)
    
    def _get_mutation_idx(self, children, num_sample=100, max_trial = 200):
        mutations = set()
        trial = 0
        overlap = 0
        while len(mutations) < num_sample and trial < max_trial:
            for child in children:
                trial += 1
                #print(self.n2n.reduced_all_data[child])
                G, edge_list, edge_labels = self.n2n.build_graph(child)
                pattern = tuple(tuple(item) for item in get_pattern_from_edge_label(edge_labels))
                if pattern in mutations:
                    overlap += 1
                    #print("overlap: -------------------------", overlap, trial, trial -overlap)
                else:
                    mutations.add(pattern)
                    #full placement 
                    
                #Generate M mutation per idx
        return list(mutations)    

    def get_explicit_path(self, item):
        path = []
        prev = "A"
        for it in item:
            start = prev
            end = it[3] if prev == it[2] else it[2] 
            path.append([start, end, it[6]])
            prev = end
        return path

    def get_mutation_idx(self, children, num_sample=100, max_trial = 200):
        mutations = defaultdict(set)
        all_placement= set()
        for child in children:
            trial = 0
            overlap = 0
            #print(self.n2n.reduced_all_data[child])
            G, edge_list, edge_labels = self.n2n.build_graph(child)
            while len(mutations[child]) < num_sample and trial < max_trial: 
                trial += 1
                item = get_pattern_from_edge_label(edge_labels)
                #convert edge_labels list to explicit path list
                item = self.get_explicit_path(item)
                pattern = tuple(tuple(it) for it in item)
                if pattern in mutations[child]:
                    overlap += 1
                    #print("overlap: -------------------------", overlap, trial, trial -overlap)
                else:
                    mutations[child].add(pattern)
                    #full placement 
            all_full_placement = list(self.n2n.build_full_placement(mutations[child], edge_labels, idx=child))   
            for placement in all_full_placement:
                if placement not in all_placement:
                    all_placement.add(placement)
                else:
                    print("overlap in merge")
            
            

        return all_placement

    def get_mutation(self, idx_routable, parent_child):
        idx_mutation = {}
        #print(idx_routable)
        #print(self.n2n.reduced_all_data)
        #print(parent_child)
        idx_routable_child = {}
        for vals in self.ks_2_idx.values():
            idx_routable_child[vals[0]] = [item for n, item in enumerate(vals) if n in idx_routable[vals[0]]]
        #print(idx_routable_child)
        for idx, children in idx_routable_child.items():
            idx_mutation[idx] = self.get_mutation_idx(children)
        return idx_mutation    
        
        

    def sample_by_UCB(self, num, eps=0.5, ucb_const=0.25, run_model_all=False):
        num_actions = len(self.unique_idx)

        def uncertain(t, na, e=1e-10):
            #return [np.sqrt(2*np.log(t)/(na[n]+1e-10)) for n in range(num_actions)]    
            return np.sqrt(2*np.log(t+1)/(na+1e-10))
                
        idx_initial_placement = defaultdict(set)
        idx_initial_placement_p = []
        idx_ucb_generate_placement_p = []
        parent_child = {}
        for vals in self.ks_2_idx.values():
            first_sample = vals[0]
            parent_child[first_sample] = vals[1:]
            for v in vals: 
                idx_initial_placement[first_sample].add(tuple(item[1:-1] for item in self.n2n.p_data[v]))
        #Initial placement P1 calculation
        
        #Run default placement
        Na = np.empty(num_actions)
        Q = np.empty(num_actions)
        Q_prev = np.empty(num_actions)
        Routable = np.empty(num_actions)
        Na.fill(0)
        Q.fill(0)
        Q_prev.fill(0)
        Routable.fill(0)
                
        idx_routable = {}
        #run_model for initial placement
        for n, idx in enumerate(self.unique_idx):
            res = self.run_model(idx_initial_placement[idx])
            Q[n] = res.mean()
            Na[n] = res.size
            Routable[n] = np.where(res > 0.5)[0].size
            idx_routable[idx] = np.where(res > 0.5)[0]
            for k, v in zip(idx_initial_placement[idx], res):
                idx_initial_placement_p.append((k, v))
        #randomly generated placements
        print(idx_routable)
        all_initial_routable_mutation = self.get_mutation(idx_routable, parent_child)
        all_placement = self.n2n.find_idx_placement(self.unique_idx, 4000)   

        #remove all_placement from all_initial_routable_mutation
        for idx in all_initial_routable_mutation:
            if idx in idx_initial_placement:
                all_initial_routable_mutation[idx] = list(all_initial_routable_mutation[idx] - set(idx_initial_placement[idx]))
            if idx in all_placement:
                all_placement[idx] = list(set(all_placement[idx]) - set(all_initial_routable_mutation[idx]))
            

        #decide to run model to all candidate
        run_model_all = True
        pre_model_data = dict()
        pre_model_data_mutations = dict()
        if run_model_all:
            for idx, data in all_placement.items():
                res = self.run_model(data)
                res = sorted([(item, n, math.ceil(item*1000 + 1e-40)) for n, item in enumerate(res)])
                P = [item[2] for item in res]
                P = [item/sum(P)for item in P]
                samples = np.random.choice(range(len(res)) , size=len(res),replace=False, p=P)
                sampled_res = [res[n] for n in samples]
                pre_model_data[idx] = sampled_res
            for idx, data in all_initial_routable_mutation.items():
                if data:
                    res = self.run_model(data)
                    res = sorted([(item, n, math.ceil(item*1000 + 1e-40)) for n, item in enumerate(res)])
                    P = [item[2] for item in res]
                    P = [item/sum(P)for item in P]
                    samples = np.random.choice(range(len(res)) , size=len(res),replace=False, p=P)
                    sampled_res = [res[n] for n in samples]
                    pre_model_data_mutations[idx] = sampled_res 
        Q_prev = Q.copy()
        for n in range(num):
            res = uncertain(n, Na, e=1e-10)
            idx = np.argmax(Q + ucb_const * res)
            res = 0
            use_mutation = True if np.random.random() <= 0.9 else False
            if self.unique_idx[idx] not in pre_model_data_mutations:
                print("can't find idx:%s during pre_model_data_mutations\n\n"%self.unique_idx[idx])
            if self.unique_idx[idx] in pre_model_data_mutations and use_mutation:
                res, pidx, _ = pre_model_data_mutations[self.unique_idx[idx]].pop(0)
                item = all_initial_routable_mutation[self.unique_idx[idx]][pidx]
            else:
                if pre_model_data[self.unique_idx[idx]]:
                    res, pidx, _ = pre_model_data[self.unique_idx[idx]].pop(0)
                    item = all_placement[self.unique_idx[idx]][pidx]
                else:
                    continue
            if item in idx_initial_placement[self.unique_idx[idx]]:
                print("pass due to init placement") 
                continue
            #res = self.run_model([item])
            idx_ucb_generate_placement_p.append((item, res))     
            if res > 0.5:
                print("found routable")
                Routable[idx] += 1
            Na[idx] += 1
            Q[idx], Q_prev[idx] = Q_prev[idx] + (res - Q_prev[idx]) / Na[idx], Q[idx]
        self.save_into_file(idx_initial_placement_p, "initial_prob.csv")
        self.save_into_file(idx_ucb_generate_placement_p, "samples_prob.csv")
        print(Q)
            
    def _sample_by_UCB(self, num, eps=0.5, ucb_const=0.25, run_model_all=False):
        num_actions = len(self.unique_idx)

        def uncertain(t, na, e=1e-10):
            #return [np.sqrt(2*np.log(t)/(na[n]+1e-10)) for n in range(num_actions)]    
            return np.sqrt(2*np.log(t+1)/(na+1e-10))
                
        idx_initial_placement = defaultdict(set)
        idx_initial_placement_p = []
        idx_ucb_generate_placement_p = []
        for vals in self.ks_2_idx.values():
            first_sample = vals[0]
            for v in vals: 
                idx_initial_placement[first_sample].add(tuple(item[1:-1] for item in self.n2n.p_data[v]))
        #Initial placement P1 calculation
        
        #Run default placement
        Na = np.empty(num_actions)
        Q = np.empty(num_actions)
        Q_prev = np.empty(num_actions)
        Routable = np.empty(num_actions)
        Na.fill(0)
        Q.fill(0)
        Q_prev.fill(0)
        Routable.fill(0)
                
        #run_model for initial placement
        for n, idx in enumerate(self.unique_idx):
            res = self.run_model(idx_initial_placement[idx])
            Q[n] = res.mean()
            Na[n] = res.size
            Routable[n] = np.where(res > 0.5)[0].size
            for k, v in zip(idx_initial_placement[idx], res):
                idx_initial_placement_p.append((k, v))
        #randomly generated placements
        all_placement = self.n2n.find_idx_placement(self.unique_idx, 20)   

        #decide to run model to all candidate
        run_model_all = True
        pre_model_data = dict()
        if run_model_all:
            for idx, data in all_placement.items():
                pre_model_data[idx] = self.run_model(data)
            
            
            
        Q_prev = Q.copy()
        for n in range(num):
            res = uncertain(n, Na, e=1e-10)
            idx = np.argmax(Q + ucb_const * res)
            if all_placement[self.unique_idx[idx]]:
                item = all_placement[self.unique_idx[idx]].pop(0)
            else:
                continue
            if item in idx_initial_placement[self.unique_idx[idx]]:
                print("pass due to init placement") 
                continue
            res = self.run_model([item])
            idx_ucb_generate_placement_p.append((item, res[0]))     
            if res > 0.5:
                print("found routable")
                Routable[idx] += 1
            Na[idx] += 1
            Q[idx], Q_prev[idx] = Q_prev[idx] + (res[0] - Q_prev[idx]) / Na[idx], Q[idx]
        self.save_into_file(idx_initial_placement_p, "initial_prob.csv")
        self.save_into_file(idx_ucb_generate_placement_p, "samples_prob.csv")
        print(Q)
        pass
        
        
        
    def run_model(self, placements):
        all_data = self.n2n.process_data_from_code(placements)        
        tbl = self.prouter.predict_codes(all_data)
        return np.array(tbl["p1"].values)
         
    def _run_model(self, res_file):
        #placements = self.n2n.codes_2_placement(self.all_placement)
        tbl = self.prouter.predict_codes(self.all_data)
        tbl["placement"] = ["".join(item) for item in self.all_placement]
        tbl.to_csv(res_file, index=False)
        
    def generate_placement(self):
        print("unique_idx: ", len(self.unique_idx))
        all_placement = self.n2n.find_all_placement(self.unique_idx)
        self.n2n.save_into_file(all_placement, "new_placement_pos.dat") 
        return all_placement
        
    def save_into_file(self, placements, fname):
        with open(fname, "w") as f:
            f.write("code,part_length,routability,probability\n")
            for placement, prob in placements:
                l = len(placement[0])
                for p in placement:
                    f.write("%s"%p)
                f.write(",%s,%s,%s"%(l, 1 if prob > 0.5 else 0, prob))
                f.write("\n")
    
if __name__ == "__main__":
    if 0:
        #fname = "../data/test_encode_pos.csv"
        #fname = "../data/test_encode_mix.csv"
        fname = "../data/test_encode_swap.csv"
        spec = "../data/test_spec.csv"
        model_file = "/nfs/site/disks/ad_wa_skim501/graphrp_test/golden_test1_again/model_best_doe.pt"
        ucb = UCBPL(fname, spec, model_file, 5000, "res_5000.csv")
    if 0:
        fname = "/nfs/site/disks/ad_wa_skim501/graphrp_test/golden_test_4_1_2024/toSungwon/out.table"
        spec = "/nfs/site/disks/ad_wa_skim501/graphrp_test/golden_test_4_1_2024/toSungwon/out.spec"
        model_file = "/nfs/site/disks/ad_wa_skim501/graphrp_test/golden_test_4_1_2024/model_best_doe.pt"
        ucb = UCBPL(fname, spec, model_file, 5000, "res_5000.csv")

    if 1:
        fname = "/nfs/site/disks/ad_wa_skim501/graphrp_test/golden_test_4_10_2024/toSungwon/out.table"
        spec = "/nfs/site/disks/ad_wa_skim501/graphrp_test/golden_test_4_10_2024/toSungwon/out.spec"
        model_file = "/nfs/site/disks/ad_wa_skim501/graphrp_test/golden_test_4_10_2024/model_best_doe.pt"
        ucb = UCBPL(fname, spec, model_file, 3000, "res_5000.csv")

