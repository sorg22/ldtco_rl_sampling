from collections import defaultdict
from copy import deepcopy
import random
from re import I


edge_label = {('A', 'r', 0): (0, 11, 'A', 'r'), ('A', 'r', 1): (27, 28, 'r', 'A'),
              ('A', 'r', 2): (28, 31, 'A', 'r'), ('A', 'r', 3): (31, 34, 'r', 'A'),
              ('A', 'r', 4): (44, 47, 'A', 'r'), ('A', 'r', 5): (47, 50, 'r', 'A'),
              ('A', 'A', 0): (34, 44, 'A', 'A'), ('r', 'r', 0): (11, 19, 'r', 'r'),
              ('r', 'r', 1): (19, 27, 'r', 'r')}

edge_label_1 = {('A', 'r', 0): (0, 11, 'A', 'r'), ('A', 'r', 1): (27, 28, 'r', 'A'),
              ('A', 'r', 2): (28, 31, 'A', 'r'), ('A', 'm', 3): (31, 34, 'r', 'A'),
              ('m', 'r', 4): (44, 47, 'A', 'r')}

edge_label_2 = {('A', 'r', 0): (0, 11, 'A', 'r'), ('A', 'r', 1): (27, 28, 'r', 'A'),
              ('A', 'r', 2): (28, 31, 'A', 'r'), ('A', 'r', 3): (31, 34, 'r', 'A'),
              ('k', 'r', 0): (44, 47, 'A', 'r'), ('k', 'r', 1): (47, 50, 'r', 'A'),
              ('k', 'r', 2): (44, 47, 'A', 'r'), ('k', 'r', 3): (47, 50, 'r', 'A'),
              ('A', 'k', 0): (19, 27, 'r', 'r'), ('A', 'k', 1): (19, 27, 'r', 'r')}

def get_substring_indices(str1, str2):
    indices = []
    try:
        index = str1.index(str2)
        while index != -1:
            indices.append(index)
            index = str1.index(str2, index + 1)
    except ValueError:
        pass
    return indices              
              
              
def _gen_key(edge_label):
    k_edge = defaultdict(int)
    k_set = set()
    depth = len(edge_label)
    for k in edge_label:
        if k[0] == k[1]:
            continue
        k_edge[k[:2]] += 1
        if k[0] != k[1]:
            k_edge[k[1::-1]] += 1
    for k, v in k_edge.items():
        k_set.add((k, v))
    k_map = {}
    k_adj = defaultdict(set)
    for (s, t), val in k_set:
        k_map[(s, t)] = val
        k_adj[s].add(t)
        k_adj[t].add(s)
        
    return k_set, depth, k_map, k_adj    
        
class Euler_Trie():
    def __init__(self, node=None):
        self.node = node
        self.children = defaultdict(Euler_Trie)

def _build_trie(k_map, k_adj):
    start = "A"
    end = "A"
    
    def dfs(parent, km, ka):
        if not ka[parent.node]:
            if sum(km.values()) == 0 and parent.node == end:
                return parent #proper ending: consumed all edges and finish with end node
            return None #Fail to reach end
        child_choice = list(ka[parent.node])
        random.shuffle(child_choice)
        #for child in ka[parent.node]:
        for child in child_choice:
            v = km[(parent.node, child)] 
            child_km = deepcopy(km)
            child_ka = deepcopy(ka)
            v -= 1
            if v == 0:
                child_km.pop((parent.node, child))
                child_km.pop((child, parent.node))
                child_ka[parent.node].remove(child)
                child_ka[child].remove(parent.node)
            else:
                child_km[(parent.node, child)] = v
                child_km[(child, parent.node)] = v
            t = dfs(Euler_Trie(child), child_km, child_ka)
            if t:
                parent.children[child] = t
        return parent

    start_node = Euler_Trie(start)
    return dfs(start_node, k_map, k_adj)

def get_all_path(trie):
    all_path = []
    path = [trie.node]
    
    def rec(root):
        if not root.children:
            all_path.append(path[:])
        for child, node in root.children.items():
            path.append(child)
            rec(node)
            path.pop()
    
    rec(trie)
    return all_path

def _get_random_pattern(trie, pattern):
    pattern = "".join(pattern)
    all_path = []
    path = [trie.node]
    all_pattern = []
    found_pattern = False
    
    def rec(root):
        nonlocal found_pattern
        if not root.children:
            all_idx = get_substring_indices("".join(path), pattern)
            if all_idx:
                all_path.append(path[:])
                all_pattern.append(all_idx)
                found_pattern = True

        if found_pattern:
            return         
        for child, node in root.children.items():
            path.append(child)
            rec(node)
            path.pop()
    
    rec(trie)
    return all_path, all_pattern

def _gen_full_path(edge_label):
    full_path = sorted([list(v)+list(k) for k, v in edge_label.items()])
    full_nodes = [item[2] for item in full_path] + [full_path[-1][3]]
    is_good_sample = False
    while not is_good_sample:
        idx = sorted(random.sample(range(len(full_path)), 2))
        start, end = idx
        if end - start < len(full_path) - 1 and end - start > 2:
            is_good_sample = True
         
    sample_path = full_path[idx[0]:idx[1]+1]
    non_sample_path = full_path[0:idx[0]] + full_path[idx[1]+1:]
    sample_node = [item for item in sample_path if item[2] != item[3]]
    if sample_node:
        sample_node = [item[2] for item in sample_node] + [sample_node[-1][3]]
    return full_path, full_nodes, non_sample_path, sample_path, sample_node
def divide_self_nonself_edge(path):
    non_self_edges = defaultdict(list)
    self_edges = defaultdict(list)
    for item in path: 
        if item[4] == item[5]:
            self_edges[item[4]].append(item)
        else:
            non_self_edges[tuple(sorted([item[4], item[5]]))].append(item)
    for k in self_edges.keys():
        random.shuffle(self_edges[k])
    for k in non_self_edges.keys():
        random.shuffle(non_self_edges[k])
    return self_edges, non_self_edges
            
def get_explicit_path(item):
    path = []
    prev = "A"
    test = set()
    for it in item:
        start = prev
        end = it[3] if prev == it[2] else it[2] 
        path.append([start, end, it[6]])
        test.add(tuple((sorted([start, end])+ [it[6]])))
        prev = end
    if len(test) != len(item):
        print("WONG !!!!!!!!!!!!!!!")    
    return path
            
def get_pattern_from_edge_label(edge_label):    
    full_path, full_nodes, non_sample_path, sample_path, sample_node = _gen_full_path(edge_label)
    u_key, depth, k_map, k_adj = _gen_key(edge_label)
    root = _build_trie(k_map, k_adj)

    path, pattern = _get_random_pattern(root, sample_node)        
    if sample_node:
        idx_start = random.choice(pattern[0])
        idx_end = idx_start + len(sample_node) - 1
    else:
        c = sample_path[0][2]
        idx_start = random.choice([n for n, item in enumerate(path[0]) if item == c])
        idx_end = idx_start
    mutation_head = list(range(0,idx_start))
    mutation_tail = list(range(idx_end + 1, len(path[0])))
    self_node_candidate = defaultdict(list)
    for item in mutation_head + mutation_tail + [idx_start, idx_end]:
        self_node_candidate[path[0][item]].append(item)
    #reconstruct edge_label_mutation using non_sample_path and sample_path
    #update patter)n edge_label --> sample_path
    #allocate non_self_node edge head and tail
    self_edges, non_self_edges = divide_self_nonself_edge(non_sample_path) 
    non_sample_head, non_sample_tail = [], []
    if sample_node and non_self_edges:
        while mutation_head:
            idx = mutation_head.pop(0)
            non_sample_head.append(non_self_edges[tuple(sorted([path[0][idx], path[0][idx+1]]))].pop(0))
        while mutation_tail:
            idx = mutation_tail.pop(0)
            non_sample_tail.append(non_self_edges[tuple(sorted([path[0][idx-1], path[0][idx]]))].pop(0))
    else:
        #When sample node is empty. It means selected part only have self edge
        non_sample_head = non_sample_path
    #allocate self_node
    self_edge_with_new_idx = defaultdict(list)
    for k, v in self_edges.items():
        while v:
            e = v.pop(0)
            l = random.choice(self_node_candidate[k])
            self_edge_with_new_idx[l].append(e)
            
        
    #insert self edges in head and tail
    head_l = len(non_sample_head)
    tail_l = len(non_sample_tail)
    for v in sorted(self_edge_with_new_idx.keys(), reverse=True):
        if v <= idx_start:
            non_sample_head = non_sample_head[0:v] + self_edge_with_new_idx[v] + non_sample_head[v:]
        else:
            non_sample_tail = non_sample_tail[0:v-idx_end] + self_edge_with_new_idx[v] + non_sample_tail[v-idx_end:]
            #for n in range(len(self_edge_with_new_idx[v])):
            #    non_sample_tail.insert(self_edge_with_new_idx[v][n], v - idx_end)
            
    all_path = non_sample_head + sample_path + non_sample_tail
    #Sanity check
    #get_explicit_path(all_path)
    return all_path

def __get_pattern_from_edge_label(edge_label):    
    u_key, depth, k_map, k_adj = _gen_key(edge_label)
    root = _build_trie(k_map, k_adj)
    path, pattern = _get_random_pattern(root, ['r', 'A'])        
    return path, pattern   


if __name__ == "__main__":
    path, pattern = get_pattern_from_edge_label(edge_label)    
    print(path, pattern)
    if 0:
        u_key, depth, k_map, k_adj = _gen_key(edge_label)
        print(u_key)
        print(k_map)
        print(k_adj)
        root = _build_trie(k_map, k_adj)
        path = get_all_path(root)
        a = set()
        print("path len: ", len(path))
        for p in path:
            if tuple(p) in a:
                print("dup: ", p)
            else:
                a.add(tuple(p))
                print(p)
        #path, pattern = _get_random_pattern(root, ['r', 'k', 'A', 'r', 'k'])        
        path, pattern = _get_random_pattern(root, ['r', 'A'])        
        a = set()
        print("path len: ", len(path))
        for p0, p1 in zip(path, pattern):
            if tuple(p0) in a:
                print("dup: ", p0)
            else:
                a.add(tuple(p0))
                print(p0)         
                print(p1)

        
    
