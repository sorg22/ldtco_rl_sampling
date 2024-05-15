from tkinter import W
import numpy
import random
from collections import defaultdict
from itertools import product
import copy

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

class GraphMutation():
    def __init__(self, graph):
        self.graph = graph
        self.n = len(self.graph)
        self.generic_solution = {}
        #self.mutate()
            
    def mutate_sample(self, min_length=0):
        if min_length == 0:
            min_length = random.choice(range(2, max(3, self.n-3)))
        print("min_length: ", min_length)    
        if min_length >= self.n: return None
        
        start = random.choice(range(self.n - 1 - min_length))
        end = random.choice(range(start + min_length -1, self.n))
        mut = self.mutate(start, end)
        return mut

    def mutate_all(self, min_length=2):
        all_specific_solutions = set()
        for l in range(min_length, self.n-3):
            for s in range(0, self.n - l):
                e = s + l
                #print("s:%s e:%s "%(s, e), self.mutate(s, e))
                all_specific_solutions.update(set(self.mutate(s, e)))
        return all_specific_solutions
        
        
    
    def mutate(self, start, end):
        mutated_graphs = []
        unique_specific_edge = {}
        
        left = copy.deepcopy(self.graph[0:start])
        right = copy.deepcopy(self.graph[end+1:])
        print("\n", start, end)
        #print(left, self.graph[start: end+1], right)
        #print("left: ", left)
        left_all = self.gen_all_sub_graphs(left)
        #print("left_all: ", left_all)
        #print("left: ", right)
        right_all = self.gen_all_sub_graphs(right)
        #print("right_all: ", right_all)
        if not (left_all or right_all):
            return [self.graph]
        if left_all:
            left_all = copy.deepcopy([item + self.graph[start:end+1] for item in left_all])
        else:
            left_all = copy.deepcopy([self.graph[start:end+1]])
        if right_all:
            for c, r in product(left_all, right_all):
                mutated_graphs.append(c+r) 
        else:
            mutated_graphs = left_all        
        return to_tuple(mutated_graphs)
            
        
    def gen_all_sub_graphs(self, g):
        #empty graph
        if not g: return g
        print("start graph g: ", g)
        sub_graph_n = len(g)
        #get start and end of sub graph
        start = g[0][0]
        end = g[-1][-2]
        print("start: ", start, "end: ", end)

        #map specific edges to generic edge
        label_id = defaultdict(int)
        generic_to_specific = {}
        specific_generic = []
        for item in sorted(g):
            generic_key = sorted(item[0:2])
            generic_edge = [generic_key[0], generic_key[1], label_id[tuple(generic_key)]]
            label_id[tuple(generic_key)] += 1
            specific_generic.append((tuple(item), tuple(generic_edge)))
            generic_to_specific[tuple(generic_edge)] = item
            
        #print("specific_generic: ", specific_generic)

        generic_edges = tuple(item for _, item in specific_generic)
        
        #build graph
        edge_map = defaultdict(list)
        for edge in generic_edges:
            if edge[0] == edge[1]:
                edge_map[edge[0]].append(tuple(edge))
            else:
                edge_map[edge[0]].append(tuple(edge))
                edge_map[edge[1]].append(tuple(edge))
                
        
        #recoreds all possible sub_g
        all_sub_g = []
        path = []
        #back tracking dfs
        def dfs(node, path):      
            #No more edge available in node  
            if not (set(edge_map[node]) - set(path)):
                #success when all the edges are used and end point matches
                if len(path) == sub_graph_n and node == end:
                    all_sub_g.append(path[:])
                else:
                    return        
                    
            for edge in edge_map[node]:
                if len(path) == 0 and node != start:
                    continue
                if edge in path:
                    continue
                path.append(edge)
                if node != edge[0]:
                    next_node = edge[0]
                else:
                    next_node = edge[1]    
                dfs(next_node, path)
                path.pop()
        if generic_edges in self.generic_solution:
            print("Found Cached solution")
            all_sub_g = self.generic_solution[generic_edges]
        else:
            print("try new solution: ") 
            dfs(start, [])
            self.generic_solution[generic_edges] = all_sub_g[:]
        if all_sub_g:
            #return after change to specific edges
            all_real_path = []
            for sub_g in all_sub_g:
                prev = start
                real_sub_g = []
                for item in sub_g:
                    next = item[0] if prev not in item[0] else item[1]
                    s_edge = generic_to_specific[item] 
                    if s_edge[1] == prev:
                        s_edge[0], s_edge[1] = s_edge[1], s_edge[0]

                    real_sub_g.append(s_edge[:])
                    prev = next
                    
                #print(real_sub_g, sub_g) 
                all_real_path.append(real_sub_g[:])
            #print("all_real_path: ", all_real_path)
            #generic_to_specific

            return all_real_path
        else:
            return []

    #def gen_db(self, )




if __name__ == "__main__":
    graph = [['A', 'r', 1], ['r', 'A', 2], ['A', 'r', 5], ['r', 'A', 4],
            ['A', 'r', 3], ['r', 'r', 1], ['r', 'r', 0], ['r', 'A', 0], ['A', 'A', 0]] 

    mg = GraphMutation(graph)
    ans = mg.mutate_all(3)
    for item in ans:
        print (item)
        
        
    for i in range(0):
        print("i: ", i)
        muts = mg.mutate_sample()
        if len(muts) > 1:
            for item in muts:
                print(item)