import numpy as np


class SumTree(object):
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data_store = np.zeros(capacity,dtype=object)
        self.data_pointer = 0
        
        
    def add(self,priority,experience):
        tree_canopy = self.capacity -1
        tree_data_index = self.data_pointer + tree_canopy
        self.data_store[self.data_pointer] = experience
        self.update_tree(tree_data_index,priority)
        self.data_pointer +=1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
            
    def update_tree(self,tree_data_index,new_priority):
        
        inital_priority = self.tree[tree_data_index]
        priority_delta = new_priority - inital_priority
        self.tree[tree_data_index] = new_priority
        
        while tree_data_index != 0:
            tree_data_index = (tree_data_index-1)//2
            self.tree[tree_data_index] += priority_delta
            
    def get_node(self,value):
        
        parent_node_index = 0
        
        while True:
            left_child_index = 2*parent_node_index + 1
            right_child_index = 2*parent_node_index + 2
            
            if left_child_index >= len(self.tree):
                leaf_node_index = parent_node_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_node_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_node_index = right_child_index
            
        data_index = leaf_node_index - self.capacity + 1
        return leaf_node_index, self.tree[leaf_node_index], data_index
    
    @property
    def total_priority(self):
        return self.tree[0]