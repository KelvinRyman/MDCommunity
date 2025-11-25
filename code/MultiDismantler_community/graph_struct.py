# graph_struct.py

# Lightweight adjacency container with dynamic resizing.
class LinkedTable:
    def __init__(self):
        self.n = 0
        self.ncap = 0
        self.head = [[]]

    def add_entry(self, head_id, content):
        """Add an entry to the list indexed by head_id."""
        if head_id >= self.n:
            if head_id + 1 > self.ncap:
                self.ncap = max(self.ncap * 2, head_id + 1)
            if head_id + 1 > len(self.head):
                self.head.extend([[] for _ in range(head_id + 1 - len(self.head))])
            self.n = head_id + 1
        self.head[head_id].append(content)

    def resize(self, new_n):
        """Resize the table to at least new_n slots."""
        if new_n > self.ncap:
            self.ncap = max(self.ncap * 2, new_n)
        if new_n > len(self.head):
            self.head.extend([[] for _ in range(new_n - len(self.head))])
        self.n = new_n
        for entry in self.head:
            if entry is not None:
                entry.clear()


# Simple graph container used by batch prep.
class GraphStruct:
    def __init__(self):
        self.out_edges = LinkedTable()
        self.in_edges = LinkedTable()
        self.subgraph = LinkedTable()
        self.edge_list = []
        self.num_nodes = 0
        self.num_edges = 0
        self.num_subgraph = 0

    def add_edge(self, idx, x, y):
        """Add a directed edge x->y with index idx."""
        self.out_edges.add_entry(x, (idx, y))
        self.in_edges.add_entry(y, (idx, x))
        self.num_edges += 1
        self.edge_list.append((x, y))
        assert self.num_edges == len(self.edge_list)
        assert self.num_edges - 1 == idx

    def resize(self, num_subgraph, num_nodes=0):
        self.num_nodes = num_nodes
        self.num_edges = 0
        self.edge_list = []
        self.num_subgraph = num_subgraph
        self.in_edges.resize(self.num_nodes)
        self.out_edges.resize(self.num_nodes)
        self.subgraph.resize(self.num_subgraph)

    def add_node(self, subg_id, n_idx):
        self.subgraph.add_entry(subg_id, n_idx)
