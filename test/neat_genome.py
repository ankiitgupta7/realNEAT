# neat_genome.py
import random
import numpy as np
from collections import namedtuple

InnovationRecord = namedtuple("InnovationRecord", ["in_node", "out_node", "innovation_number"])

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled, innovation_num):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_num = innovation_num

    def mutate_weight(self):
        """Mutate the weight slightly or reassign completely."""
        if random.uniform(0, 1) < 0.8:
            self.weight += np.random.normal(0, 0.1)
        else:
            self.weight = np.random.uniform(-1, 1)

class NodeGene:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type  # 'input', 'hidden', 'output'

class Genome:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nodes = {}
        self.connections = []
        self.innovation_history = []
        self.next_node_id = num_inputs + num_outputs

        # Initialize input and output nodes
        for i in range(num_inputs):
            self.nodes[i] = NodeGene(i, 'input')
        for i in range(num_outputs):
            self.nodes[num_inputs + i] = NodeGene(num_inputs + i, 'output')

        # Fully connect input and output layers
        self._fully_connect()

    def _fully_connect(self):
        for i in range(self.num_inputs):
            for j in range(self.num_inputs, self.num_inputs + self.num_outputs):
                self.add_connection(i, j, np.random.uniform(-1, 1))

    def add_connection(self, in_node, out_node, weight):
        # Check duplicate
        for c in self.connections:
            if c.in_node == in_node and c.out_node == out_node:
                return
        innovation_num = len(self.innovation_history) + 1
        self.innovation_history.append(InnovationRecord(in_node, out_node, innovation_num))
        self.connections.append(ConnectionGene(in_node, out_node, weight, True, innovation_num))

    def mutate_weights(self):
        for c in self.connections:
            c.mutate_weight()

    def add_random_connection(self):
        possible_nodes = list(self.nodes.keys())
        in_node, out_node = random.sample(possible_nodes, 2)
        # no duplicates
        for c in self.connections:
            if c.in_node == in_node and c.out_node == out_node:
                return
        self.add_connection(in_node, out_node, np.random.uniform(-1, 1))

    def add_random_node(self):
        if not self.connections:
            return
        conn = random.choice(self.connections)
        if not conn.enabled:
            return
        conn.enabled = False
        new_node = self.next_node_id
        self.next_node_id += 1
        self.nodes[new_node] = NodeGene(new_node, 'hidden')
        # connect in_node -> new_node
        self.add_connection(conn.in_node, new_node, 1.0)
        # connect new_node -> out_node
        self.add_connection(new_node, conn.out_node, conn.weight)

    def mutate(self):
        """Randomly apply weight, connection, or node mutation."""
        # Weight mutation
        if random.uniform(0, 1) < 0.8:
            self.mutate_weights()
        # Add connection
        if random.uniform(0, 1) < 0.1:
            self.add_random_connection()
        # Add node
        if random.uniform(0, 1) < 0.05:
            self.add_random_node()
