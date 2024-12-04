import numpy as np
import json
import copy
import hashlib
import time
from multiprocessing import Pool

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9
ALLOWED_EDGES = [0, 1]
CANONICAL_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_OPS = CANONICAL_OPS
OPS = CANONICAL_OPS

class OutOfDomainError(Exception):
  """Indicates that the requested graph is outside of the search domain."""

def is_upper_triangular(matrix: np.ndarray):
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False

    return True

def hash_module(matrix, labeling):
    """Computes a graph-invariance MD5 hash of the matrix and label pair.

    Args:
        matrix: np.ndarray square upper-triangular adjacency matrix.
        labeling: list of int labels of length equal to both dimensions of
        matrix.

    Returns:
        MD5 hash of the matrix and labeling.
    """
    vertices = np.shape(matrix)[0]
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()

    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5(
                (''.join(sorted(in_neighbors)) + '|' +
                ''.join(sorted(out_neighbors)) + '|' +
                hashes[v]).encode('utf-8')).hexdigest())
        hashes = new_hashes
    fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

    return fingerprint

def process_item(item):
    arch, acc = item
    arch = json.loads(arch)
    mat = np.array(arch['adj_mat'])
    ops = arch['node_list']
    cell = ModelSpec(mat, ops)
    return (cell.module_hash, [cell, acc])

class ModelSpec(object):
    # Model specification given adjacency matrix and labeling.
    def __init__(self, matrix: np.ndarray, ops: list[str]) -> None:
        shape = np.shape(matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('A square matrix is required')
        if len(ops) != shape[0]:
            raise ValueError('length of ops must match matrix dimensions')
        if not is_upper_triangular(matrix):
            raise ValueError('matrix must be upper triangular')
        self.original_matrix = matrix
        self.original_ops = ops

        self.matrix = copy.deepcopy(self.original_matrix)
        self.ops = copy.deepcopy(self.original_ops)
        self.valid_spec = True
        self.prune()
        if self.is_valid():
            self.module_hash = self.hash_spec()
        else:
            raise OutOfDomainError

    def prune(self):
        # Prune the extraneous parts of the graph.
        # General procedure:
        #   1) Remove parts of graph not connected to input.
        #   2) Remove parts of graph not connected to output.
        #   3) Reorder the vertices so that they are consecutive after steps 1 and 2.
        num_vertices = np.shape(self.original_matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if self.original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if self.original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            self.matrix = None
            self.ops = None
            self.valid_spec = False
            return

        self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
        self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del self.ops[index]

    def is_valid(self):
        try:
            self._check_spec()
        except OutOfDomainError:
            return False
        return True

    def _check_spec(self):
        if not self.valid_spec:
            raise OutOfDomainError('invalid spec, provided graph is disconnected.')
        
        num_vertices = len(self.ops)
        num_edges = np.sum(self.matrix)

        if num_vertices > 7:
            raise OutOfDomainError('too many vertices')
        if num_edges > 9:
            raise OutOfDomainError('too many edges')
        if self.ops[0] != 'input':
            raise OutOfDomainError('first operation should be \'input\'')
        if self.ops[-1] != 'output':
            raise OutOfDomainError('last operation should be \'output\'')
        for op in self.ops[1: -1]:
            if op not in CANONICAL_OPS:
                raise OutOfDomainError('unsupported op')

    def hash_spec(self):
        """Computes the isomorphism-invariant graph hash of this spec.

        Args:
        canonical_ops: list of operations in the canonical ordering which they
            were assigned (i.e. the order provided in the config['available_ops']).

        Returns:
        MD5 hash of this spec which can be used to query the dataset.
        """
        # Invert the operations back to integer label indices used in graph gen.
        labeling = [-1] + [CANONICAL_OPS.index(op) for op in self.ops[1:-1]] + [-2]
        return hash_module(self.matrix, labeling)
    
    def encoding(self, cutoff=40):
        num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
        path_indices = self.get_path_indices()
        encoding = np.zeros(num_paths)
        for index in path_indices:
            encoding[index] = 1
        return encoding[:cutoff]
    
    def get_path_indices(self):
        paths = self.get_paths()
        mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
        path_indices = []

        for path in paths:
            index = 0
            for i in range(NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(OPS) ** i * (mapping[path[i]] + 1)

        path_indices.sort()
        return tuple(path_indices)
    
    def get_paths(self):
        """ 
        return all paths from input to output
        """
        paths = []
        for j in range(0, len(self.ops)):
            paths.append([[]]) if self.matrix[0][j] else paths.append([])
        
        # create paths sequentially
        for i in range(1, len(self.ops) - 1):
            for j in range(1, len(self.ops)):
                if self.matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, self.ops[i]])
        return paths[-1]



class NasBench(object):
    def __init__(self, dataset_file) -> None:
        self.dataset_file = dataset_file

        print('Loading dataset from file... This may take a few minutes...')
        start = time.time()

        # Stores the fixed statistics that are independent of evaluation (i.e.,
        # adjacency matrix, operations, and number of parameters).
        # hash --> val acc
        self.fixed_statistics = {}

        with open(self.dataset_file, 'r') as infile:
            raw_data_dict = json.loads(infile.read())
        size = len(raw_data_dict)
        
        with Pool() as pool:
            results = pool.map(process_item, raw_data_dict.items())

        self.fixed_statistics = dict(results)
        self.all_hash = self.fixed_statistics.keys()

        elapsed = time.time() - start
        print(f'Loaded dataset of size {size} in {int(elapsed)} seconds')

    def query(self, model_spec):
        # query the val acc from cell
        return self.fixed_statistics[model_spec.module_hash]

    def is_valid(self, model_spec):
        if model_spec.module_hash in self.all_hash:
            return True
        else:
            return False
        
    def generate_random_dataset(self, num):
        hashs = np.random.choice(list(self.all_hash), num, replace=False)
        all_cells = []
        all_features = []
        all_accs = []
        for hash_value in hashs:
            cell = self.fixed_statistics[hash_value][0]
            all_cells.append(cell)
            all_features.append(cell.encoding())
            all_accs.append(self.fixed_statistics[hash_value][1])
        return all_cells, np.array(all_features), np.array(all_accs)
    
    def random_spec(self):
        """Returns a random valid spec."""
        while True:
            matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            try:
                spec = ModelSpec(matrix=matrix, ops=ops)
                if self.is_valid(spec):
                    return spec
            except:
                continue




