from dataset import *

dataset = NasBench('./nasbench_dataset')

def get_spec(adj_indxs, op_indxs):
    """
    Construct a NASBench spec from adjacency matrix and op indicators
    """
    op_names = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    ops = ['input']
    ops.extend([op_names[i] for i in op_indxs])
    ops.append('output')
    iu = np.triu_indices(7, k=1)
    adj_matrix = np.zeros((7, 7), dtype=np.int32)
    adj_matrix[(iu[0][adj_indxs], iu[1][adj_indxs])] = 1
    try:
        spec = ModelSpec(adj_matrix, ops)
        if dataset.is_valid(spec):
            return spec
    except:
        return None


def evaluate_x(x, model):
    """
    Evaluate NASBench on the model defined by x.

    x is a 36-d array.
    The first 21 are for the adjacency matrix. Largest entries will have the
    corresponding element in the adjacency matrix set to 1, with as many 1s as
    possible within the NASBench model space.
    The last 15 are for the ops in each of the five NASBench model components.
    One-hot encoded for each of the 5 components, 3 options.
    """
    assert len(x) == 36
    x_adj = x[:21]
    x_op = x[-15:]
    x_ord = x_adj.argsort()[::-1]
    op_indxs = x_op.reshape(3, 5).argmax(axis=0).tolist()
    last_good = None
    for i in range(1, 22):
        model_spec = get_spec(x_ord[:i], op_indxs)
        if model_spec is not None:
            last_good = model_spec
            break
    if last_good is None:
        # Could not get a valid spec from this x. Return bad metric values.
        return 0.80
    acc = model.predict(np.array(last_good.encoding()).reshape(1,-1,1)).reshape(1)
    return acc


class NASBenchRunner:
    """
    A runner for non-Ax methods.
    Assumes method MINIMIZES.
    """
    def __init__(self, max_eval, model):
        # For tracking iterations
        self.fs = []
        self.n_eval = 0
        self.max_eval = max_eval
        self.model = model

    def f(self, x):
        if self.n_eval >= self.max_eval:
            raise ValueError("Evaluation budget exhuasted")
        test_acc = evaluate_x(x, self.model)
        self.fs.append(test_acc)  # Store the true, not-negated value
        return test_acc  # ASSUMES METHOD MINIMIZES
