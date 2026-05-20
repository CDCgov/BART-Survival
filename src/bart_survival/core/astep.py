import numpy as np
from pymc_bart.pgbart import PGBART

class FastSurvPGBART(PGBART):
    """
    Inherits all the PyTensor graph administration from PyMC-BART, 
    but hijacks the tree storage and mutation step for C-speeds.
    """
    def __init__(self, vars, *args, **kwargs):
        # 1. Let the core PGBART do all its standard setup (parsing X, Y, m, etc.)
        super().__init__(vars, *args, **kwargs)
        
        # 2. THE HANDOFF INIT: Allocate the C-Memory once.
        # Now that super() has extracted self.m and self.Y, we can use them!
        self.c_tree_state = FastTreeState(m_trees=self.m, init_mean=self.Y.mean())
        
        # (Optional) Extract the deterministic baseline offset here if passing to Numba
        self.numba_X = self.X 
        self.numba_Y = self.Y

    def astep(self, q0):
        """
        This completely overwrites PyMC's native Particle Gibbs loop.
        """
        # 1. Grab the raw NumPy arrays from our FastTreeState
        F = self.c_tree_state.F
        T = self.c_tree_state.T
        L = self.c_tree_state.L
        R = self.c_tree_state.R
        V = self.c_tree_state.V
        node_counts = self.c_tree_state.node_counts

        # 2. THE NUMBA HANDOFF: Pass the arrays into the C-compiled Proposer
        # (This is where the actual math happens instantly)
        acceptance_flag = propose_and_evaluate_mutations(
            self.numba_X, self.numba_Y, 
            F, T, L, R, V, node_counts
        )

        # 3. Handle the trace storage (simplified)
        # You just return the state so PyMC can record the draw.
        stats = {"tune": self.tune}
        
        # For simplicity, returning the V array (leaf values) as the state representation
        return V, [stats]

class FastTreeState:
    """
    A pure NumPy container holding the state of the entire BART ensemble.
    This object is created once in Python, and its raw arrays are handed to Numba.
    """
    def __init__(self, m_trees, init_mean, max_nodes=255):
        self.m_trees = m_trees
        self.max_nodes = max_nodes
        
        # 1. Allocate the memory blocks
        # Initialize F, L, R with -1 (meaning "Leaf / No Child")
        self.F = np.full((m_trees, max_nodes), -1, dtype=np.int16)
        self.L = np.full((m_trees, max_nodes), -1, dtype=np.int16)
        self.R = np.full((m_trees, max_nodes), -1, dtype=np.int16)
        
        # Initialize T and V with 0.0
        self.T = np.zeros((m_trees, max_nodes), dtype=np.float32)
        self.V = np.zeros((m_trees, max_nodes), dtype=np.float32)
        
        # 2. Trackers
        # How many nodes are currently active in each tree? 
        # (Starts at 1 because the root node is index 0)
        self.node_counts = np.ones(m_trees, dtype=np.int16)
        
        # 3. Apply PyMC's Initial Assumptions
        # At step 0, PyMC assumes every tree is a single leaf (the root) 
        # predicting exactly (init_mean / m_trees).
        initial_leaf_value = init_mean / m_trees
        
        for m in range(m_trees):
            # Node 0 is the root. It starts as a leaf (-1 feature, no children)
            self.V[m, 0] = initial_leaf_value

# F (Feature): int16. The column index of $X$ used for the split. A value of -1 indicates the node is a leaf.
# T (Threshold): float32. The continuous value the feature is tested against.
# L Left Child): int16. The array index of the left child node. -1 if leaf.
# R (Right Child): int16. The array index of the right child node. -1 if leaf.
# V (Value): float32. The hazard contribution (residual prediction) if the node is a leaf.
# N (Node Count): int16 1D array of shape (m_trees,). Tracks the next available blank node index for each tree so we know where to append mutations.