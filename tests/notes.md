random testing notes:
times:
    pre-change: 5.9 s


# fast bart
The "No Objects Allowed" ArchitectureAs we discovered, PyMC-BART slows down because it constantly spawns and modifies Python Node and Tree objects inside the step loop. To make our custom _patched_astep fly, we must translate the concept of a "decision tree" into flat, pre-allocated NumPy arrays.For a model with $M$ trees and a maximum of $K$ nodes per tree, we pre-allocate a rigid 3D matrix (or a set of parallel 2D arrays) at the very beginning of the .fit() call.We define the state of the entire ensemble using five primitive arrays of shape (M, K):F (Feature Index): Integer array. Which column of $X$ is this node splitting on? (Leaf nodes = -1).T (Threshold): Float array. The continuous value the feature is split against.L (Left Child): Integer array. The index of the left child node.R (Right Child): Integer array. The index of the right child node.V (Value): Float array. The predicted hazard contribution if this node is a leaf.The Survival Engine WorkflowInside our custom astep, we intercept PyMC's sampling loop. Instead of calling particle.sample_tree(), we route the data through four purely C-compiled @njit functions:The Array Allocator: Translates PyMC's current tree states into our F, T, L, R, V arrays right before the mutations begin.The Numba Proposer: A highly aggressive function that picks a tree, finds an active leaf node (where F == -1), and randomly proposes a new split (F_new, T_new) and two new leaf values (V_left, V_right).The C-Level Likelihood: We pass the proposed array state and the deterministic $t + t^2$ baseline offset into a mathematically hardcoded Probit log-likelihood function. No PyTensor graph evaluations allowed.The Metropolis-Hastings Router: If the likelihood ratio is accepted, we keep the array updates. If rejected, we revert the arrays.Because we are doing this strictly for standard survival analysis, we can hardcode the binary target variables and the continuous split rules directly into the compiler, stripping away thousands of lines of generalized safety checks.Should we start by defining the exact array dimensions and data types for the F, T, L, R, V matrix, or would you prefer to map out the math for the Numba Proposer first?

1. The Array Allocator: Translates PyMC's current tree states into our F, T, L, R, V arrays right before the mutations begin.
2. The Numba Proposer: A highly aggressive function that picks a tree, finds an active leaf node (where F == -1), and randomly proposes a new split (F_new, T_new) and two new leaf values (V_left, V_right).
3. The C-Level Likelihood: We pass the proposed array state and the deterministic $t + t^2$ baseline offset into a mathematically hardcoded Probit log-likelihood function. 
4. No PyTensor graph evaluations allowed.The Metropolis-Hastings Router: If the likelihood ratio is accepted, we keep the array updates. If rejected, we revert the arrays.


You identify Tree $m$ to mutate.
You propose a "Grow" mutation, which instantly alters the F, L, R arrays.
You run get_tree_prediction on that newly mutated array state.
You pass the result into calc_probit_loglik to see what the model's overall probability is.
If the new likelihood is better (or accepted via Metropolis-Hastings), you keep the mutation. 
If it is worse, you revert the arrays back to their old state.


The MCMC Sweep LogicTo make this blazing fast, we maintain a sum_trees array (a running tally of all tree predictions) so we never have to recalculate the entire forest.When it is Tree $m$'s turn to mutate:Isolate: Subtract Tree $m$'s current predictions from sum_trees to get the partial_pred (the rest of the forest).Mutate: Randomly pick Grow, Prune, or Change and alter Tree $m$'s arrays in memory.Evaluate: Calculate the new log-likelihood.Decide: Calculate the acceptance ratio. If accepted, we keep the new arrays. If rejected, we instantly overwrite the pointers to revert the mutation.