"""
bart_survival.bart_adj
======================
High-performance Metropolis-Hastings BART sampler and Numba prediction kernels.

Provides:
- FastSurvMHBART: Specialized MCMC step method for survival analysis with time-varying
  probit hazard offsets and warm-started DART Dirichlet sparsity updates.
- _fast_predict_trace: Parallel Numba out-of-sample tree trace evaluator.
- predict_out_of_sample: High-level posterior predictive hazard calculator.
- simple_feature_importance: Variable split inclusion rate analyzer.
"""

import os
import numpy as np
import math
import pymc as pm
import pymc_bart as pmb
from pymc_bart.pgbart import PGBART
from numba import njit, prange
import scipy as sp


# ==========================================
# 1. THE NUMBA ROULETTE WHEEL (SPARSITY)
# ==========================================
@njit(fastmath=True)
def sample_feature(alpha_vec):
    total_weight = np.sum(alpha_vec)
    r = np.random.uniform(0.0, total_weight)
    cumulative = 0.0
    for i in range(len(alpha_vec)):
        cumulative += alpha_vec[i]
        if r <= cumulative:
            return i
    return len(alpha_vec) - 1

@njit(fastmath=True)
def get_split_counts(F, num_features):
    counts = np.zeros(num_features, dtype=np.int32)
    m_trees = F.shape[0]
    max_nodes = F.shape[1]
    for m in range(m_trees):
        for i in range(max_nodes):
            feat = F[m, i]
            # Only count living internal node features
            if feat >= 0: 
                counts[feat] += 1
    return counts

# ==========================================
# 2. TREE TOPOLOGY & LEAF-ADAPTIVE SPLITS
# ==========================================
@njit(fastmath=True)
def get_active_leaves(F_tree, current_node_count):
    leaves = np.empty(current_node_count, dtype=np.int16)
    leaf_count = 0
    for i in range(current_node_count):
        if F_tree[i] == -1:
            leaves[leaf_count] = i
            leaf_count += 1
    return leaves[:leaf_count]

@njit(fastmath=True)
def get_prunable_nodes(F_tree, L_tree, R_tree, current_node_count):
    prunable = np.empty(current_node_count, dtype=np.int16)
    count = 0
    for i in range(current_node_count):
        if F_tree[i] >= 0:  # Ignore masked (-2) dead nodes
            left, right = L_tree[i], R_tree[i]
            if left != -1 and right != -1:
                if F_tree[left] == -1 and F_tree[right] == -1:
                    prunable[count] = i
                    count += 1
    return prunable[:count]

@njit(fastmath=True)
def gather_leaf_indices(obs_leaf, leaf_a, leaf_b, leaf_idx_buffer):
    n_rows = obs_leaf.shape[0]
    k = 0
    if leaf_b < 0:
        for i in range(n_rows):
            if obs_leaf[i] == leaf_a:
                leaf_idx_buffer[k] = i
                k += 1
    else:
        for i in range(n_rows):
            lf = obs_leaf[i]
            if lf == leaf_a or lf == leaf_b:
                leaf_idx_buffer[k] = i
                k += 1
    return k


@njit(fastmath=True)
def prepare_leaf_cutpoints(X, leaf_idx_buffer, n_sub, split_feature, min_node_size, val_buffer, unique_midpoints=False):
    """
    Prepares sorted leaf values in val_buffer for fast single- or multi-try cutpoint sampling.
    Returns (ok, low_val, high_val, u_count).
    """
    if n_sub < 2 * min_node_size:
        return False, np.float32(0.0), np.float32(0.0), 0

    max_sample = 512 if unique_midpoints else 256

    if n_sub <= max_sample:
        for j in range(n_sub):
            val_buffer[j] = X[leaf_idx_buffer[j], split_feature]
        m_len = n_sub
        margin = min_node_size
    else:
        stride = n_sub // max_sample
        offset = np.random.randint(0, stride)
        k = 0
        for j in range(offset, n_sub, stride):
            if k < max_sample:
                val_buffer[k] = X[leaf_idx_buffer[j], split_feature]
                k += 1
        m_len = k
        margin = max(1, (min_node_size * m_len) // n_sub + 1)

    if m_len < 2 * margin:
        return False, np.float32(0.0), np.float32(0.0), 0

    sub_arr = val_buffer[:m_len]
    sub_arr.sort()

    low_val = np.float32(sub_arr[margin - 1])
    high_val = np.float32(sub_arr[m_len - margin])

    if low_val >= high_val:
        return False, np.float32(0.0), np.float32(0.0), 0

    if not unique_midpoints:
        return True, low_val, high_val, 0

    # Collapse duplicate values within [low_val, high_val] in-place
    u_count = 1
    sub_arr[0] = low_val
    for i in range(margin, m_len - margin + 1):
        v = sub_arr[i]
        if v > sub_arr[u_count - 1]:
            sub_arr[u_count] = v
            u_count += 1
            if v >= high_val:
                break

    if u_count < 2:
        return False, np.float32(0.0), np.float32(0.0), 0

    return True, low_val, high_val, u_count


@njit(fastmath=True)
def draw_prepared_cutpoint(low_val, high_val, u_count, val_buffer, unique_midpoints=False):
    if not unique_midpoints:
        return np.float32(np.random.uniform(low_val, high_val))
    idx = np.random.randint(0, u_count - 1)
    return np.float32(0.5 * (val_buffer[idx] + val_buffer[idx + 1]))


@njit(fastmath=True)
def sample_valid_midpoint_split(X, obs_leaf, leaf_a, leaf_b, n_sub, split_feature, min_node_size, val_buffer, unique_midpoints=False):
    """
    Backwards-compatible single-draw wrapper.
    """
    if n_sub < 2 * min_node_size:
        return False, np.float32(0.0)
    n_rows = X.shape[0]
    max_sample = 512 if unique_midpoints else 256
    if n_sub <= max_sample:
        k = 0
        for i in range(n_rows):
            lf = obs_leaf[i]
            if lf == leaf_a or lf == leaf_b:
                val_buffer[k] = X[i, split_feature]
                k += 1
        m_len = k
        margin = min_node_size
    else:
        stride = n_sub // max_sample
        offset = np.random.randint(0, stride)
        seen = 0
        k = 0
        for i in range(n_rows):
            lf = obs_leaf[i]
            if lf == leaf_a or lf == leaf_b:
                if (seen >= offset) and ((seen - offset) % stride == 0) and (k < max_sample):
                    val_buffer[k] = X[i, split_feature]
                    k += 1
                seen += 1
        m_len = k
        margin = max(1, (min_node_size * m_len) // n_sub + 1)

    if m_len < 2 * margin:
        return False, np.float32(0.0)

    sub_arr = val_buffer[:m_len]
    sub_arr.sort()
    low_val = sub_arr[margin - 1]
    high_val = sub_arr[m_len - margin]
    if low_val >= high_val:
        return False, np.float32(0.0)
    if not unique_midpoints:
        return True, np.float32(np.random.uniform(low_val, high_val))

    u_count = 1
    sub_arr[0] = low_val
    for i in range(margin, m_len - margin + 1):
        v = sub_arr[i]
        if v > sub_arr[u_count - 1]:
            sub_arr[u_count] = v
            u_count += 1
            if v >= high_val:
                break
    idx = np.random.randint(0, u_count - 1)
    return True, np.float32(0.5 * (sub_arr[idx] + sub_arr[idx + 1]))


@njit(fastmath=True)
def allocate_child_nodes(F_tree, current_count):
    left_child_idx, right_child_idx = -1, -1
    for i in range(current_count):
        if F_tree[i] == -2:
            if left_child_idx == -1:
                left_child_idx = i
            elif right_child_idx == -1:
                right_child_idx = i
                break

    new_count = current_count
    if left_child_idx == -1:
        left_child_idx = new_count
        new_count += 1
    if right_child_idx == -1:
        right_child_idx = new_count
        new_count += 1

    return left_child_idx, right_child_idx, new_count


# ==========================================
# 3. LIKELIHOOD, PRIORS, AND LATENT Z
# ==========================================
@njit(fastmath=True)
def leaf_loglik(n, sum_r, tau2):
    if n <= 0:
        return 0.0
    return -0.5 * math.log(1.0 + n * tau2) + (tau2 * (sum_r * sum_r)) / (2.0 * (1.0 + n * tau2))


@njit(fastmath=True)
def calc_grow_prior_ratio(node_depth, leaf_priors):
    if node_depth + 1 >= len(leaf_priors):
        return -1e9
    p_leaf_d = leaf_priors[node_depth]
    p_leaf_d_plus_1 = leaf_priors[node_depth + 1]
    p_split = 1.0 - p_leaf_d
    if p_leaf_d <= 0.0 or p_split <= 0.0:
        return -1e9
    prior_ratio = (p_split * (p_leaf_d_plus_1 ** 2)) / p_leaf_d
    return math.log(prior_ratio)


@njit(fastmath=True)
def _sample_std_normal_above(a):
    """
    Samples U ~ N(0, 1) truncated to U > a.
    Uses standard normal rejection for a <= 0.45 and Robert (1995) optimal
    exponential rejection for tail a > 0.45 (no clamping required).
    """
    if a <= 0.45:
        while True:
            u = np.random.normal(0.0, 1.0)
            if u > a:
                return u
    else:
        alpha_star = 0.5 * (a + math.sqrt(a * a + 4.0))
        while True:
            u = a + np.random.exponential(1.0) / alpha_star
            diff = u - alpha_star
            if np.random.exponential(1.0) >= 0.5 * diff * diff:
                return u


@njit(fastmath=True)
def draw_latent_z_numba(Y, eta):
    n = Y.shape[0]
    Z = np.empty(n, dtype=np.float32)
    for i in range(n):
        mu = float(eta[i])
        if Y[i] == 1:
            # Z = mu + U > 0  <=>  U > -mu
            u = _sample_std_normal_above(-mu)
            Z[i] = np.float32(mu + u)
        else:
            # Z = mu - U <= 0 <=>  U >= mu
            u = _sample_std_normal_above(mu)
            Z[i] = np.float32(mu - u)
    return Z


# ==========================================
# 4. SINGLE-PASS MASTER SWEEP (WITH K=5 MTM)
# ==========================================
@njit(fastmath=True, cache=True)
def mcmc_fractional_sweep(X, Z, F, T, L, R, V, D, node_counts, 
                          sum_trees, alpha_vec, 
                          leaf_priors, leaf_prior_std,
                          accept_stats,
                          update_fraction=0.1,
                          min_node_size=5,
                          global_mins=None,
                          global_maxes=None,
                          unique_midpoints=False
                          ):
    m_trees = F.shape[0]
    n_to_update = max(1, int(m_trees * update_fraction))
    indices = np.arange(m_trees)
    np.random.shuffle(indices)
    trees_to_update = indices[:n_to_update]

    n_rows = X.shape[0]
    max_nodes = F.shape[1]
    tau2 = float(leaf_prior_std * leaf_prior_std)
    K_TRY = 5

    obs_leaf = np.empty(n_rows, dtype=np.int16)
    partial_pred = np.empty(n_rows, dtype=np.float32)
    obs_res = np.empty(n_rows, dtype=np.float32)
    sum_res_buffer = np.zeros(max_nodes, dtype=np.float32)
    count_buffer = np.zeros(max_nodes, dtype=np.int32)
    val_buffer = np.empty(512, dtype=np.float32)
    leaf_idx_buffer = np.empty(n_rows, dtype=np.int32)

    cand_t = np.empty(K_TRY, dtype=np.float32)
    cand_l_sum = np.empty(K_TRY, dtype=np.float64)
    cand_l_cnt = np.empty(K_TRY, dtype=np.int32)
    cand_ll = np.empty(K_TRY, dtype=np.float64)
    cand_w = np.empty(K_TRY, dtype=np.float64)

    ref_t = np.empty(K_TRY - 1, dtype=np.float32)
    ref_l_sum = np.empty(K_TRY - 1, dtype=np.float64)
    ref_l_cnt = np.empty(K_TRY - 1, dtype=np.int32)
    ref_ll = np.empty(K_TRY, dtype=np.float64)

    for m in trees_to_update:
        accept_stats[0] += 1
        current_nodes = node_counts[m]
        sum_res_buffer[:current_nodes] = 0.0
        count_buffer[:current_nodes] = 0

        # SINGLE PASS: route observations to leaves & accumulate sufficient statistics
        for i in range(n_rows):
            node = 0
            while F[m, node] != -1:
                if X[i, F[m, node]] <= T[m, node]:
                    node = L[m, node]
                else:
                    node = R[m, node]
            obs_leaf[i] = node
            p_pred = sum_trees[i] - V[m, node]
            partial_pred[i] = p_pred
            res = Z[i] - p_pred
            obs_res[i] = res
            sum_res_buffer[node] += res
            count_buffer[node] += 1

        step_type = np.random.random()
        is_stump = (F[m, 0] == -1)
        if is_stump:
            p_grow, p_prune, p_change = 1.0, 0.0, 0.0
        else:
            p_grow, p_prune, p_change = 0.33, 0.33, 0.34

        if step_type < p_grow:
            # --- GROW PROPOSAL (Multiple-Try Metropolis K=5) ---
            leaves = get_active_leaves(F[m], current_nodes)
            num_leaves_old = len(leaves)
            if current_nodes < max_nodes - 2 and num_leaves_old > 0:
                leaf_to_split = leaves[np.random.randint(0, num_leaves_old)]
                n_leaf = count_buffer[leaf_to_split]
                if n_leaf >= 2 * min_node_size:
                    split_feature = sample_feature(alpha_vec)
                    n_sub = gather_leaf_indices(obs_leaf, leaf_to_split, -1, leaf_idx_buffer)
                    ok, low_val, high_val, u_count = prepare_leaf_cutpoints(
                        X, leaf_idx_buffer, n_sub, split_feature, min_node_size, val_buffer, unique_midpoints
                    )
                    if ok:
                        left_idx, right_idx, new_count = allocate_child_nodes(F[m], current_nodes)
                        if new_count < max_nodes:
                            for k in range(K_TRY):
                                cand_t[k] = draw_prepared_cutpoint(low_val, high_val, u_count, val_buffer, unique_midpoints)
                                cand_l_sum[k] = 0.0
                                cand_l_cnt[k] = 0

                            t0, t1, t2, t3, t4 = cand_t[0], cand_t[1], cand_t[2], cand_t[3], cand_t[4]
                            s0, s1, s2, s3, s4 = 0.0, 0.0, 0.0, 0.0, 0.0
                            c0, c1, c2, c3, c4 = 0, 0, 0, 0, 0

                            for j in range(n_sub):
                                idx = leaf_idx_buffer[j]
                                x_val = X[idx, split_feature]
                                r_val = float(obs_res[idx])
                                if x_val <= t0:
                                    s0 += r_val
                                    c0 += 1
                                if x_val <= t1:
                                    s1 += r_val
                                    c1 += 1
                                if x_val <= t2:
                                    s2 += r_val
                                    c2 += 1
                                if x_val <= t3:
                                    s3 += r_val
                                    c3 += 1
                                if x_val <= t4:
                                    s4 += r_val
                                    c4 += 1

                            cand_l_sum[0], cand_l_sum[1], cand_l_sum[2], cand_l_sum[3], cand_l_sum[4] = s0, s1, s2, s3, s4
                            cand_l_cnt[0], cand_l_cnt[1], cand_l_cnt[2], cand_l_cnt[3], cand_l_cnt[4] = c0, c1, c2, c3, c4

                            total_res_leaf = float(sum_res_buffer[leaf_to_split])
                            old_node_ll = leaf_loglik(n_leaf, total_res_leaf, tau2)
                            n_valid = 0
                            max_ll = -1e30

                            for k in range(K_TRY):
                                lc = cand_l_cnt[k]
                                rc = n_leaf - lc
                                if lc >= min_node_size and rc >= min_node_size:
                                    ls = cand_l_sum[k]
                                    rs = total_res_leaf - ls
                                    ll_k = leaf_loglik(lc, ls, tau2) + leaf_loglik(rc, rs, tau2)
                                    cand_ll[k] = ll_k
                                    n_valid += 1
                                    if ll_k > max_ll:
                                        max_ll = ll_k
                                else:
                                    cand_ll[k] = -1e30

                            if n_valid > 0:
                                sum_w = 0.0
                                for k in range(K_TRY):
                                    if cand_ll[k] > -1e20:
                                        w_k = math.exp(cand_ll[k] - max_ll)
                                    else:
                                        w_k = 0.0
                                    cand_w[k] = w_k
                                    sum_w += w_k

                                r_draw = np.random.uniform(0.0, sum_w)
                                cum_w = 0.0
                                best_k = 0
                                for k in range(K_TRY):
                                    cum_w += cand_w[k]
                                    if r_draw <= cum_w:
                                        best_k = k
                                        break

                                split_value = cand_t[best_k]
                                left_count = cand_l_cnt[best_k]
                                left_sum_res = np.float32(cand_l_sum[best_k])
                                right_count = n_leaf - left_count
                                right_sum_res = np.float32(total_res_leaf - cand_l_sum[best_k])

                                # Temporarily apply structure to count prunable nodes in new tree
                                F[m, leaf_to_split] = split_feature
                                T[m, leaf_to_split] = split_value
                                L[m, leaf_to_split] = left_idx
                                R[m, leaf_to_split] = right_idx
                                F[m, left_idx] = -1
                                F[m, right_idx] = -1
                                node_counts[m] = new_count

                                num_prunable_new = len(get_prunable_nodes(F[m], L[m], R[m], new_count))
                                prior_ratio = calc_grow_prior_ratio(D[m, leaf_to_split], leaf_priors)
                                prob_ratio = math.log(0.33) - math.log(p_grow)
                                transition_ratio = math.log(num_leaves_old) - math.log(num_prunable_new)
                                penalty = prior_ratio + transition_ratio + prob_ratio

                                log_mean_exp_new = max_ll + math.log(sum_w / K_TRY)
                                log_alpha = (log_mean_exp_new - old_node_ll) + penalty

                                if math.log(np.random.random()) < log_alpha:
                                    accept_stats[1] += 1
                                    parent_depth = D[m, leaf_to_split]
                                    D[m, left_idx] = parent_depth + 1
                                    D[m, right_idx] = parent_depth + 1
                                    count_buffer[left_idx] = left_count
                                    sum_res_buffer[left_idx] = left_sum_res
                                    count_buffer[right_idx] = right_count
                                    sum_res_buffer[right_idx] = right_sum_res
                                    for j in range(n_sub):
                                        idx = leaf_idx_buffer[j]
                                        if X[idx, split_feature] <= split_value:
                                            obs_leaf[idx] = left_idx
                                        else:
                                            obs_leaf[idx] = right_idx
                                else:
                                    F[m, leaf_to_split] = -1
                                    F[m, left_idx] = -2
                                    F[m, right_idx] = -2
                                    node_counts[m] = current_nodes

        elif step_type < (p_grow + p_prune):
            # --- PRUNE PROPOSAL (Exact MTM K=5 reverse balance) ---
            prunable_nodes = get_prunable_nodes(F[m], L[m], R[m], current_nodes)
            num_prunable_old = len(prunable_nodes)
            if num_prunable_old > 0:
                node_to_prune = prunable_nodes[np.random.randint(0, num_prunable_old)]
                left_c = L[m, node_to_prune]
                right_c = R[m, node_to_prune]
                old_feature = F[m, node_to_prune]

                merged_count = count_buffer[left_c] + count_buffer[right_c]
                merged_sum_res = sum_res_buffer[left_c] + sum_res_buffer[right_c]

                num_leaves_old = len(get_active_leaves(F[m], current_nodes))
                num_leaves_new = num_leaves_old - 1

                prior_ratio = -calc_grow_prior_ratio(D[m, node_to_prune], leaf_priors)
                p_grow_new = 1.0 if (node_to_prune == 0) else 0.33
                prob_ratio = math.log(p_grow_new) - math.log(p_prune)
                transition_ratio = math.log(num_prunable_old) - math.log(num_leaves_new)
                penalty = prior_ratio + transition_ratio + prob_ratio

                old_pair_ll = leaf_loglik(count_buffer[left_c], sum_res_buffer[left_c], tau2) + leaf_loglik(count_buffer[right_c], sum_res_buffer[right_c], tau2)
                new_merged_ll = leaf_loglik(merged_count, merged_sum_res, tau2)

                n_sub = gather_leaf_indices(obs_leaf, left_c, right_c, leaf_idx_buffer)
                ok_ref, low_ref, high_ref, u_ref = prepare_leaf_cutpoints(
                    X, leaf_idx_buffer, n_sub, old_feature, min_node_size, val_buffer, unique_midpoints
                )
                ref_ll[0] = old_pair_ll
                max_ll_old = old_pair_ll

                if ok_ref:
                    for k in range(K_TRY - 1):
                        ref_t[k] = draw_prepared_cutpoint(low_ref, high_ref, u_ref, val_buffer, unique_midpoints)

                    rt0, rt1, rt2, rt3 = ref_t[0], ref_t[1], ref_t[2], ref_t[3]
                    rs0, rs1, rs2, rs3 = 0.0, 0.0, 0.0, 0.0
                    rc0, rc1, rc2, rc3 = 0, 0, 0, 0

                    for j in range(n_sub):
                        idx = leaf_idx_buffer[j]
                        x_val = X[idx, old_feature]
                        r_val = float(obs_res[idx])
                        if x_val <= rt0:
                            rs0 += r_val
                            rc0 += 1
                        if x_val <= rt1:
                            rs1 += r_val
                            rc1 += 1
                        if x_val <= rt2:
                            rs2 += r_val
                            rc2 += 1
                        if x_val <= rt3:
                            rs3 += r_val
                            rc3 += 1

                    ref_l_sum[0], ref_l_sum[1], ref_l_sum[2], ref_l_sum[3] = rs0, rs1, rs2, rs3
                    ref_l_cnt[0], ref_l_cnt[1], ref_l_cnt[2], ref_l_cnt[3] = rc0, rc1, rc2, rc3

                    total_res_m = float(merged_sum_res)
                    for k in range(K_TRY - 1):
                        lc = ref_l_cnt[k]
                        rc = merged_count - lc
                        if lc >= min_node_size and rc >= min_node_size:
                            ls = ref_l_sum[k]
                            rs = total_res_m - ls
                            ll_k = leaf_loglik(lc, ls, tau2) + leaf_loglik(rc, rs, tau2)
                        else:
                            ll_k = old_pair_ll
                        ref_ll[k + 1] = ll_k
                        if ll_k > max_ll_old:
                            max_ll_old = ll_k
                else:
                    for k in range(1, K_TRY):
                        ref_ll[k] = old_pair_ll

                sum_w_old = 0.0
                for k in range(K_TRY):
                    sum_w_old += math.exp(ref_ll[k] - max_ll_old)
                log_mean_exp_old = max_ll_old + math.log(sum_w_old / K_TRY)

                log_alpha = (new_merged_ll - log_mean_exp_old) + penalty

                if math.log(np.random.random()) < log_alpha:
                    accept_stats[1] += 1
                    F[m, node_to_prune] = -1
                    F[m, left_c] = -2
                    F[m, right_c] = -2
                    count_buffer[node_to_prune] = merged_count
                    sum_res_buffer[node_to_prune] = merged_sum_res
                    for j in range(n_sub):
                        obs_leaf[leaf_idx_buffer[j]] = node_to_prune

        else:
            # --- CHANGE PROPOSAL (Multiple-Try Metropolis K=5 on singly-internal nodes) ---
            prunable_nodes = get_prunable_nodes(F[m], L[m], R[m], current_nodes)
            num_prunable = len(prunable_nodes)
            if num_prunable > 0:
                node_to_change = prunable_nodes[np.random.randint(0, num_prunable)]
                left_c = L[m, node_to_change]
                right_c = R[m, node_to_change]
                old_feature = F[m, node_to_change]
                n_parent = count_buffer[left_c] + count_buffer[right_c]
                sum_res_parent = sum_res_buffer[left_c] + sum_res_buffer[right_c]

                if n_parent >= 2 * min_node_size:
                    new_feature = sample_feature(alpha_vec)
                    n_sub = gather_leaf_indices(obs_leaf, left_c, right_c, leaf_idx_buffer)
                    ok, low_val, high_val, u_count = prepare_leaf_cutpoints(
                        X, leaf_idx_buffer, n_sub, new_feature, min_node_size, val_buffer, unique_midpoints
                    )
                    if ok:
                        for k in range(K_TRY):
                            cand_t[k] = draw_prepared_cutpoint(low_val, high_val, u_count, val_buffer, unique_midpoints)

                        t0, t1, t2, t3, t4 = cand_t[0], cand_t[1], cand_t[2], cand_t[3], cand_t[4]
                        s0, s1, s2, s3, s4 = 0.0, 0.0, 0.0, 0.0, 0.0
                        c0, c1, c2, c3, c4 = 0, 0, 0, 0, 0

                        for j in range(n_sub):
                            idx = leaf_idx_buffer[j]
                            x_val = X[idx, new_feature]
                            r_val = float(obs_res[idx])
                            if x_val <= t0:
                                s0 += r_val
                                c0 += 1
                            if x_val <= t1:
                                s1 += r_val
                                c1 += 1
                            if x_val <= t2:
                                s2 += r_val
                                c2 += 1
                            if x_val <= t3:
                                s3 += r_val
                                c3 += 1
                            if x_val <= t4:
                                s4 += r_val
                                c4 += 1

                        cand_l_sum[0], cand_l_sum[1], cand_l_sum[2], cand_l_sum[3], cand_l_sum[4] = s0, s1, s2, s3, s4
                        cand_l_cnt[0], cand_l_cnt[1], cand_l_cnt[2], cand_l_cnt[3], cand_l_cnt[4] = c0, c1, c2, c3, c4

                        total_res_p = float(sum_res_parent)
                        n_valid = 0
                        max_ll_new = -1e30

                        for k in range(K_TRY):
                            lc = cand_l_cnt[k]
                            rc = n_parent - lc
                            if lc >= min_node_size and rc >= min_node_size:
                                ls = cand_l_sum[k]
                                rs = total_res_p - ls
                                ll_k = leaf_loglik(lc, ls, tau2) + leaf_loglik(rc, rs, tau2)
                                cand_ll[k] = ll_k
                                n_valid += 1
                                if ll_k > max_ll_new:
                                    max_ll_new = ll_k
                            else:
                                cand_ll[k] = -1e30

                        if n_valid > 0:
                            sum_w_new = 0.0
                            for k in range(K_TRY):
                                if cand_ll[k] > -1e20:
                                    w_k = math.exp(cand_ll[k] - max_ll_new)
                                else:
                                    w_k = 0.0
                                cand_w[k] = w_k
                                sum_w_new += w_k

                            r_draw = np.random.uniform(0.0, sum_w_new)
                            cum_w = 0.0
                            best_k = 0
                            for k in range(K_TRY):
                                cum_w += cand_w[k]
                                if r_draw <= cum_w:
                                    best_k = k
                                    break

                            old_pair_ll = leaf_loglik(count_buffer[left_c], sum_res_buffer[left_c], tau2) + leaf_loglik(count_buffer[right_c], sum_res_buffer[right_c], tau2)
                            ref_ll[0] = old_pair_ll
                            max_ll_old = old_pair_ll

                            ok_ref, low_ref, high_ref, u_ref = prepare_leaf_cutpoints(
                                X, leaf_idx_buffer, n_sub, old_feature, min_node_size, val_buffer, unique_midpoints
                            )
                            if ok_ref:
                                for k in range(K_TRY - 1):
                                    ref_t[k] = draw_prepared_cutpoint(low_ref, high_ref, u_ref, val_buffer, unique_midpoints)

                                rt0, rt1, rt2, rt3 = ref_t[0], ref_t[1], ref_t[2], ref_t[3]
                                rs0, rs1, rs2, rs3 = 0.0, 0.0, 0.0, 0.0
                                rc0, rc1, rc2, rc3 = 0, 0, 0, 0

                                for j in range(n_sub):
                                    idx = leaf_idx_buffer[j]
                                    x_val = X[idx, old_feature]
                                    r_val = float(obs_res[idx])
                                    if x_val <= rt0:
                                        rs0 += r_val
                                        rc0 += 1
                                    if x_val <= rt1:
                                        rs1 += r_val
                                        rc1 += 1
                                    if x_val <= rt2:
                                        rs2 += r_val
                                        rc2 += 1
                                    if x_val <= rt3:
                                        rs3 += r_val
                                        rc3 += 1

                                ref_l_sum[0], ref_l_sum[1], ref_l_sum[2], ref_l_sum[3] = rs0, rs1, rs2, rs3
                                ref_l_cnt[0], ref_l_cnt[1], ref_l_cnt[2], ref_l_cnt[3] = rc0, rc1, rc2, rc3

                                for k in range(K_TRY - 1):
                                    lc = ref_l_cnt[k]
                                    rc = n_parent - lc
                                    if lc >= min_node_size and rc >= min_node_size:
                                        ls = ref_l_sum[k]
                                        rs = total_res_p - ls
                                        ll_k = leaf_loglik(lc, ls, tau2) + leaf_loglik(rc, rs, tau2)
                                    else:
                                        ll_k = old_pair_ll
                                    ref_ll[k + 1] = ll_k
                                    if ll_k > max_ll_old:
                                        max_ll_old = ll_k
                            else:
                                for k in range(1, K_TRY):
                                    ref_ll[k] = old_pair_ll

                            sum_w_old = 0.0
                            for k in range(K_TRY):
                                sum_w_old += math.exp(ref_ll[k] - max_ll_old)

                            log_mean_exp_new = max_ll_new + math.log(sum_w_new / K_TRY)
                            log_mean_exp_old = max_ll_old + math.log(sum_w_old / K_TRY)
                            log_alpha = log_mean_exp_new - log_mean_exp_old

                            if math.log(np.random.random()) < log_alpha:
                                accept_stats[1] += 1
                                new_threshold = cand_t[best_k]
                                new_left_count = cand_l_cnt[best_k]
                                new_left_sum_res = np.float32(cand_l_sum[best_k])
                                new_right_count = n_parent - new_left_count
                                new_right_sum_res = np.float32(total_res_p - cand_l_sum[best_k])

                                F[m, node_to_change] = new_feature
                                T[m, node_to_change] = new_threshold
                                count_buffer[left_c] = new_left_count
                                sum_res_buffer[left_c] = new_left_sum_res
                                count_buffer[right_c] = new_right_count
                                sum_res_buffer[right_c] = new_right_sum_res
                                for j in range(n_sub):
                                    idx = leaf_idx_buffer[j]
                                    if X[idx, new_feature] <= new_threshold:
                                        obs_leaf[idx] = left_c
                                    else:
                                        obs_leaf[idx] = right_c

        # O(leaves) CONJUGATE GIBBS LEAF UPDATE & O(N) FLAT PREDICTION UPDATE (NO TREE TRAVERSAL)
        active_nodes = node_counts[m]
        for node in range(active_nodes):
            if F[m, node] == -1:
                n = count_buffer[node]
                if n > 0:
                    post_var = tau2 / (1.0 + n * tau2)
                    post_mean = post_var * sum_res_buffer[node]
                    V[m, node] = post_mean + np.random.normal(0.0, math.sqrt(post_var))
                else:
                    V[m, node] = np.random.normal(0.0, leaf_prior_std)

        for i in range(n_rows):
            sum_trees[i] = partial_pred[i] + V[m, obs_leaf[i]]

# ==========================================
# 5. PYTHON API HANDOFF
# ==========================================
class FastTreeState:
    def __init__(self, m_trees, init_mean, max_nodes=255):
        self.m_trees = m_trees
        self.max_nodes = max_nodes
        self.F = np.full((m_trees, max_nodes), -1, dtype=np.int16)
        self.L = np.full((m_trees, max_nodes), -1, dtype=np.int16)
        self.R = np.full((m_trees, max_nodes), -1, dtype=np.int16)
        self.D = np.ones((m_trees, max_nodes), dtype=np.int8)
        # self.D = np.zeros((m_trees, max_nodes), dtype=np.int8) 
        self.T = np.zeros((m_trees, max_nodes), dtype=np.float32)
        self.V = np.zeros((m_trees, max_nodes), dtype=np.float32)
        self.node_counts = np.ones(m_trees, dtype=np.int16)
        self.V[:, 0] = init_mean / m_trees


class FastSurvMHBART(PGBART):
    def __init__(self, 
                 vars, 
                 target_x=None, 
                 target_y=None, 
                 use_sparsity=False, 
                 sparsity_warmup=300,
                 update_fraction=1.0, 
                 k=2.0,
                 min_node_size=5,
                 offset=None,
                 split_method="empirical",
                 unique_midpoints=None,
                 return_estimates=False, *args, **kwargs):
        super().__init__(vars, *args, **kwargs)
        
        self.accept_count = 0
        self.total_count = 0
        self.use_sparsity = bool(use_sparsity)
        self.sparsity_warmup = int(sparsity_warmup)
        self.step_counter = 0
        self.update_fraction = float(update_fraction)
        self.min_node_size = int(min_node_size)
        if unique_midpoints is not None:
            self.unique_midpoints = bool(unique_midpoints)
        else:
            self.unique_midpoints = (str(split_method).lower() in ("empirical", "empirical_512", "midpoints", "true", "1"))
        
        if target_x is None or target_y is None or return_estimates == True:
            actual_X = self.X
            actual_Y = self.Y
            self.return_estimates = True
            print("returning estimates")
        else:
            self.return_estimates = False
            actual_X = target_x
            actual_Y = target_y
        
        self.num_features = int(actual_X.shape[1])

        # Upgrade 1: Time-specific empirical probit baseline hazard offset alpha_t = Phi^{-1}(p_t)
        if offset is None:
            t_col = actual_X[:, 0]
            y_flat = actual_Y.ravel()
            unique_t = np.unique(t_col).astype(np.float64)
            alpha_t = np.zeros(len(unique_t), dtype=np.float32)
            for idx_t, t_val in enumerate(unique_t):
                mask_t = (t_col == t_val)
                n_t = int(np.sum(mask_t))
                e_t = float(np.sum(y_flat[mask_t]))
                p_t = np.clip((e_t + 0.5) / (n_t + 1.0), 1e-5, 1.0 - 1e-5)
                alpha_t[idx_t] = np.float32(sp.stats.norm.ppf(p_t))
            self.offset = (unique_t, alpha_t)
            self.train_offset_vec = np.interp(t_col, unique_t, alpha_t).astype(np.float32)
        elif isinstance(offset, tuple):
            self.offset = offset
            self.train_offset_vec = np.interp(actual_X[:, 0], offset[0], offset[1]).astype(np.float32)
        else:
            self.offset = offset
            if np.ndim(offset) == 0:
                self.train_offset_vec = np.full(actual_X.shape[0], offset, dtype=np.float32)
            else:
                self.train_offset_vec = np.asarray(offset, dtype=np.float32)
        
        self.c_tree_state = FastTreeState(m_trees=self.m, init_mean=0.0, max_nodes=255)
        self.flat_sum_trees = np.zeros(actual_X.shape[0], dtype=np.float32)
        self.k = k
        self.leaf_prior_std = float(3.0 / (self.k * np.sqrt(self.m)))

        self.numba_X = np.copy(np.ascontiguousarray(actual_X))
        self.numba_Y = np.copy(np.ascontiguousarray(actual_Y)).ravel()
                
        self.global_mins = np.min(self.numba_X, axis=0).astype(np.float32)
        self.global_maxes = np.max(self.numba_X, axis=0).astype(np.float32)

        self.alpha_vec = np.ones(self.num_features, dtype=np.float32)
        self.numba_leaf_priors = np.array(self.prior_prob_leaf_node, dtype=np.float32)
        self.stats_dtypes = [{
            "variable_inclusion": np.ndarray,
            "tune": bool,
            "F_trees": np.ndarray,
            "T_trees": np.ndarray,
            "L_trees": np.ndarray,
            "R_trees": np.ndarray,
            "V_trees": np.ndarray,
            "node_counts": np.ndarray
        }]
        self.accept_stats = np.zeros(2, dtype=np.int64)

    def astep(self, q0):
        self.step_counter += 1
        current_eta = self.flat_sum_trees + self.train_offset_vec
        Z = draw_latent_z_numba(self.numba_Y, current_eta)
        
        Z_residual = Z - self.train_offset_vec
        Z_numba = np.ascontiguousarray(Z_residual, dtype=np.float32)

        mcmc_fractional_sweep(
            self.numba_X, Z_numba, 
            self.c_tree_state.F, self.c_tree_state.T, 
            self.c_tree_state.L, self.c_tree_state.R, 
            self.c_tree_state.V, self.c_tree_state.D, 
            self.c_tree_state.node_counts,
            self.flat_sum_trees, 
            self.alpha_vec, 
            self.numba_leaf_priors,
            self.leaf_prior_std,
            self.accept_stats,
            self.update_fraction,
            self.min_node_size,
            self.global_mins,
            self.global_maxes,
            self.unique_midpoints
        )
        
        var_counts = get_split_counts(self.c_tree_state.F, self.num_features)
        
        # Upgrade 2: Warm-Started DART Dirichlet Gibbs Update (default off)
        if self.use_sparsity and self.step_counter >= self.sparsity_warmup:
            base_prior = 1.0 / self.num_features
            concentration_params = base_prior + var_counts.astype(np.float64)
            new_weights = np.random.dirichlet(concentration_params)
            self.alpha_vec = new_weights.astype(np.float32)

        stats = {
            "variable_inclusion": var_counts, 
            "tune": self.tune,
            "F_trees": self.c_tree_state.F.copy(),
            "T_trees": self.c_tree_state.T.copy(),
            "L_trees": self.c_tree_state.L.copy(),
            "R_trees": self.c_tree_state.R.copy(),
            "V_trees": self.c_tree_state.V.copy(),
            "node_counts": self.c_tree_state.node_counts.copy()
        }

        if self.return_estimates:
            self.sum_trees[0, 0, :] = self.flat_sum_trees
            return self.sum_trees, [stats]
        else:
            return np.zeros(1), [stats]

@njit(fastmath=True, parallel=True)
def _fast_predict_trace(X, F, T, L, R, V):
    n_samples = F.shape[0]
    m_trees = F.shape[1]
    n_obs = X.shape[0]
    preds = np.zeros((n_samples, n_obs), dtype=np.float32)
    for s in prange(n_samples):
        for m in range(m_trees):
            f_tree = F[s, m]
            t_tree = T[s, m]
            l_tree = L[s, m]
            r_tree = R[s, m]
            v_tree = V[s, m]
            for i in range(n_obs):
                node = 0
                while f_tree[node] != -1:
                    feat = f_tree[node]
                    if X[i, feat] <= t_tree[node]:
                        node = l_tree[node]
                    else:
                        node = r_tree[node]
                preds[s, i] += v_tree[node]
    return preds

def predict_out_of_sample(trace, X_new, offset, m_trees=50, max_nodes=255, thin=1):
    print("Starting predict ...")
    stats = trace.sample_stats
    n_samples_raw = stats["F_trees"].values.size // (m_trees * max_nodes)
    
    F = np.ascontiguousarray(stats["F_trees"].values).reshape(n_samples_raw, m_trees, max_nodes)[::thin].astype(np.int16, copy=False)
    T = np.ascontiguousarray(stats["T_trees"].values).reshape(n_samples_raw, m_trees, max_nodes)[::thin].astype(np.float32, copy=False)
    L = np.ascontiguousarray(stats["L_trees"].values).reshape(n_samples_raw, m_trees, max_nodes)[::thin].astype(np.int16, copy=False)
    R = np.ascontiguousarray(stats["R_trees"].values).reshape(n_samples_raw, m_trees, max_nodes)[::thin].astype(np.int16, copy=False)
    V = np.ascontiguousarray(stats["V_trees"].values).reshape(n_samples_raw, m_trees, max_nodes)[::thin].astype(np.float32, copy=False)
    
    n_samples_thinned = F.shape[0]
    X_numba = np.ascontiguousarray(X_new, dtype=np.float64)
    n_obs = X_new.shape[0]
    
    print(f"Processing: {n_obs} observations, {n_samples_thinned} thinned draws (Thin factor: {thin})...")
    raw_tree_sums = _fast_predict_trace(X_numba, F, T, L, R, V)
    if offset is None:
        latent_scores = raw_tree_sums
    elif isinstance(offset, tuple):
        offset_vec = np.interp(X_new[:, 0], offset[0], offset[1]).astype(np.float32)
        latent_scores = raw_tree_sums + offset_vec[np.newaxis, :]
    else:
        latent_scores = raw_tree_sums + offset
    
    probabilities = sp.stats.norm.cdf(latent_scores)
    return probabilities


def simple_feature_importance(trace):
    raw_counts = trace.sample_stats["variable_inclusion"].values
    # 2. Sum the counts across all chains (axis=0) and all draws (axis=1)
    # This gives you a single 1D array of total splits per feature across the whole posterior
    total_counts = np.nansum(raw_counts, axis=(0, 1))

    # 3. Normalize to get relative feature importance (probabilities)
    feature_importance = total_counts / total_counts.sum()

    # 4. (Optional) Print the results neatly sorted by importance
    sorted_indices = np.argsort(feature_importance)[::-1]

    print("--- Posterior Feature Importance ---")
    for idx in sorted_indices:
        importance = feature_importance[idx]
        if importance > 0.001:  # Only show features that were actually used
            print(f"Feature {idx:2d}: {importance:.4f}  ({importance * 100:>5.2f}%)")