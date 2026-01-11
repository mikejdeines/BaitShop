import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as opt
from ott.geometry import geometry
from ott.solvers.quadratic.gromov_wasserstein_lr import LRGromovWasserstein
from typing import List, Tuple

def compute_full_cost_matrix(correlation_matrix, 
                           distance_matrix,
                           penalty_function='inverse'):
    """
    Compute the full 4D cost tensor for the assignment problem.
    
    Args:
        correlation_matrix: (n_genes, n_genes) gene correlation matrix
        distance_matrix: (n_barcodes, n_barcodes) barcode distance matrix
        penalty_function: How to convert distance to penalty ('inverse', 'exponential', 'linear')
        
    Returns:
        cost_tensor: (n_genes, n_barcodes, n_genes, n_barcodes) cost tensor
                    where cost_tensor[i,k,j,l] is the cost of assigning
                    gene i to barcode k and gene j to barcode l
    """
    n_genes = correlation_matrix.shape[0]
    n_barcodes = distance_matrix.shape[0]
    
    print(f"Computing full cost tensor of size ({n_genes}, {n_barcodes}, {n_genes}, {n_barcodes})...")
    
    # Initialize cost tensor
    cost_tensor = np.zeros((n_genes, n_barcodes, n_genes, n_barcodes))
    
    # Convert distance to penalty based on the chosen function
    if penalty_function == 'inverse':
        # f(D) = 1/(1 + D), so high distance -> low penalty
        penalty_matrix = 1.0 / (1.0 + distance_matrix)
    elif penalty_function == 'exponential':
        # f(D) = exp(-lambda * D), with lambda chosen so exp(-lambda * max_D) ≈ 0.1
        max_d = np.max(distance_matrix)
        lambda_val = -np.log(0.1) / max_d if max_d > 0 else 1.0
        penalty_matrix = np.exp(-lambda_val * distance_matrix)
    elif penalty_function == 'linear':
        # f(D) = (max_D - D) / max_D, so high distance -> low penalty
        max_d = np.max(distance_matrix)
        penalty_matrix = (max_d - distance_matrix) / max_d if max_d > 0 else np.ones_like(distance_matrix)
    else:
        raise ValueError(f"Unknown penalty function: {penalty_function}")
    
    # Fill the cost tensor
    # Cost when gene i->barcode k and gene j->barcode l is:
    # C_ij * f(D_kl) for i ≠ j and k ≠ l
    for i in range(n_genes):
        for k in range(n_barcodes):
            for j in range(n_genes):
                for l in range(n_barcodes):
                    if i == j or k == l:
                        # No cost for same gene or same barcode
                        cost_tensor[i, k, j, l] = 0
                    else:
                        # High correlation + low distance = high cost (bad)
                        # High correlation + high distance = low cost (good)
                        cost_tensor[i, k, j, l] = correlation_matrix[i, j] * penalty_matrix[k, l]
    
    return cost_tensor

def compute_linearized_cost_matrix(correlation_matrix,
                                 distance_matrix,
                                 penalty_function='inverse'):
    """
    Compute a linearized 2D cost matrix for use with Hungarian algorithm.
    This approximates the full 4D cost tensor.
    
    Args:
        correlation_matrix: (n_genes, n_genes) gene correlation matrix
        distance_matrix: (n_barcodes, n_barcodes) barcode distance matrix
        penalty_function: How to convert distance to penalty
        
    Returns:
        cost_matrix: (n_genes, n_barcodes) linearized cost matrix
    """
    n_genes = correlation_matrix.shape[0]
    n_barcodes = distance_matrix.shape[0]
    
    # Convert distance to penalty
    if penalty_function == 'inverse':
        penalty_matrix = 1.0 / (1.0 + distance_matrix)
    elif penalty_function == 'exponential':
        max_d = np.max(distance_matrix)
        lambda_val = -np.log(0.1) / max_d if max_d > 0 else 1.0
        penalty_matrix = np.exp(-lambda_val * distance_matrix)
    else:  # linear
        max_d = np.max(distance_matrix)
        penalty_matrix = (max_d - distance_matrix) / max_d if max_d > 0 else np.ones_like(distance_matrix)
    
    cost_matrix = np.zeros((n_genes, n_barcodes))
    
    for i in range(n_genes):
        for k in range(n_barcodes):
            # Cost for assigning gene i to barcode k is the sum over all other genes j
            # of: correlation(i,j) * average_penalty_to_other_barcodes(k)
            total_cost = 0
            for j in range(n_genes):
                if i != j:
                    # Average penalty from barcode k to all other barcodes
                    other_barcodes = [l for l in range(n_barcodes) if l != k]
                    avg_penalty = np.mean(penalty_matrix[k, other_barcodes])
                    total_cost += correlation_matrix[i, j] * avg_penalty
            cost_matrix[i, k] = total_cost
    
    return cost_matrix

def solve_with_hungarian(cost_matrix):
    """
    Solve assignment using Hungarian algorithm.
    
    Args:
        cost_matrix: (n_genes, n_barcodes) cost matrix
        
    Returns:
        assignment: Array where assignment[i] is barcode index for gene i
    """
    # Hungarian algorithm minimizes total cost
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    
    # Create assignment array
    assignment = np.zeros(cost_matrix.shape[0], dtype=int)
    assignment[row_ind] = col_ind
    
    return assignment

def solve_with_simulated_annealing(cost_tensor,
                                 initial_temperature,
                                 cooling_rate,
                                 max_iterations=10000):
    """
    Solve assignment using simulated annealing with the full cost tensor.
    
    Args:
        cost_tensor: Full 4D cost tensor
        initial_temperature: Starting temperature for SA
        cooling_rate: Geometric cooling rate
        max_iterations: Maximum number of iterations
        
    Returns:
        assignment: Array where assignment[i] is barcode index for gene i
    """
    n_genes, n_barcodes, _, _ = cost_tensor.shape
    
    # Initialize random assignment
    current_assignment = np.random.choice(n_barcodes, size=n_genes, replace=False)
    current_cost = compute_assignment_cost(cost_tensor, current_assignment)
    
    best_assignment = current_assignment.copy()
    best_cost = current_cost
    
    temperature = initial_temperature
    
    for iteration in range(max_iterations):
        # Generate neighbor by swapping two random genes
        i, j = np.random.choice(n_genes, size=2, replace=False)
        neighbor_assignment = current_assignment.copy()
        neighbor_assignment[i], neighbor_assignment[j] = neighbor_assignment[j], neighbor_assignment[i]
        
        neighbor_cost = compute_assignment_cost(cost_tensor, neighbor_assignment)
        
        # Accept or reject
        cost_diff = neighbor_cost - current_cost
        if cost_diff < 0 or np.random.random() < np.exp(-cost_diff / temperature):
            current_assignment = neighbor_assignment
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_assignment = current_assignment.copy()
                best_cost = current_cost
        
        # Cool down
        temperature *= cooling_rate
        
        if temperature < 1e-6:
            break
    
    return best_assignment

def compute_assignment_cost(cost_tensor, assignment):
    """
    Compute the total cost of an assignment using the full cost tensor.
    
    Args:
        cost_tensor: Full 4D cost tensor
        assignment: Array where assignment[i] is barcode index for gene i
        
    Returns:
        total_cost: Total cost of the assignment
    """
    n_genes = len(assignment)
    total_cost = 0.0
    
    # Sum over all pairs of genes
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            k = assignment[i]
            l = assignment[j]
            total_cost += cost_tensor[i, k, j, l]
    
    return total_cost

def select_diverse_barcodes(distance_matrix, n_select):
    """
    Select a diverse subset of barcodes using max-min diversity.
    
    Args:
        distance_matrix: (n_barcodes, n_barcodes) distance matrix
        n_select: Number of barcodes to select
        
    Returns:
        selected_indices: Indices of selected barcodes
    """
    n_barcodes = distance_matrix.shape[0]
    selected = []
    remaining = set(range(n_barcodes))
    
    # Start with two most distant barcodes
    if n_barcodes >= 2:
        max_dist = -1
        for i in range(n_barcodes):
            for j in range(i + 1, n_barcodes):
                if distance_matrix[i, j] > max_dist:
                    max_dist = distance_matrix[i, j]
                    start_pair = [i, j]
        selected = start_pair.copy()
        remaining -= set(start_pair)
    else:
        selected = [0]
        remaining = set()
    
    # Greedily add barcodes that maximize minimum distance to selected set
    while len(selected) < n_select and remaining:
        best_candidate = -1
        best_min_dist = -1
        
        for candidate in remaining:
            min_dist = min(distance_matrix[candidate, s] for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate
        
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    
    return np.array(selected)

def evaluate_assignment(correlation_matrix,
                      distance_matrix,
                      assignment):
    """
    Evaluate the quality of an assignment.
    
    Args:
        correlation_matrix: (n_genes, n_genes) gene correlation matrix
        distance_matrix: (n_barcodes, n_barcodes) barcode distance matrix
        assignment: Array where assignment[i] is barcode index for gene i
        
    Returns:
        metrics: Dictionary of quality metrics
    """
    n_genes = len(assignment)
    
    # Compute all pairwise correlations and distances
    correlations = []
    distances = []
    
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            correlations.append(correlation_matrix[i, j])
            barcode_i = assignment[i]
            barcode_j = assignment[j]
            distances.append(distance_matrix[barcode_i, barcode_j])
    
    correlations = np.array(correlations)
    distances = np.array(distances)
    
    # Compute metrics for different correlation ranges
    high_corr = correlations > 0.7
    med_corr = (correlations > 0.3) & (correlations <= 0.7)
    low_corr = correlations <= 0.3
    
    metrics = {
        'mean_distance_high_corr': np.mean(distances[high_corr]) if np.any(high_corr) else 0,
        'mean_distance_med_corr': np.mean(distances[med_corr]) if np.any(med_corr) else 0,
        'mean_distance_low_corr': np.mean(distances[low_corr]) if np.any(low_corr) else 0,
        'min_distance_high_corr': np.min(distances[high_corr]) if np.any(high_corr) else 0,
        'correlation_distance_pearson': np.corrcoef(correlations, distances)[0, 1] if len(correlations) > 1 else 0,
        'correlation_distance_spearman': _spearman_correlation(correlations, distances) if len(correlations) > 1 else 0,
    }
    
    return metrics

def _spearman_correlation(x, y):
    """Compute Spearman correlation."""
    from scipy.stats import spearmanr
    if len(x) > 1:
        return spearmanr(x, y).correlation
    return 0.0

def assign_barcodes_linear(genes,
                            barcodes,
                            correlation_matrix,
                            distance_matrix,
                            method,
                            use_full_cost=False):
    """
    Main function to assign barcodes to genes using Optimal Transport approach.
    
    Args:
        genes: List of gene names
        barcodes: List of barcode strings
        correlation_matrix: (n_genes, n_genes) gene correlation matrix
        distance_matrix: (n_barcodes, n_barcodes) barcode distance matrix
        method: 'hungarian' or 'simulated_annealing'
        use_full_cost: Whether to use full 4D cost tensor (computationally expensive)
        
    Returns:
        assignment: Array where assignment[i] is barcode index for gene i
        metrics: Quality metrics dictionary
    """
    n_genes = len(genes)
    n_barcodes = len(barcodes)
    
    # Validate that we have enough barcodes
    if n_barcodes < len(correlation_matrix):
        raise ValueError(
            f"Insufficient barcodes: {n_barcodes} barcodes provided but "
            f"{len(correlation_matrix)} required (correlation_matrix size)"
        )
    
    print(f"Assigning {n_genes} genes to pool of {n_barcodes} barcodes...")
    
    # Step 1: If we have more barcodes than genes, select a diverse subset
    if n_barcodes > n_genes:
        print("Selecting diverse barcode subset...")
        selected_barcode_indices = select_diverse_barcodes(distance_matrix, n_genes)
        selected_barcodes = [barcodes[i] for i in selected_barcode_indices]
        selected_distance_matrix = distance_matrix[np.ix_(selected_barcode_indices, selected_barcode_indices)]
        working_barcodes = selected_barcodes
        working_distance_matrix = selected_distance_matrix
        original_to_working = selected_barcode_indices
    else:
        working_barcodes = barcodes
        working_distance_matrix = distance_matrix
        original_to_working = np.arange(n_barcodes)
    
    n_working = len(working_barcodes)
    
    # Step 2: Compute cost matrix
    if use_full_cost and method == 'simulated_annealing':
        print("Computing full cost tensor...")
        cost_tensor = compute_full_cost_matrix(correlation_matrix, working_distance_matrix)
        assignment_working = solve_with_simulated_annealing(cost_tensor)
    else:
        print("Computing linearized cost matrix...")
        cost_matrix = compute_linearized_cost_matrix(correlation_matrix, working_distance_matrix)
        assignment_working = solve_with_hungarian(cost_matrix)
    
    # Step 3: Map back to original barcode indices if we used subset selection
    if n_barcodes > n_genes:
        assignment = original_to_working[assignment_working]
        # For evaluation, use full distance matrix
        eval_distance_matrix = distance_matrix
    else:
        assignment = assignment_working
        eval_distance_matrix = working_distance_matrix
    
    # Step 4: Evaluate assignment quality
    metrics = evaluate_assignment(correlation_matrix, eval_distance_matrix, assignment)
    
    return assignment, metrics


def assign_barcodes_lowrank_gw(
    genes: List[str],
    barcodes: List[str],
    correlation_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    variance_fraction: float = 0.99,
    epsilon: float = 1e-3,
    max_iterations: int = 2000,
):
    """
    Assign barcodes to genes using low-rank Gromov-Wasserstein (GW) with OTT-JAX.
    Automatically selects rank based on cumulative variance explained by C.

    Args:
        genes: List of n gene names
        barcodes: List of m barcode strings
        correlation_matrix: (n x n) gene correlations
        distance_matrix: (m x m) barcode similarity/distances
        variance_fraction: Fraction of variance to capture for low-rank factorization
        epsilon: Entropic regularization
        max_iterations: GW solver iterations

    Returns:
        assignment: Array of length n (barcode index per gene)
        coupling: Soft coupling matrix (n x m)
    """

    n = correlation_matrix.shape[0]
    m = distance_matrix.shape[0]

    # 1) Symmetrize correlation matrix
    C = 0.5 * (correlation_matrix + correlation_matrix.T)

    # 2) Eigen-decomposition for low-rank factorization
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals_sorted_idx = np.argsort(eigvals)[::-1]  # descending
    eigvals_sorted = eigvals[eigvals_sorted_idx]
    eigvecs_sorted = eigvecs[:, eigvals_sorted_idx]

    # 3) Determine rank r to capture desired variance_fraction
    cumulative_variance = np.cumsum(np.maximum(eigvals_sorted, 0))
    total_variance = cumulative_variance[-1]
    r = int(np.searchsorted(cumulative_variance / total_variance, variance_fraction)) + 1

    # 4) Construct low-rank U
    U = eigvecs_sorted[:, :r] @ np.diag(np.sqrt(np.maximum(eigvals_sorted[:r], 0.0)))
    U = jnp.array(U)

    # 5) Geometry for barcodes
    D = jnp.array(distance_matrix)
    geom_barcodes = geometry.Geometry(cost_matrix=D, epsilon=epsilon)

    # 6) Uniform marginals
    a = jnp.ones(n) / n
    b = jnp.ones(m) / m

    # 7) Solve low-rank GW
    solver = LRGromovWasserstein(
        rank=r,
        epsilon=epsilon,
        max_iterations=max_iterations,
    )
    out = solver.solve(
        geom_barcodes,
        a=a,
        b=b,
        x=U,
        x_cost_matrix=C,  # optional full C
    )

    coupling = out.matrix  # soft coupling (n x m)

    # 8) Hard assignment
    assignment = jnp.argmax(coupling, axis=1)

    return np.array(assignment), coupling
