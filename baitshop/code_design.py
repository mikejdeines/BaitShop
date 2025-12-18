import pandas as pd
import requests
import csv
import os
import numpy as np
import scipy.optimize as opt
from sklearn.metrics.pairwise import pairwise_distances
from typing import List, Tuple

def filter_hd_gt4(csv_path):
    """"
    Filter codewords from a CSV file to ensure pairwise Hamming distance >= 4.
    Args:
        csv_path: path to CSV file (may include ~)
    Returns:
        List of binary barcode strings after filtering
    """
    df = pd.read_csv(os.path.expanduser(csv_path), header=None)
    try:
        # Convert rows to frozensets
        sets = [frozenset(row.dropna().astype(int).tolist()) for _, row in df.iterrows()]
        if not sets:
            print("No rows found in CSV.")
            return df.iloc[[]]

        # Find common elements across all rows and remove them
        try:
            common_elements = set.intersection(*sets)
        except TypeError:
            # If only one set present, set.intersection expects iterable; handle gracefully
            common_elements = set()

        # Remove common elements from each set
        sets = [s - common_elements for s in sets]
    except ValueError as e:
        print("Error converting rows to integers:", e)
        return None


    keep = []
    for i, s in enumerate(sets):
        ok = True
        for j, t in enumerate(sets):
            if i == j:
                continue
            # intersection size < 2  <=> HD > 4
            if len(s & t) > 2:
                ok = False
                break
        if ok:
            keep.append(i)

    filtered_df = df.iloc[keep]

    # Determine the length of the largest value in the dataframe
    max_value = df.max().max()

    # Convert filtered rows to binary barcodes as strings
    def indices_to_binary_string(indices_list, length):
        """
        Converts a list of positive indices to a binary string where the specified indices are set to 1.
        The binary string will have a fixed length.
        """
        binary_list = ['0'] * length
        for index in indices_list:
            if index < 0:
                raise ValueError("Indices must be non-negative.")
            binary_list[index - 1] = '1'  # Shift to zero-indexing
        return ''.join(binary_list)

    binary_barcodes = [
        indices_to_binary_string(row.dropna().astype(int).tolist(), max_value) for _, row in filtered_df.iterrows()
    ]

    return binary_barcodes

def download_ljcr_csv(v, k, t, output_path):
    """Download codewords from the LJCR cover page and save as CSV.

    Args:
        v, k, t: parameters for the LJCR cover show page
        output_path: path to save CSV (may include ~)
    """
    # Expand user directory if ~ is used in the path
    output_path = os.path.expanduser(output_path)

    url = f"https://ljcr.dmgordon.org/cover/show_cover.php?v={v}&k={k}&t={t}"
    resp = requests.get(url)
    resp.raise_for_status()

    lines = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip header or non-data lines
        if not line[0].isdigit():
            continue
        parts = line.split()
        # Keep only rows with exactly k entries (k columns expected)
        if len(parts) == k:
            lines.append(parts)

    if not lines:
        raise ValueError(f"No codewords found for v={v}, k={k}, t={t}.")

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in lines:
            writer.writerow(row)

    print(f"Saved {len(lines)} codewords to {output_path}")






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

def assign_barcodes_optimal_transport(genes,
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
    if use_full_cost:
        print("Computing full cost tensor...")
        cost_tensor = compute_full_cost_matrix(correlation_matrix, working_distance_matrix)
        
        if method == 'simulated_annealing':
            assignment_working = solve_with_simulated_annealing(cost_tensor)
        else:
            # For full cost with Hungarian, we need to linearize
            cost_matrix = compute_linearized_cost_matrix(correlation_matrix, working_distance_matrix)
            assignment_working = solve_with_hungarian(cost_matrix)
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

def select_best_isoform(isoform_expression, gene_names: List[str],
                        appris_data, transcript_lengths):
    """
    Select the best isoform for each gene based on APPRIS data and expression levels.
    Args:
        isoform_expression: DataFrame with isoform expression data (rows: isoforms, columns: samples)
        gene_names: List of gene names in the panel
        appris_data: DataFrame with APPRIS annotations (columns: 'isoform', 'gene', 'appris_level')
        transcript_lengths: DataFrame with transcript lengths (columns: 'isoform', 'length')
    Returns:
        DataFrame with selected isoforms for each gene
    """
    selected_isoforms = []
    # Subset APPRIS data to PRINCIPAL:1 and PRINCIPAL:2
    appris_filtered = appris_data[appris_data['annotation_level'].isin(['PRINCIPAL:1', 'PRINCIPAL:2'])]
    for gene in gene_names:
        gene_isoforms = appris_filtered[appris_filtered['gene'] == gene]
        if len(gene_isoforms) == 1:
            selected_isoforms.append(gene_isoforms['isoform'].values[0])
        else:
            # Select isoform with highest mean expression
            isoform_means = {}
            for isoform in gene_isoforms['isoform']:
                if isoform in isoform_expression['Name'].values:
                    isoform_means[isoform] = isoform_expression[isoform_expression['Name'] == isoform]['TPM']
            if isoform_means:
                best_isoform = max(isoform_means, key=isoform_means.get)
                selected_isoforms.append(best_isoform)
            else:
                # Select longest isoform if no expression data
                lengths = transcript_lengths[transcript_lengths['isoform'].isin(gene_isoforms['isoform'])]
                if not lengths.empty:
                    longest_isoform = lengths.loc[lengths['length'].idxmax()]['isoform']
                    selected_isoforms.append(longest_isoform)
                else:
                    selected_isoforms.append(gene_isoforms['isoform'].values[0])  # Fallback
    isoform_df = pd.DataFrame({'gene': gene_names, 'selected_isoform': selected_isoforms})
    return isoform_df

def create_codebook(genes: List[str], barcodes: List[str], assignment: np.ndarray,
                    isoform_df: pd.DataFrame, readouts) -> pd.DataFrame:
    """
    Create a codebook DataFrame mapping genes to assigned barcodes and selected isoforms.
    Args:
        genes: List of gene names
        barcodes: List of barcode strings
        assignment: Array where assignment[i] is barcode index for gene i
        isoform_df: DataFrame with selected isoforms for each gene
    """
    codebook_rows = []
    readout_names = list(readouts.keys())
    for i, gene in enumerate(genes):
        barcode_index = assignment[i]
        barcode = barcodes[barcode_index]
        selected_isoform = isoform_df[isoform_df['gene'] == gene]['selected_isoform'].values[0]
        row = {
            'name': gene,
            'id': selected_isoform
        }
        for j, bit in enumerate(barcode):
            readout_name = readout_names[j]
            row[readout_name] = int(bit)
        codebook_rows.append(row)
    codebook_df = pd.DataFrame(codebook_rows)
    return codebook_df