import pandas as pd
import requests
import csv
import os
import numpy as np
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


def select_best_isoform(isoform_expression, gene_names: List[str],
                        appris_data, transcript_lengths):
    """
    Select the best isoform for each gene based on APPRIS data and expression levels.
    Args:
        isoform_expression: DataFrame with isoform expression data (rows: isoforms, columns: samples)
        gene_names: List of gene names in the panel
        appris_data: DataFrame with APPRIS annotations (columns: 'isoform', 'gene', 'appris_level')
        transcript_lengths: DataFrame with transcript lengths (columns: 'isoform', 'gene', 'length')
    Returns:
        DataFrame with selected isoforms for each gene
    """
    selected_isoforms = []
    # Subset APPRIS data to PRINCIPAL:1 and PRINCIPAL:2
    appris_filtered = appris_data[appris_data['annotation_level'].isin(['PRINCIPAL:1', 'PRINCIPAL:2'])]
    for gene in gene_names:
        # Use transcript_lengths to inform possible transcripts for this gene
        gene_transcripts = transcript_lengths[transcript_lengths['gene'] == gene]['isoform'].values if 'gene' in transcript_lengths.columns else []
        
        # Filter APPRIS data to only include transcripts present in transcript_lengths
        if len(gene_transcripts) > 0:
            gene_isoforms = appris_filtered[(appris_filtered['gene'] == gene) & 
                                           (appris_filtered['isoform'].isin(gene_transcripts))]
        else:
            gene_isoforms = appris_filtered[appris_filtered['gene'] == gene]
        
        # If no APPRIS isoforms found, fall back to all transcripts from transcript_lengths
        candidate_isoforms = gene_isoforms['isoform'].values if len(gene_isoforms) > 0 else gene_transcripts
        
        if len(candidate_isoforms) == 1:
            selected_isoforms.append(candidate_isoforms[0])
        elif len(candidate_isoforms) > 1:
            # Select isoform with highest mean expression
            isoform_means = {}
            for isoform in candidate_isoforms:
                if isoform in isoform_expression['Name'].values:
                    isoform_means[isoform] = isoform_expression[isoform_expression['Name'] == isoform]['TPM'].mean()
            if isoform_means:
                best_isoform = max(isoform_means, key=isoform_means.get)
                selected_isoforms.append(best_isoform)
            else:
                # Select longest isoform if no expression data
                lengths = transcript_lengths[transcript_lengths['isoform'].isin(candidate_isoforms)]
                if not lengths.empty:
                    longest_isoform = lengths.loc[lengths['length'].idxmax()]['isoform']
                    selected_isoforms.append(longest_isoform)
                else:
                    selected_isoforms.append(candidate_isoforms[0])  # Fallback
        else:
            # No candidates found at all - should not happen, but handle gracefully
            if len(gene_transcripts) > 0:
                selected_isoforms.append(gene_transcripts[0])
            else:
                raise ValueError(f"No isoforms found for gene {gene}")
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