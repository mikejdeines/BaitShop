from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import gc_fraction
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO import write
import pandas as pd
import random
import re
import math
from seqfold import dg
from concurrent.futures import ThreadPoolExecutor


def filter_trna_rrna(input_fasta, output_fasta):
    """
    Filter tRNA, rRNA, mt-tRNA, and mt-rRNA sequences from the input FASTA file and save them to a new FASTA file.

    Args:
        input_fasta (str): Path to the input FASTA file.
        output_fasta (str): Path to the output FASTA file containing only tRNA and rRNA sequences.
    """
    with open(output_fasta, "w") as out_fasta:
        for record in SeqIO.parse(input_fasta, "fasta"):
            desc = record.description
            # Check for tRNA, rRNA, mt-tRNA, or mt-rRNA in the description
            if any(x in desc for x in ["tRNA", "rRNA", "mt-tRNA", "mt-rRNA"]):
                SeqIO.write(record, out_fasta, "fasta")
    print(f"Filtered tRNA, rRNA, mt-tRNA, and mt-rRNA sequences saved to {output_fasta}")

def generate_probes_dict_by_gene(codebook, sequences, overlap=0):
    """
    Generate a dictionary of probes for all genes in the codebook.

    Args:
        codebook (pd.DataFrame): DataFrame containing transcript and gene information.
        sequences (dict): Dictionary of transcript sequences.
        overlap (int): Overlap between consecutive probes.

    Returns:
        dict: Dictionary with gene symbols as keys and lists of probes as values.
    """
    probes_dict = {}
    
    for gene_symbol in codebook['name'].unique():  # Iterate over unique gene symbols
        transcripts = codebook[codebook['name'] == gene_symbol]  # Filter transcripts by gene symbol
        
        if transcripts.empty:
            print(f"Warning: Gene symbol {gene_symbol} not found in the codebook.")
            continue
        
        probes_dict[gene_symbol] = []
        for _, row in transcripts.iterrows():
            transcript_id = row['id']  # Assuming the column containing transcript IDs is named 'id'
            if transcript_id in sequences:  # Check if the transcript ID exists in the .fasta sequences
                sequence = sequences[transcript_id]
                # Generate 30 nt regions with the specified overlap
                regions = [sequence[i:i+30] for i in range(0, len(sequence) - 29, 30 - overlap)]
                # Append each region with transcript_id information
                probes_dict[gene_symbol].extend([{'transcript_id': transcript_id, 'sequence': region} for region in regions])
            else:
                print(f"Warning: Transcript ID {transcript_id} not found in the .fasta file.")
    return probes_dict

def calculate_gc_and_tm(probes_dict):
    """
    Calculate GC fraction and Tm for each probe in the probes_dict.

    Args:
        probes_dict (dict): Dictionary of probes per gene.

    Returns:
        dict: Updated probes_dict with GC_fraction and Tm added to each probe.
    """
    updated_probes_dict = {}
    for gene_symbol, probes in probes_dict.items():
        updated_probes_dict[gene_symbol] = []
        for probe_info in probes:
            sequence = probe_info['sequence']
            gc = gc_fraction(sequence)
            tm = mt.Tm_NN(sequence, nn_table=mt.DNA_NN4,
                          dnac1 = 5, # Oligo concentration in nM
                          dnac2 = 0,
                          Na = 300 # Na+ concentration for 2xSSC
                          )
            # Add GC_fraction and Tm to the probe info
            updated_probe_info = probe_info.copy()
            updated_probe_info['GC_fraction'] = gc
            updated_probe_info['Tm'] = tm
            updated_probes_dict[gene_symbol].append(updated_probe_info)
    return updated_probes_dict

def filter_probes_by_homopolymer(probes_dict, max_homopolymer_length=5):
    """
    Remove probes with a homopolymeric run of more than `max_homopolymer_length` bases.

    Args:
        probes_dict (dict): Dictionary of probes per gene.
        max_homopolymer_length (int): Maximum allowed length of homopolymeric runs.

    Returns:
        dict: Filtered probes dictionary.
    """
    filtered_probes_dict = {}
    homopolymer_pattern = re.compile(f"(A{{{max_homopolymer_length + 1},}}|T{{{max_homopolymer_length + 1},}}|C{{{max_homopolymer_length + 1},}}|G{{{max_homopolymer_length + 1},}})")

    for gene_symbol, probes in probes_dict.items():
        started = len(probes)
        filtered_probes = [
            probe_info for probe_info in probes if not homopolymer_pattern.search(probe_info['sequence'])
        ]
        retained = len(filtered_probes)
        filtered_probes_dict[gene_symbol] = filtered_probes
        print(f"{gene_symbol}: {retained}/{started} probes retained with homopolymeric runs of length <= {max_homopolymer_length}")
    
    return filtered_probes_dict

def filter_probes_by_gc_and_tm(probes_dict, min_gc=0.3, max_gc=0.7, min_tm=61, max_tm=81):
    """
    Remove probes with 0.3 < GC_fraction < 0.7 and 61 < Tm < 81.
    Also returns the number of probes retained and started for each gene, and prints the stats.

    Args:
        probes_dict (dict): Dictionary of probes with GC_fraction and Tm values.
        min_gc (float): Minimum GC fraction threshold.
        max_gc (float): Maximum GC fraction threshold.
        min_tm (float): Minimum Tm threshold.
        max_tm (float): Maximum Tm threshold.

    Returns:
        tuple: (filtered_probes_dict, stats_dict)
            filtered_probes_dict: Filtered probes dictionary.
            stats_dict: Dictionary {gene_symbol: (retained, started)}
    """
    filtered_probes_dict = {}
    for gene_symbol, probes in probes_dict.items():
        started = len(probes)
        filtered_probes = [
            probe_info for probe_info in probes
            if (min_gc < probe_info['GC_fraction'] < max_gc and min_tm < probe_info['Tm'] < max_tm)
        ]
        retained = len(filtered_probes)
        filtered_probes_dict[gene_symbol] = filtered_probes
        print(f"{gene_symbol}: {retained}/{started} probes retained with {min_gc} < GC < {max_gc} and {min_tm} < Tm < {max_tm}")
    return filtered_probes_dict

def extract_15mers_from_trna_rrna(trna_rrna_fasta):
    """
    Extract all 15 nt sequences from the tRNA/rRNA FASTA file.

    Args:
        trna_rrna_fasta (str): Path to the tRNA/rRNA FASTA file.

    Returns:
        set: A set of all unique 15 nt sequences.
    """
    kmer_set = set()
    for record in SeqIO.parse(trna_rrna_fasta, "fasta"):
        sequence = str(record.seq)
        # Extract all 15 nt substrings (sliding window)
        for i in range(len(sequence) - 14):
            kmer_set.add(sequence[i:i+15])
    return kmer_set

# Function to filter probes with homology to any 15 nt sequence in tRNA/rRNA
def filter_probes_by_trna_rrna_homology(probes_dict, trna_rrna_kmers):
    """
    Remove probes with homology to any 15 nt sequence in the tRNA/rRNA sequences.

    Args:
        probes_dict (dict): Dictionary of probes per gene.
        trna_rrna_kmers (set): Set of 15 nt sequences from tRNA/rRNA.

    Returns:
        dict: Filtered probes dictionary.
    """
    filtered_probes_dict = {}
    for gene_symbol, probes in probes_dict.items():
        started = len(probes)
        filtered_probes_dict[gene_symbol] = []
        for probe_info in probes:
            sequence = probe_info['sequence']
            # Check for homology with any 15 nt sequence in tRNA/rRNA
            has_homology = any(sequence[i:i+15] in trna_rrna_kmers for i in range(len(sequence) - 14))
            if not has_homology:
                filtered_probes_dict[gene_symbol].append(probe_info)
        retained = len(filtered_probes_dict[gene_symbol])
        print(f"{gene_symbol}: {retained}/{started} probes retained after filtering by tRNA/rRNA homology")
    return filtered_probes_dict

def extract_17mers_by_gene(sequences, codebook):
    """
    Extract all 17 nt sequences from cDNA sequences grouped by gene, excluding the gene itself.

    Args:
        sequences (dict): Dictionary of transcript sequences (key: transcript ID, value: sequence).
        codebook (pd.DataFrame): DataFrame containing transcript and gene information.

    Returns:
        dict: A dictionary where keys are gene symbols and values are sets of 17-mers from other genes.
    """
    gene_to_17mers = {}
    transcript_to_gene = {row['id']: row['name'] for _, row in codebook.iterrows()}  # Map transcript ID to gene name

    for transcript_id, sequence in sequences.items():
        gene_name = transcript_to_gene.get(transcript_id)
        if not gene_name:
            continue  # Skip if transcript ID is not in the codebook

        if gene_name not in gene_to_17mers:
            gene_to_17mers[gene_name] = set()

        # Add 17-mers from this sequence to all other genes
        for i in range(len(sequence) - 16):
            kmer = sequence[i:i+17]
            for other_gene in gene_to_17mers:
                if other_gene != gene_name:
                    gene_to_17mers[other_gene].add(kmer)

    return gene_to_17mers

def filter_probes_by_homology_threshold(probes_dict, gene_to_17mers, homology_threshold=0.8):
    """
    Filter probes with more than a specified homology threshold to any 17-mer of cDNA from other genes,
    checking both the probe and its reverse complement.

    Args:
        probes_dict (dict): Dictionary of probes per gene.
        gene_to_17mers (dict): Dictionary mapping gene symbols to sets of 17-mers from other genes.
        homology_threshold (float): Maximum allowed homology (fraction) to 17-mers.

    Returns:
        dict: Filtered probes dictionary.
    """
    def calculate_homology(seq1, seq2):
        """Calculate the fraction of matching bases between two sequences."""
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)

    filtered_probes_dict = {}
    for gene_symbol, probes in probes_dict.items():
        started = len(probes)
        cdna_kmers = gene_to_17mers.get(gene_symbol, set())

        # Precompute a dictionary of 17-mers grouped by their hash
        kmer_hash_map = {}
        for kmer in cdna_kmers:
            kmer_hash = hash(kmer)
            kmer_hash_map.setdefault(kmer_hash, []).append(kmer)

        filtered_probes_dict[gene_symbol] = []
        for probe_info in probes:
            sequence = probe_info['sequence']
            rc_sequence = str(Seq(sequence).reverse_complement())
            has_high_homology = False

            # Check both sequence and its reverse complement
            for seq_variant in (sequence, rc_sequence):
                for i in range(len(seq_variant) - 16):
                    probe_kmer = seq_variant[i:i+17]
                    probe_kmer_hash = hash(probe_kmer)
                    if probe_kmer_hash in kmer_hash_map:
                        for kmer in kmer_hash_map[probe_kmer_hash]:
                            if calculate_homology(probe_kmer, kmer) > homology_threshold:
                                has_high_homology = True
                                break
                    if has_high_homology:
                        break
                if has_high_homology:
                    break

            if not has_high_homology:
                filtered_probes_dict[gene_symbol].append(probe_info)
        retained = len(filtered_probes_dict[gene_symbol])
        print(f"{gene_symbol}: {retained}/{started} probes retained after filtering by homology threshold {homology_threshold}")
    return filtered_probes_dict

def select_probes_greedy(probes_dict, codebook, num_probes=64, target_gc=0.5, target_tm=81):
    """
    Select a specified number of probes for each gene using a greedy algorithm.
    The selection prioritizes the lowest score probe first, then calculates the overlap
    that all other probes would have, and selects the lowest score probe with the lowest overlap.

    Args:
        probes_dict (dict): Dictionary of probes per gene.
        codebook (pd.DataFrame): DataFrame containing transcript and gene information.
        num_probes (int): Number of probes to select for each gene.
        target_gc (float): Target GC fraction.
        target_tm (float): Target melting temperature (Tm).

    Returns:
        dict: Dictionary with selected probes for each gene.
    """
    selected_probes_dict = {}
    valid_transcript_ids = set(codebook['id'])  # Extract valid transcript IDs from the codebook

    for gene_symbol, probes in probes_dict.items():
        # Calculate a score for each probe based on closeness to target GC_fraction, Tm, and transcript_id membership
        for probe in probes:
            gc_score = abs(probe['GC_fraction'] - target_gc) * 20 # Scale GC score relative to max difference in Tm
            tm_score = abs(probe['Tm'] - target_tm)
            transcript_penalty = 100 if probe['transcript_id'] not in valid_transcript_ids else 0
            probe['score'] = gc_score + tm_score + transcript_penalty  # Lower score is better

        # Initialize selected probes and sequences
        selected_probes = []
        selected_sequences = set()  # To track selected sequences and minimize overlap

        while len(selected_probes) < num_probes and probes:
            # Sort probes by score (ascending)
            probes = sorted(probes, key=lambda x: x['score'])

            # Select the probe with the lowest score
            best_probe = probes[0]
            best_probe_overlap = 0

            # Calculate overlap for all other probes
            for probe in probes:
                sequence = probe['sequence']
                overlap_count = sum(
                    1 for i in range(len(sequence) - 14)
                    if sequence[i:i+15] in selected_sequences
                )
                probe['overlap'] = overlap_count

            # Find the probe with the lowest overlap among the remaining probes
            best_probe = min(probes, key=lambda x: (x['overlap'], x['score']))
            best_probe_overlap = best_probe['overlap']

            # Add the selected probe to the list
            selected_probes.append(best_probe)
            sequence = best_probe['sequence']
            selected_sequences.update(sequence[i:i+15] for i in range(len(sequence) - 14))

            # Remove the selected probe from the list of remaining probes
            probes.remove(best_probe)

        # Store the selected probes for the current gene
        selected_probes_dict[gene_symbol] = selected_probes

    return selected_probes_dict

def filter_probes_by_deltaG_parallel(probes_dict, min_deltaG=0, temperature=76, num_workers=8):
    """
    Filter probes with deltaG greater than the specified min_deltaG in parallel.

    Args:
        probes_dict (dict): Dictionary of probes per gene.
        min_deltaG (float): Minimum allowed deltaG value.
        temperature (float): Temperature in °C for deltaG calculation.

    Returns:
        dict: Filtered probes dictionary.
    """
    def filter_gene_probes(gene_symbol, probes):
        filtered = [
            probe_info for probe_info in probes
            if dg(probe_info['sequence'], temperature) > min_deltaG
        ]
        print(f"{gene_symbol}: {len(filtered)}/{len(probes)} probes retained after deltaG filtering")
        return gene_symbol, filtered

    filtered_probes_dict = {}
    with ThreadPoolExecutor(max_workers = num_workers) as executor:
        futures = [executor.submit(filter_gene_probes, gene_symbol, probes) for gene_symbol, probes in probes_dict.items()]
        for future in futures:
            gene_symbol, filtered_probes = future.result()
            filtered_probes_dict[gene_symbol] = filtered_probes

    return filtered_probes_dict

def filter_self_and_cross_complementary_probes(probes_dict, min_complement_length=8):
    """
    Remove probes that are self-complementary or complementary to any other probe in the same gene set.
    A probe is filtered if it contains a region of at least min_complement_length bases that is the reverse complement
    of any region in itself (self-complementary) or in another probe (cross-complementary).

    Args:
        probes_dict (dict): Dictionary of probes per gene.
        min_complement_length (int): Minimum length of complementarity to consider.

    Returns:
        dict: Filtered probes dictionary.
    """
    filtered_probes_dict = {}

    for gene, probes in probes_dict.items():
        sequences = [probe['sequence'] for probe in probes]
        # Precompute all k-mers for all probes for cross-complementarity
        kmer_to_indices = {}
        for idx, seq in enumerate(sequences):
            for i in range(len(seq) - min_complement_length + 1):
                kmer = seq[i:i+min_complement_length]
                kmer_to_indices.setdefault(kmer, set()).add(idx)

        filtered_probes = []
        for idx, probe in enumerate(probes):
            seq = probe['sequence']
            rc_seq = str(Seq(seq).reverse_complement())

            # Self-complementarity check (use set for fast lookup)
            self_kmers = set(seq[i:i+min_complement_length] for i in range(len(seq) - min_complement_length + 1))
            self_comp = any(
                rc_seq[i:i+min_complement_length] in self_kmers
                for i in range(len(rc_seq) - min_complement_length + 1)
            )
            if self_comp:
                continue  # Remove self-complementary probe

            # Cross-complementarity check (use precomputed kmer_to_indices)
            cross_comp = False
            for i in range(len(rc_seq) - min_complement_length + 1):
                kmer = rc_seq[i:i+min_complement_length]
                indices = kmer_to_indices.get(kmer, set())
                if any(j != idx for j in indices):
                    cross_comp = True
                    break
            if cross_comp:
                continue  # Remove cross-complementary probe

            filtered_probes.append(probe)
        print(f"{gene}: {len(filtered_probes)}/{len(probes)} probes retained after self/cross complementarity filtering")
        filtered_probes_dict[gene] = filtered_probes

    return filtered_probes_dict

def assign_readouts_to_probes(selected_probes, codebook, readouts, num_readouts=2):
    """
    Assign readouts to probes ensuring balanced distribution across all probes for each gene.
    
    This function implements an improved algorithm that ensures:
    1. Each readout is assigned equally across all probes for a gene (round-robin allocation)
    2. No probe receives the same readout more than once
    3. The same readout is not assigned to the same gene more than its balanced quota
    
    The algorithm works by:
    - Calculating a balanced quota for each readout based on total assignments needed
    - Using a round-robin approach to systematically assign readouts to probes
    - Ensuring even distribution by cycling through available readouts
    
    Args:
        selected_probes (dict): Dictionary with gene names as keys and lists of probe 
                               dictionaries as values. Each probe dictionary will be 
                               modified to include a 'readouts' key.
        codebook (pd.DataFrame): DataFrame containing gene information with readout 
                                availability (1/0 values for each readout column).
        readouts (list): List of readout column names available for assignment.
        num_readouts (int): Number of readouts to assign to each probe.
    
    Returns:
        dict: The same selected_probes dictionary with readouts assigned to each probe.
    
    Raises:
        ValueError: If there are not enough unique readouts available for a gene.
    """
    for gene, probes in selected_probes.items():
        # Find available readouts for this gene
        readout_candidates = [
            readout for readout in readouts
            if codebook.loc[codebook['name'] == gene, readout].values[0] == 1
        ]
        n_probes = len(probes)
        n_readouts = len(readout_candidates)
        total_assignments = n_probes * num_readouts

        # Defensive: ensure enough unique readouts for assignment
        if n_readouts < num_readouts:
            raise ValueError(f"Not enough unique readouts for gene {gene} (needed {num_readouts}, got {n_readouts})")

        # Calculate how many times each readout should be assigned (balanced quota)
        base_count = total_assignments // n_readouts
        extra = total_assignments % n_readouts

        # Create a balanced assignment schedule using round-robin allocation
        # Each readout gets assigned base_count times, with the first 'extra' readouts getting one additional assignment
        readout_quotas = {}
        for i, readout in enumerate(readout_candidates):
            readout_quotas[readout] = base_count + (1 if i < extra else 0)

        # Track how many times each readout has been assigned
        readout_usage = {readout: 0 for readout in readout_candidates}
        
        # Assign readouts to probes using round-robin approach
        for probe in probes:
            probe['readouts'] = []
            probe_assigned = set()  # Track readouts already assigned to this probe
            
            # For each readout slot in this probe
            for readout_slot in range(num_readouts):
                # Find the best readout to assign for this slot
                best_readout = None
                min_usage = float('inf')
                
                # Look for the readout with lowest usage that hasn't been assigned to this probe yet
                for readout in readout_candidates:
                    if (readout not in probe_assigned and 
                        readout_usage[readout] < readout_quotas[readout] and
                        readout_usage[readout] < min_usage):
                        best_readout = readout
                        min_usage = readout_usage[readout]
                
                # If no readout found with minimum usage, find any available readout
                if best_readout is None:
                    for readout in readout_candidates:
                        if (readout not in probe_assigned and 
                            readout_usage[readout] < readout_quotas[readout]):
                            best_readout = readout
                            break
                
                # Assign the selected readout
                if best_readout is not None:
                    probe['readouts'].append(best_readout)
                    probe_assigned.add(best_readout)
                    readout_usage[best_readout] += 1
                else:
                    raise ValueError(f"Cannot assign readout to probe in gene {gene} - insufficient readouts or quota exceeded")

        # Defensive: verify all readouts were assigned according to their quotas
        for readout, used in readout_usage.items():
            expected = readout_quotas[readout]
            if used != expected:
                raise ValueError(f"Readout assignment mismatch for {readout} in gene {gene}: used {used}, expected {expected}")

    return selected_probes

def export_probes_to_fasta(selected_probes, readouts, fprimer, rprimer, output_fasta_file="Test_probes.fasta"):
    """
    Export selected probes with primers and readouts to a FASTA file.

    Args:
        selected_probes (dict): Dictionary of selected probes per gene.
        readouts (dict): Dictionary of readout sequences.
        fprimer (SeqRecord): Forward primer as a SeqRecord.
        rprimer (SeqRecord): Reverse primer as a SeqRecord.
        output_fasta_file (str): Output FASTA file name.
    """
    records = []
    for gene, probes in selected_probes.items():
        for probe in probes:
            readout_seq = "".join(readouts[readout] for readout in probe['readouts'])
            readout_rc_seq = "".join(str(Seq(readouts[readout]).reverse_complement()) for readout in probe['readouts'])
            sequence = fprimer.seq + Seq(probe['sequence']) + Seq(readout_seq) + Seq(readout_rc_seq) + rprimer.seq
            header = f"{gene}|{probe['transcript_id']}|{'-'.join(probe['readouts'])}"
            record = SeqRecord(
                seq=sequence,
                id=header,
                description=""
            )
            records.append(record)
    with open(output_fasta_file, "w") as fasta_out:
        write(records, fasta_out, "fasta")
    print(f"Probes with primers and readouts exported to {output_fasta_file}")
