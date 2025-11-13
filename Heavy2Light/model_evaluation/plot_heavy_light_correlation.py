"""
analyze and plot the correlation between heavy chain and true light chain
germline percent identities.
"""

# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, t
import numpy as np
import matplotlib.colors as mcolors

def load_data(json_file):
    """
    Load data from JSON file and extract sequence information.

    Args:
        json_file: Path to the JSON file

    Returns:
        DataFrame with heavy and light chain information
    """
    data = []

    with open(json_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            seq_id = entry['Sequence ID']

            # Extract percent identity from the Total field
            if 'Total' in entry and 'percent identity' in entry['Total']:
                percent_identity = entry['Total']['percent identity']

                data.append({
                    'sequence_id': seq_id,
                    'percent_identity': percent_identity
                })

    return pd.DataFrame(data)

def match_heavy_light_pairs(df):
    """
    Match heavy chains with their corresponding true light chains.

    Args:
        df: DataFrame with sequence information

    Returns:
        DataFrame with matched heavy-light pairs
    """
    # Separate heavy chains and true light chains
    heavy_chains = df[df['sequence_id'].str.startswith('heavy_chain_')].copy()
    true_light_chains = df[df['sequence_id'].str.startswith('true_light_chain_')].copy()

    # Extract the chain number from sequence IDs
    heavy_chains['chain_number'] = heavy_chains['sequence_id'].str.extract(r'heavy_chain_(\d+)')[0]
    true_light_chains['chain_number'] = true_light_chains['sequence_id'].str.extract(r'heavy_chain_(\d+)')[0]

    # Merge on chain number
    matched_pairs = pd.merge(
        heavy_chains[['chain_number', 'percent_identity']],
        true_light_chains[['chain_number', 'percent_identity']],
        on='chain_number',
        suffixes=('_heavy', '_light')
    )

    print(f"Found {len(matched_pairs)} matched heavy-light pairs")
    print(f"\nHeavy chain germline identity: {matched_pairs['percent_identity_heavy'].mean():.2f} ± {matched_pairs['percent_identity_heavy'].std():.2f}%")
    print(f"Light chain germline identity: {matched_pairs['percent_identity_light'].mean():.2f} ± {matched_pairs['percent_identity_light'].std():.2f}%")

    return matched_pairs

def match_heavy_gen_light_pairs(df_heavy, df_gen_light):
    """
    Match heavy chains with their corresponding generated light chains (first generated only).

    Args:
        df_heavy: DataFrame with heavy chain information
        df_gen_light: DataFrame with generated light chain information

    Returns:
        DataFrame with matched heavy-generated light pairs
    """
    # Separate heavy chains and generated light chains
    heavy_chains = df_heavy[df_heavy['sequence_id'].str.startswith('heavy_chain_')].copy()
    gen_light_chains = df_gen_light[df_gen_light['sequence_id'].str.startswith('gen_light_')].copy()

    # Extract the chain number from sequence IDs
    heavy_chains['chain_number'] = heavy_chains['sequence_id'].str.extract(r'heavy_chain_(\d+)')[0]
    gen_light_chains['chain_number'] = gen_light_chains['sequence_id'].str.extract(r'heavy_chain_(\d+)')[0]

    print(f"Heavy chains: {len(heavy_chains)}")
    print(f"Generated light chains: {len(gen_light_chains)}")

    # Merge on chain number
    matched_pairs = pd.merge(
        heavy_chains[['chain_number', 'percent_identity']],
        gen_light_chains[['chain_number', 'percent_identity']],
        on='chain_number',
        suffixes=('_heavy', '_light')
    )

    print(f"Found {len(matched_pairs)} matched heavy-generated light pairs")
    print(f"\nHeavy chain germline identity: {matched_pairs['percent_identity_heavy'].mean():.2f} ± {matched_pairs['percent_identity_heavy'].std():.2f}%")
    print(f"Generated light chain germline identity: {matched_pairs['percent_identity_light'].mean():.2f} ± {matched_pairs['percent_identity_light'].std():.2f}%")

    return matched_pairs

def match_random_heavy_light_pairs(df, seed=42):
    """
    Create random pairings of heavy chains and true light chains.

    Args:
        df: DataFrame with sequence information
        seed: Random seed for reproducibility

    Returns:
        DataFrame with randomly matched heavy-light pairs
    """
    # Separate heavy chains and true light chains
    heavy_chains = df[df['sequence_id'].str.startswith('heavy_chain_')].copy()
    true_light_chains = df[df['sequence_id'].str.startswith('true_light_chain_')].copy()

    print(f"Heavy chains: {len(heavy_chains)}")
    print(f"True light chains: {len(true_light_chains)}")

    # Randomly shuffle the light chains
    np.random.seed(seed)
    shuffled_light_chains = true_light_chains.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Take equal number of heavy and light chains
    n_pairs = min(len(heavy_chains), len(shuffled_light_chains))
    heavy_subset = heavy_chains.head(n_pairs).reset_index(drop=True)
    light_subset = shuffled_light_chains.head(n_pairs).reset_index(drop=True)

    # Create random pairs
    random_pairs = pd.DataFrame({
        'percent_identity_heavy': heavy_subset['percent_identity'].values,
        'percent_identity_light': light_subset['percent_identity'].values
    })

    print(f"Created {len(random_pairs)} random heavy-light pairs")
    print(f"\nHeavy chain germline identity: {random_pairs['percent_identity_heavy'].mean():.2f} ± {random_pairs['percent_identity_heavy'].std():.2f}%")
    print(f"Light chain germline identity: {random_pairs['percent_identity_light'].mean():.2f} ± {random_pairs['percent_identity_light'].std():.2f}%")

    return random_pairs

def darken_color(color, factor=0.7):
    """Darken a color by a given factor."""
    try:
        c = mcolors.to_rgb(color)
        return tuple([x * factor for x in c])
    except:
        return color

def plot_correlation(matched_pairs, output_file='heavy_light_germline_correlation.png',
                    title='Native', color='gray'):
    """
    Create a scatter plot with correlation analysis in the style of reference plots.

    Args:
        matched_pairs: DataFrame with matched heavy-light pairs
        output_file: Path to save the plot
        title: Title for the plot (e.g., 'Native', 'Random', 'Generated')
        color: Color for scatter points and regression line
    """
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(matched_pairs['percent_identity_heavy'],
                                     matched_pairs['percent_identity_light'])
    spearman_r, spearman_p = spearmanr(matched_pairs['percent_identity_heavy'],
                                        matched_pairs['percent_identity_light'])

    print(f"\nPearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4e}")
    print(f"Spearman correlation: r = {spearman_r:.4f}, p = {spearman_p:.4e}")

    # Create the plot with matching size from reference
    fig, ax = plt.subplots(figsize=(3, 3))

    # Scatter plot matching reference style
    scatter = ax.scatter(matched_pairs['percent_identity_heavy'],
                        matched_pairs['percent_identity_light'],
                        alpha=0.03, s=15, c=color, edgecolors='none')

    # Prepare data for regression
    x = matched_pairs['percent_identity_heavy'].values
    y = matched_pairs['percent_identity_light'].values

    ax.set_ylim(50, 100)
    ax.set_yticks([50,60,70,80,90,100])

    # Calculate regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = p(x_line)

    # Calculate confidence interval (95%)
    n = len(x)
    y_pred = p(x)
    residuals = y - y_pred
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    t_val = t.ppf(0.975, n - 2)  # 95% CI

    x_mean = np.mean(x)
    conf_interval = t_val * s_err * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))

    # Plot confidence interval
    ax.fill_between(x_line, y_line - conf_interval, y_line + conf_interval,
                    color=color, alpha=0.2)

    # Plot regression line with darker color
    dark_color = darken_color(color, factor=0.6)
    ax.plot(x_line, y_line, color=dark_color, alpha=1.0, linewidth=1.5)

    # Add labels matching reference style
    ax.set_xlabel('Heavy chain identity [%]', fontsize=11)
    ax.set_ylabel('Light chain identity [%]', fontsize=11)

    # Format p-value for display in scientific notation
    if pearson_p == 0.0 or pearson_p < 1e-300:
        # P-value underflowed to 0, display as extremely small
        p_display = "< 1e-300"
    elif pearson_p < 0.001:
        # Format as scientific notation with explicit -e notation
        exponent = int(np.floor(np.log10(pearson_p)))
        mantissa = pearson_p / (10 ** exponent)
        p_display = f"{mantissa:.1f}e{exponent}"
    else:
        p_display = f"{pearson_p:.3f}"

    # Add title and statistics text box - positioned at bottom-left with transparent background
    stats_text = f'{title}\nR: {pearson_r:.3f},\nP: {p_display}'
    ax.text(0.05, 0.05, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom')

    # Clean up plot - remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    return fig, ax

def main():
    """Main function to run the analysis."""
    # Load the data for true light chains
    print("="*60)
    print("ANALYZING TRUE LIGHT CHAINS")
    print("="*60)
    print("\nLoading data from JSON file...")
    json_file = 'data/matching_seqs_multiple_light_seqs_203276_cls_predictions.json'
    df = load_data(json_file)

    print(f"Loaded {len(df)} sequences")
    print(f"Unique sequence types:")
    for seq_type in df['sequence_id'].str.extract(r'^([a-z_]+)')[0].unique():
        count = df['sequence_id'].str.startswith(seq_type).sum()
        print(f"  {seq_type}: {count}")

    # Match heavy and true light chain pairs
    print("\n" + "="*60)
    print("Matching heavy and true light chain pairs...")
    print("="*60)
    matched_pairs_true = match_heavy_light_pairs(df)

    # Plot the correlation for true light chains
    print("\n" + "="*60)
    print("Creating correlation plot for true light chains...")
    print("="*60)
    plot_correlation(matched_pairs_true,
                    output_file='heavy_true_light_germline_correlation_smaller_unit.png',
                    title='Native',
                    color='#90EE90')  # Light green

    # Load the data for generated light chains
    print("\n\n" + "="*60)
    print("ANALYZING GENERATED LIGHT CHAINS")
    print("="*60)
    print("\nLoading data from JSON file...")
    gen_json_file = 'data/matching_seqs_only_first_gen_multiple_light_seqs_203276_cls_predictions.json'
    df_gen = load_data(gen_json_file)

    print(f"Loaded {len(df_gen)} sequences")
    print(f"Unique sequence types:")
    for seq_type in df_gen['sequence_id'].str.extract(r'^([a-z_]+)')[0].unique():
        count = df_gen['sequence_id'].str.startswith(seq_type).sum()
        print(f"  {seq_type}: {count}")

    # Match heavy and generated light chain pairs
    # Use heavy chains from the first file (df) and generated light chains from the second file (df_gen)
    print("\n" + "="*60)
    print("Matching heavy and generated light chain pairs...")
    print("="*60)
    matched_pairs_gen = match_heavy_gen_light_pairs(df, df_gen)

    # Plot the correlation for generated light chains
    print("\n" + "="*60)
    print("Creating correlation plot for generated light chains...")
    print("="*60)
    plot_correlation(matched_pairs_gen,
                    output_file='heavy_generated_light_germline_correlation_smaller_unit.png',
                    title='Generated',
                    color='#D95F5F')  # Salmon/red color matching reference

    # Create random pairings of heavy and true light chains
    print("\n\n" + "="*60)
    print("ANALYZING RANDOM PAIRINGS")
    print("="*60)
    print("\nCreating random pairings of heavy and true light chains...")
    matched_pairs_random = match_random_heavy_light_pairs(df)

    # Plot the correlation for random pairings
    print("\n" + "="*60)
    print("Creating correlation plot for random pairings...")
    print("="*60)
    plot_correlation(matched_pairs_random,
                    output_file='heavy_random_light_germline_correlation_smaller_unit.png',
                    title='Random',
                    color='gray')

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTrue light chains:")
    print(f"  - Number of pairs: {len(matched_pairs_true)}")
    print(f"  - Heavy chain identity: {matched_pairs_true['percent_identity_heavy'].mean():.2f} ± {matched_pairs_true['percent_identity_heavy'].std():.2f}%")
    print(f"  - Light chain identity: {matched_pairs_true['percent_identity_light'].mean():.2f} ± {matched_pairs_true['percent_identity_light'].std():.2f}%")

    print(f"\nGenerated light chains:")
    print(f"  - Number of pairs: {len(matched_pairs_gen)}")
    print(f"  - Heavy chain identity: {matched_pairs_gen['percent_identity_heavy'].mean():.2f} ± {matched_pairs_gen['percent_identity_heavy'].std():.2f}%")
    print(f"  - Light chain identity: {matched_pairs_gen['percent_identity_light'].mean():.2f} ± {matched_pairs_gen['percent_identity_light'].std():.2f}%")

    print(f"\nRandom pairings:")
    print(f"  - Number of pairs: {len(matched_pairs_random)}")
    print(f"  - Heavy chain identity: {matched_pairs_random['percent_identity_heavy'].mean():.2f} ± {matched_pairs_random['percent_identity_heavy'].std():.2f}%")
    print(f"  - Light chain identity: {matched_pairs_random['percent_identity_light'].mean():.2f} ± {matched_pairs_random['percent_identity_light'].std():.2f}%")

    print("\nAnalysis complete!")
    print(f"\nGenerated plots:")
    print(f"  1. heavy_true_light_germline_correlation.png")
    print(f"  2. heavy_generated_light_germline_correlation.png")
    print(f"  3. heavy_random_light_germline_correlation.png")

if __name__ == "__main__":
    main()
