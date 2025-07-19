import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Tuple, Optional

class SimpleVGeneAnalyzer:
    """
    A class to analyze V gene families from sequence data without count information.
    Each sequence is counted as 1 occurrence.
    """
    
    def __init__(self, json_file_path: str):
        """
        Initialize the analyzer with a JSON file containing sequence data.
        
        Args:
            json_file_path (str): Path to the JSON file with sequence data
        """
        self.json_file_path = json_file_path
        self.data = []
        self.load_data()
    
    def load_data(self):
        """Load data from the JSON file (one JSON object per line)."""
        try:
            with open(self.json_file_path, 'r') as file:
                for line in file:
                    if line.strip():  # Skip empty lines
                        self.data.append(json.loads(line.strip()))
            print(f"Loaded {len(self.data)} sequences from {self.json_file_path}")
        except FileNotFoundError:
            print(f"Error: File {self.json_file_path} not found.")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    
    def extract_v_gene_info(self, hits: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract V gene family and full gene name from hits using the first hit.
        
        Args:
            hits (List[Dict]): List of gene hits
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (gene_family, full_gene_name)
        """
        if not hits:
            return None, None
        
        # Use the first hit (highest bit score)
        first_hit = hits[0]
        gene_name = first_hit.get('gene', '')
        
        # Extract gene family (e.g., "IGKV2" from "IGKV2-28*01")
        family_match = re.match(r'(IG[HKL]V\d+)', gene_name)
        gene_family = family_match.group(1) if family_match else None
        
        # Extract full gene name (e.g., "IGKV2-28*01" from "IGKV2-28*01 unnamed protein product")
        gene_match = re.match(r'([^\\s]+)', gene_name)
        full_gene = gene_match.group(1) if gene_match else None
        
        return gene_family, full_gene
    
    def analyze_v_gene_families(self) -> pd.DataFrame:
        """
        Analyze V gene family distribution.
        
        Returns:
            pd.DataFrame: V gene family analysis with counts and percentages
        """
        family_counts = Counter()
        gene_counts = Counter()
        total_sequences = len(self.data)
        
        for entry in self.data:
            hits = entry.get('Hits', [])
            gene_family, full_gene = self.extract_v_gene_info(hits)
            
            if gene_family:
                family_counts[gene_family] += 1
            if full_gene:
                gene_counts[full_gene] += 1
        
        # Create family summary
        family_data = []
        for family, count in family_counts.most_common():
            percentage = (count / total_sequences) * 100
            family_data.append({
                'V_Gene_Family': family,
                'Count': count,
                'Percentage': round(percentage, 2),
                'Total_Sequences': total_sequences
            })
        
        return pd.DataFrame(family_data)
    
    def analyze_v_genes(self) -> pd.DataFrame:
        """
        Analyze individual V gene distribution.
        
        Returns:
            pd.DataFrame: Individual V gene analysis with counts and percentages
        """
        gene_counts = Counter()
        total_sequences = len(self.data)
        
        results = []
        for entry in self.data:
            sequence_id = entry.get('Sequence ID', '')
            hits = entry.get('Hits', [])
            gene_family, full_gene = self.extract_v_gene_info(hits)
            
            if full_gene:
                gene_counts[full_gene] += 1
            
            results.append({
                'Sequence_ID': sequence_id,
                'V_Gene_Family': gene_family,
                'V_Gene': full_gene,
                'Bit_Score': hits[0].get('bit_score') if hits else None,
                'E_Value': hits[0].get('e_value') if hits else None,
                'Sequence_Length': entry.get('Sequence Length')
            })
        
        # Create gene summary
        gene_data = []
        for gene, count in gene_counts.most_common():
            percentage = (count / total_sequences) * 100
            # Extract family from gene name
            family_match = re.match(r'(IG[HKL]V\\d+)', gene)
            family = family_match.group(1) if family_match else 'Unknown'
            
            gene_data.append({
                'V_Gene': gene,
                'V_Gene_Family': family,
                'Count': count,
                'Percentage': round(percentage, 2),
                'Total_Sequences': total_sequences
            })
        
        return pd.DataFrame(gene_data)
    
    def get_summary_stats(self) -> Dict:
        """
        Get overall summary statistics.
        
        Returns:
            Dict: Summary statistics
        """
        family_df = self.analyze_v_gene_families()
        gene_df = self.analyze_v_genes()
        
        return {
            'total_sequences': len(self.data),
            'unique_families': len(family_df),
            'unique_genes': len(gene_df),
            'most_common_family': family_df.iloc[0]['V_Gene_Family'] if len(family_df) > 0 else 'None',
            'most_common_family_percentage': family_df.iloc[0]['Percentage'] if len(family_df) > 0 else 0,
            'most_common_gene': gene_df.iloc[0]['V_Gene'] if len(gene_df) > 0 else 'None',
            'most_common_gene_percentage': gene_df.iloc[0]['Percentage'] if len(gene_df) > 0 else 0
        }
    
    def plot_family_distribution(self, top_n: int = 10, figsize: tuple = (12, 8), 
                                title_fontsize: int = 14, label_fontsize: int = 12, 
                                tick_fontsize: int = 10, bar_label_fontsize: int = 9,
                                bar_color: str = 'steelblue', show_both_metrics: bool = True,
                                save_path: str = None, dpi: int = 300):
        """
        Plot V gene family distribution with customizable styling.
        
        Args:
            top_n (int): Number of top families to show
            figsize (tuple): Figure size (width, height)
            title_fontsize (int): Font size for titles
            label_fontsize (int): Font size for axis labels
            tick_fontsize (int): Font size for tick labels
            bar_label_fontsize (int): Font size for count/percentage labels on bars
            bar_color (str): Color for bars (matplotlib color name or hex)
            show_both_metrics (bool): If True, show counts as bars with percentages as labels
            save_path (str): Path to save the plot (e.g., 'my_plot.png', 'plots/family_dist.pdf')
            dpi (int): Resolution for saved plot (dots per inch)
        """
        family_df = self.analyze_v_gene_families()
        top_families = family_df.head(top_n)
        
        if show_both_metrics:
            # Single plot with counts as bars and percentages as labels
            plt.figure(figsize=figsize)
            
            bars = plt.bar(top_families['V_Gene_Family'], top_families['Count'], 
                          color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            plt.title(f'Top {top_n} V Gene Families Distribution', fontsize=title_fontsize, pad=20)
            plt.xlabel('V Gene Family', fontsize=label_fontsize)
            plt.ylabel('Count', fontsize=label_fontsize)
            plt.xticks(rotation=45, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add combined labels (count and percentage) on bars
            max_count = max(top_families['Count'])
            for i, (family, count, pct) in enumerate(zip(top_families['V_Gene_Family'], 
                                                        top_families['Count'], 
                                                        top_families['Percentage'])):
                plt.text(i, count + max_count * 0.01, f'{count}\n({pct}%)', 
                        ha='center', va='bottom', fontsize=bar_label_fontsize, fontweight='bold')
            
            # Add some padding to the top
            plt.ylim(0, max_count * 1.15)
            
        else:
            # Two subplots as before
            plt.figure(figsize=figsize)
            
            # Plot 1: Counts
            plt.subplot(2, 1, 1)
            plt.bar(top_families['V_Gene_Family'], top_families['Count'], 
                    color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.title(f'Top {top_n} V Gene Families by Count', fontsize=title_fontsize)
            plt.xlabel('V Gene Family', fontsize=label_fontsize)
            plt.ylabel('Count', fontsize=label_fontsize)
            plt.xticks(rotation=45, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add count labels on bars
            for i, (family, count) in enumerate(zip(top_families['V_Gene_Family'], top_families['Count'])):
                plt.text(i, count + max(top_families['Count']) * 0.01, str(count), 
                        ha='center', va='bottom', fontsize=bar_label_fontsize)
            
            # Plot 2: Percentages
            plt.subplot(2, 1, 2)
            plt.bar(top_families['V_Gene_Family'], top_families['Percentage'], 
                    color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.title(f'Top {top_n} V Gene Families by Percentage', fontsize=title_fontsize)
            plt.xlabel('V Gene Family', fontsize=label_fontsize)
            plt.ylabel('Percentage (%)', fontsize=label_fontsize)
            plt.xticks(rotation=45, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add percentage labels on bars
            for i, (family, pct) in enumerate(zip(top_families['V_Gene_Family'], top_families['Percentage'])):
                plt.text(i, pct + max(top_families['Percentage']) * 0.01, f'{pct}%', 
                        ha='center', va='bottom', fontsize=bar_label_fontsize)
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_gene_distribution(self, top_n: int = 15, figsize: tuple = (15, 8), 
                              title_fontsize: int = 14, label_fontsize: int = 12, 
                              tick_fontsize: int = 10, bar_label_fontsize: int = 8,
                              bar_color: str = 'darkgreen', show_both_metrics: bool = True,
                              save_path: str = None, dpi: int = 300):
        """
        Plot individual V gene distribution with customizable styling.
        
        Args:
            top_n (int): Number of top genes to show
            figsize (tuple): Figure size (width, height)
            title_fontsize (int): Font size for titles
            label_fontsize (int): Font size for axis labels
            tick_fontsize (int): Font size for tick labels
            bar_label_fontsize (int): Font size for count/percentage labels on bars
            bar_color (str): Color for bars (matplotlib color name or hex)
            show_both_metrics (bool): If True, show counts as bars with percentages as labels
            save_path (str): Path to save the plot (e.g., 'my_plot.png', 'plots/gene_dist.pdf')
            dpi (int): Resolution for saved plot (dots per inch)
        """
        gene_df = self.analyze_v_genes()
        top_genes = gene_df.head(top_n)
        
        if show_both_metrics:
            # Single plot with counts as bars and percentages as labels
            plt.figure(figsize=figsize)
            
            bars = plt.bar(range(len(top_genes)), top_genes['Count'], 
                          color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            plt.title(f'Top {top_n} V Genes Distribution', fontsize=title_fontsize, pad=20)
            plt.xlabel('V Gene', fontsize=label_fontsize)
            plt.ylabel('Count', fontsize=label_fontsize)
            plt.xticks(range(len(top_genes)), top_genes['V_Gene'], 
                      rotation=45, ha='right', fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add combined labels (count and percentage) on bars
            max_count = max(top_genes['Count'])
            for i, (count, pct) in enumerate(zip(top_genes['Count'], top_genes['Percentage'])):
                plt.text(i, count + max_count * 0.01, f'{count}\n({pct:.1f}%)', 
                        ha='center', va='bottom', fontsize=bar_label_fontsize, fontweight='bold')
            
            # Add some padding to the top
            plt.ylim(0, max_count * 1.15)
            
        else:
            # Two subplots as before
            plt.figure(figsize=figsize)
            
            # Plot 1: Counts
            plt.subplot(2, 1, 1)
            plt.bar(range(len(top_genes)), top_genes['Count'], 
                    color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.title(f'Top {top_n} V Genes by Count', fontsize=title_fontsize)
            plt.xlabel('V Gene', fontsize=label_fontsize)
            plt.ylabel('Count', fontsize=label_fontsize)
            plt.xticks(range(len(top_genes)), top_genes['V_Gene'], 
                      rotation=45, ha='right', fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add count labels on bars
            for i, count in enumerate(top_genes['Count']):
                plt.text(i, count + max(top_genes['Count']) * 0.01, str(count), 
                        ha='center', va='bottom', fontsize=bar_label_fontsize)
            
            # Plot 2: Percentages
            plt.subplot(2, 1, 2)
            plt.bar(range(len(top_genes)), top_genes['Percentage'], 
                    color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.title(f'Top {top_n} V Genes by Percentage', fontsize=title_fontsize)
            plt.xlabel('V Gene', fontsize=label_fontsize)
            plt.ylabel('Percentage (%)', fontsize=label_fontsize)
            plt.xticks(range(len(top_genes)), top_genes['V_Gene'], 
                      rotation=45, ha='right', fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add percentage labels on bars
            for i, pct in enumerate(top_genes['Percentage']):
                plt.text(i, pct + max(top_genes['Percentage']) * 0.01, f'{pct:.1f}%', 
                        ha='center', va='bottom', fontsize=bar_label_fontsize)
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def create_pie_chart(self, top_n: int = 8, figsize: tuple = (10, 8), 
                        title_fontsize: int = 14, label_fontsize: int = 11,
                        colors: list = None, startangle: int = 90,
                        save_path: str = None, dpi: int = 300):
        """
        Create pie chart of V gene family distribution with customizable styling.
        
        Args:
            top_n (int): Number of top families to show, rest grouped as 'Others'
            figsize (tuple): Figure size (width, height)
            title_fontsize (int): Font size for title
            label_fontsize (int): Font size for labels
            colors (list): List of colors for pie slices (optional)
            startangle (int): Starting angle for pie chart
            save_path (str): Path to save the plot (e.g., 'my_pie.png', 'plots/pie_chart.pdf')
            dpi (int): Resolution for saved plot (dots per inch)
        """
        family_df = self.analyze_v_gene_families()
        
        if len(family_df) > top_n:
            top_families = family_df.head(top_n)
            others_percentage = family_df.tail(len(family_df) - top_n)['Percentage'].sum()
            
            # Add "Others" category
            others_row = pd.DataFrame({
                'V_Gene_Family': ['Others'],
                'Count': [family_df.tail(len(family_df) - top_n)['Count'].sum()],
                'Percentage': [others_percentage],
                'Total_Sequences': [family_df.iloc[0]['Total_Sequences']]
            })
            plot_data = pd.concat([top_families, others_row], ignore_index=True)
        else:
            plot_data = family_df
        
        plt.figure(figsize=figsize)
        
        # Default colors if none provided
        if colors is None:
            colors = plt.cm.Set3(range(len(plot_data)))
        
        # Create labels with both family name and count/percentage
        labels_with_counts = [f'{family}\n({count}, {pct:.1f}%)' 
                             for family, count, pct in 
                             zip(plot_data['V_Gene_Family'], plot_data['Count'], plot_data['Percentage'])]
        
        plt.pie(plot_data['Percentage'], labels=labels_with_counts, autopct='',
                startangle=startangle, colors=colors, textprops={'fontsize': label_fontsize})
        plt.title('V Gene Family Distribution', fontsize=title_fontsize, pad=20)
        plt.axis('equal')
        
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, output_prefix: str = 'vgene_simple_analysis'):
        """
        Export analysis results to CSV files.
        
        Args:
            output_prefix (str): Prefix for output files
        """
        # Export family analysis
        family_df = self.analyze_v_gene_families()
        family_df.to_csv(f'{output_prefix}_families.csv', index=False)
        
        # Export gene analysis
        gene_df = self.analyze_v_genes()
        gene_df.to_csv(f'{output_prefix}_genes.csv', index=False)
        
        # Export summary stats
        summary = self.get_summary_stats()
        with open(f'{output_prefix}_summary.txt', 'w') as f:
            f.write("V Gene Analysis Summary\\n")
            f.write("=" * 25 + "\\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\\n")
        
        print(f"Results exported with prefix '{output_prefix}'")
        print("Files created:")
        print(f"  - {output_prefix}_families.csv")
        print(f"  - {output_prefix}_genes.csv")
        print(f"  - {output_prefix}_summary.txt")

# Convenience functions
def quick_family_analysis(json_file_path: str, figsize: tuple = (12, 8), 
                         bar_color: str = 'steelblue', fontsize_large: int = 14, 
                         fontsize_medium: int = 12, fontsize_small: int = 10,
                         bar_label_fontsize: int = 9):
    """
    Perform a quick V gene family analysis with customizable plot styling.
    
    Args:
        json_file_path (str): Path to JSON file
        figsize (tuple): Figure size for plots
        bar_color (str): Color for bar plots
        fontsize_large (int): Font size for titles
        fontsize_medium (int): Font size for axis labels  
        fontsize_small (int): Font size for tick labels
        bar_label_fontsize (int): Font size for count/percentage labels on bars
    """
    analyzer = SimpleVGeneAnalyzer(json_file_path)
    
    print("\\n=== V Gene Family Analysis ===")
    
    # Show summary
    summary = analyzer.get_summary_stats()
    print(f"\\nDataset Summary:")
    print(f"Total sequences: {summary['total_sequences']:,}")
    print(f"Unique V gene families: {summary['unique_families']}")
    print(f"Unique V genes: {summary['unique_genes']}")
    print(f"Most common family: {summary['most_common_family']} ({summary['most_common_family_percentage']:.1f}%)")
    print(f"Most common gene: {summary['most_common_gene']} ({summary['most_common_gene_percentage']:.1f}%)")
    
    # Show family distribution
    print("\\n--- V Gene Family Distribution ---")
    family_df = analyzer.analyze_v_gene_families()
    print(family_df.head(10))
    
    # Show top genes
    print("\\n--- Top V Genes ---")
    gene_df = analyzer.analyze_v_genes()
    print(gene_df.head(10))
    
    # Generate plots with custom styling
    analyzer.plot_family_distribution(
        figsize=figsize, 
        bar_color=bar_color,
        title_fontsize=fontsize_large,
        label_fontsize=fontsize_medium,
        tick_fontsize=fontsize_small,
        bar_label_fontsize=bar_label_fontsize,
        show_both_metrics=True
    )
    
    analyzer.create_pie_chart(
        figsize=figsize,
        title_fontsize=fontsize_large,
        label_fontsize=fontsize_small
    )
    
    return analyzer

def compare_datasets(file_paths: List[str], labels: List[str] = None):
    """
    Compare V gene family distributions across multiple datasets.
    
    Args:
        file_paths (List[str]): List of JSON file paths
        labels (List[str]): Optional labels for each dataset
    """
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(file_paths))]
    
    comparison_data = []
    
    for i, file_path in enumerate(file_paths):
        analyzer = SimpleVGeneAnalyzer(file_path)
        family_df = analyzer.analyze_v_gene_families()
        
        for _, row in family_df.iterrows():
            comparison_data.append({
                'Dataset': labels[i],
                'V_Gene_Family': row['V_Gene_Family'],
                'Count': row['Count'],
                'Percentage': row['Percentage'],
                'Total_Sequences': row['Total_Sequences']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create comparison plot
    families = comparison_df['V_Gene_Family'].unique()
    x = range(len(families))
    width = 0.8 / len(labels)
    
    plt.figure(figsize=(15, 8))
    
    for i, label in enumerate(labels):
        dataset_data = comparison_df[comparison_df['Dataset'] == label]
        percentages = []
        for family in families:
            family_data = dataset_data[dataset_data['V_Gene_Family'] == family]
            pct = family_data['Percentage'].iloc[0] if len(family_data) > 0 else 0
            percentages.append(pct)
        
        plt.bar([xi + i * width for xi in x], percentages, width, label=label, alpha=0.8)
    
    plt.xlabel('V Gene Family')
    plt.ylabel('Percentage (%)')
    plt.title('V Gene Family Distribution Comparison')
    plt.xticks([xi + width * (len(labels) - 1) / 2 for xi in x], families, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return comparison_df