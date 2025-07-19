import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Tuple, Optional

class VGeneAnalyzer:
    """
    A class to analyze V gene families and genes from overrepresented sequence data.
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
    
    def extract_count_from_id(self, sequence_id: str) -> int:
        """
        Extract the count from the sequence ID.
        
        Args:
            sequence_id (str): Sequence ID like "overrep_seq_0911_count=8_percentage=0.01%" 
                              or "overrep_seq_0320_count123_pct0.03"
            
        Returns:
            int: The count value
        """
        # Try format with equals sign first: count=123
        match = re.search(r'count=(\d+)', sequence_id)
        if match:
            return int(match.group(1))
        
        # Try format without equals sign: count123_
        match = re.search(r'count(\d+)_', sequence_id)
        if match:
            return int(match.group(1))
        
        # Try more general pattern: count followed by digits
        match = re.search(r'count(\d+)', sequence_id)
        if match:
            return int(match.group(1))
        
        return 0
    
    def extract_v_gene_info(self, hits: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract V gene family and full gene name from hits.
        
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
        
        # Extract gene family (e.g., "IGKV1" from "IGKV1-39*01")
        family_match = re.match(r'(IG[HKL]V\d+)', gene_name)
        gene_family = family_match.group(1) if family_match else None
        
        # Extract full gene name (e.g., "IGKV1-39*01" from "IGKV1-39*01 unnamed protein product")
        gene_match = re.match(r'([^\\s]+)', gene_name)
        full_gene = gene_match.group(1) if gene_match else None
        
        return gene_family, full_gene
    
    def filter_sequences(self, min_count: int = 0) -> List[Dict]:
        """
        Filter sequences based on minimum count threshold.
        
        Args:
            min_count (int): Minimum count threshold
            
        Returns:
            List[Dict]: Filtered sequence data
        """
        filtered_data = []
        for entry in self.data:
            count = self.extract_count_from_id(entry.get('Sequence ID', ''))
            if count >= min_count:
                filtered_data.append(entry)
        
        print(f"Filtered to {len(filtered_data)} sequences with count >= {min_count}")
        return filtered_data
    
    def analyze_v_genes(self, min_count: int = 0) -> pd.DataFrame:
        """
        Analyze V gene families and genes with optional filtering.
        
        Args:
            min_count (int): Minimum count threshold for filtering sequences
            
        Returns:
            pd.DataFrame: Analysis results
        """
        filtered_data = self.filter_sequences(min_count)
        
        results = []
        for entry in filtered_data:
            sequence_id = entry.get('Sequence ID', '')
            count = self.extract_count_from_id(sequence_id)
            hits = entry.get('Hits', [])
            
            gene_family, full_gene = self.extract_v_gene_info(hits)
            
            results.append({
                'Sequence_ID': sequence_id,
                'Count': count,
                'V_Gene_Family': gene_family,
                'V_Gene': full_gene,
                'Bit_Score': hits[0].get('bit_score') if hits else None,
                'E_Value': hits[0].get('e_value') if hits else None,
                'Sequence_Length': entry.get('Sequence Length'),
                'Raw_Sequence': entry.get('Raw Sequence', '')[:50] + '...'  # Truncated for display
            })
        
        return pd.DataFrame(results)
    
    def get_family_summary(self, min_count: int = 0) -> pd.DataFrame:
        """
        Get summary statistics for V gene families.
        
        Args:
            min_count (int): Minimum count threshold
            
        Returns:
            pd.DataFrame: Family summary statistics
        """
        df = self.analyze_v_genes(min_count)
        
        family_stats = df.groupby('V_Gene_Family').agg({
            'Count': ['sum', 'mean', 'std', 'count'],
            'Sequence_Length': 'mean',
            'Bit_Score': 'mean',
            'E_Value': 'mean'
        }).round(2)
        
        # Flatten column names
        family_stats.columns = ['_'.join(col).strip() for col in family_stats.columns]
        family_stats = family_stats.reset_index()
        
        # Sort by total count
        family_stats = family_stats.sort_values('Count_sum', ascending=False)
        
        return family_stats
    
    def get_gene_summary(self, min_count: int = 0) -> pd.DataFrame:
        """
        Get summary statistics for individual V genes.
        
        Args:
            min_count (int): Minimum count threshold
            
        Returns:
            pd.DataFrame: Gene summary statistics
        """
        df = self.analyze_v_genes(min_count)
        
        gene_stats = df.groupby('V_Gene').agg({
            'Count': ['sum', 'mean', 'std', 'count'],
            'Sequence_Length': 'mean',
            'Bit_Score': 'mean',
            'E_Value': 'mean'
        }).round(2)
        
        # Flatten column names
        gene_stats.columns = ['_'.join(col).strip() for col in gene_stats.columns]
        gene_stats = gene_stats.reset_index()
        
        # Sort by total count
        gene_stats = gene_stats.sort_values('Count_sum', ascending=False)
        
        return gene_stats
    
    def plot_family_distribution(self, min_count: int = 0, top_n: int = 10, 
                                figsize: tuple = (12, 6), title_fontsize: int = 14, 
                                label_fontsize: int = 12, tick_fontsize: int = 10, 
                                bar_label_fontsize: int = 9, bar_color: str = 'steelblue',
                                show_both_metrics: bool = True, save_path: str = None, 
                                dpi: int = 300):
        """
        Plot V gene family distribution with customizable styling.
        
        Args:
            min_count (int): Minimum count threshold
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
        family_summary = self.get_family_summary(min_count)
        top_families = family_summary.head(top_n)
        
        if show_both_metrics:
            # Single plot with counts as bars and labels showing both count and percentage
            plt.figure(figsize=figsize)
            
            # Calculate percentages
            total_sequences = family_summary['Count_count'].sum()
            percentages = (top_families['Count_sum'] / family_summary['Count_sum'].sum() * 100).round(2)
            
            bars = plt.bar(top_families['V_Gene_Family'], top_families['Count_sum'], 
                          color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            plt.title(f'Top {top_n} V Gene Families Distribution (min_count >= {min_count})', 
                     fontsize=title_fontsize, pad=20)
            plt.xlabel('V Gene Family', fontsize=label_fontsize)
            plt.ylabel('Count', fontsize=label_fontsize)
            plt.xticks(rotation=45, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add combined labels (count and percentage) on bars
            max_count = max(top_families['Count_sum'])
            for i, (family, count, pct) in enumerate(zip(top_families['V_Gene_Family'], 
                                                        top_families['Count_sum'], 
                                                        percentages)):
                plt.text(i, count + max_count * 0.01, f'{count}\n({pct:.2f}%)', 
                        ha='center', va='bottom', fontsize=bar_label_fontsize, fontweight='bold')
            
            # Add some padding to the top
            plt.ylim(0, max_count * 1.15)
            
        else:
            # Original two-panel layout
            plt.figure(figsize=figsize)
            
            # Plot 1: Total counts
            plt.subplot(1, 2, 1)
            bars1 = plt.bar(top_families['V_Gene_Family'], top_families['Count_sum'], 
                           color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.title(f'Top {top_n} V Gene Families by Total Count\\n(min_count >= {min_count})', 
                     fontsize=title_fontsize)
            plt.xlabel('V Gene Family', fontsize=label_fontsize)
            plt.ylabel('Total Count', fontsize=label_fontsize)
            plt.xticks(rotation=45, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add count labels on bars
            for i, (family, count) in enumerate(zip(top_families['V_Gene_Family'], top_families['Count_sum'])):
                plt.text(i, count + max(top_families['Count_sum']) * 0.01, str(count), 
                        ha='center', va='bottom', fontsize=bar_label_fontsize)
            
            # Plot 2: Number of sequences
            plt.subplot(1, 2, 2)
            bars2 = plt.bar(top_families['V_Gene_Family'], top_families['Count_count'], 
                           color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.title(f'Top {top_n} V Gene Families by Number of Sequences\\n(min_count >= {min_count})', 
                     fontsize=title_fontsize)
            plt.xlabel('V Gene Family', fontsize=label_fontsize)
            plt.ylabel('Number of Sequences', fontsize=label_fontsize)
            plt.xticks(rotation=45, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add sequence count labels on bars
            for i, (family, count) in enumerate(zip(top_families['V_Gene_Family'], top_families['Count_count'])):
                plt.text(i, count + max(top_families['Count_count']) * 0.01, str(count), 
                        ha='center', va='bottom', fontsize=bar_label_fontsize)
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_gene_distribution(self, min_count: int = 0, top_n: int = 15, 
                              figsize: tuple = (15, 8), title_fontsize: int = 14, 
                              label_fontsize: int = 12, tick_fontsize: int = 10, 
                              bar_label_fontsize: int = 8, bar_color: str = 'darkgreen',
                              show_both_metrics: bool = True, save_path: str = None, 
                              dpi: int = 300):
        """
        Plot individual V gene distribution with customizable styling.
        
        Args:
            min_count (int): Minimum count threshold
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
        gene_summary = self.get_gene_summary(min_count)
        top_genes = gene_summary.head(top_n)
        
        if show_both_metrics:
            # Single plot with counts as bars and labels showing both count and percentage
            plt.figure(figsize=figsize)
            
            # Calculate percentages
            percentages = (top_genes['Count_sum'] / gene_summary['Count_sum'].sum() * 100).round(2)
            
            bars = plt.bar(range(len(top_genes)), top_genes['Count_sum'], 
                          color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            plt.title(f'Top {top_n} V Genes Distribution (min_count >= {min_count})', 
                     fontsize=title_fontsize, pad=20)
            plt.xlabel('V Gene', fontsize=label_fontsize)
            plt.ylabel('Total Count', fontsize=label_fontsize)
            plt.xticks(range(len(top_genes)), top_genes['V_Gene'], 
                      rotation=45, ha='right', fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add combined labels (count and percentage) on bars
            max_count = max(top_genes['Count_sum'])
            for i, (count, pct) in enumerate(zip(top_genes['Count_sum'], percentages)):
                plt.text(i, count + max_count * 0.01, f'{count}\\n({pct:.2f}%)', 
                        ha='center', va='bottom', fontsize=bar_label_fontsize, fontweight='bold')
            
            # Add some padding to the top
            plt.ylim(0, max_count * 1.15)
            
        else:
            # Original two-panel layout
            plt.figure(figsize=figsize)
            
            # Plot 1: Total counts
            plt.subplot(2, 1, 1)
            bars1 = plt.bar(range(len(top_genes)), top_genes['Count_sum'], 
                           color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.title(f'Top {top_n} V Genes by Total Count (min_count >= {min_count})', 
                     fontsize=title_fontsize)
            plt.xlabel('V Gene', fontsize=label_fontsize)
            plt.ylabel('Total Count', fontsize=label_fontsize)
            plt.xticks(range(len(top_genes)), top_genes['V_Gene'], 
                      rotation=45, ha='right', fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add count labels on bars
            for i, count in enumerate(top_genes['Count_sum']):
                plt.text(i, count + max(top_genes['Count_sum']) * 0.01, str(count), 
                        ha='center', va='bottom', fontsize=bar_label_fontsize)
            
            # Plot 2: Number of sequences
            plt.subplot(2, 1, 2)
            bars2 = plt.bar(range(len(top_genes)), top_genes['Count_count'], 
                           color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.title(f'Top {top_n} V Genes by Number of Sequences (min_count >= {min_count})', 
                     fontsize=title_fontsize)
            plt.xlabel('V Gene', fontsize=label_fontsize)
            plt.ylabel('Number of Sequences', fontsize=label_fontsize)
            plt.xticks(range(len(top_genes)), top_genes['V_Gene'], 
                      rotation=45, ha='right', fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            
            # Add sequence count labels on bars
            for i, count in enumerate(top_genes['Count_count']):
                plt.text(i, count + max(top_genes['Count_count']) * 0.01, str(count), 
                        ha='center', va='bottom', fontsize=bar_label_fontsize)
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def create_pie_chart(self, min_count: int = 0, top_n: int = 8, 
                        figsize: tuple = (10, 8), title_fontsize: int = 14, 
                        label_fontsize: int = 11, colors: list = None, 
                        startangle: int = 90, save_path: str = None, dpi: int = 300):
        """
        Create pie chart of V gene family distribution with customizable styling.
        
        Args:
            min_count (int): Minimum count threshold
            top_n (int): Number of top families to show, rest grouped as 'Others'
            figsize (tuple): Figure size (width, height)
            title_fontsize (int): Font size for title
            label_fontsize (int): Font size for labels
            colors (list): List of colors for pie slices (optional)
            startangle (int): Starting angle for pie chart
            save_path (str): Path to save the plot (e.g., 'my_pie.png', 'plots/pie_chart.pdf')
            dpi (int): Resolution for saved plot (dots per inch)
        """
        family_summary = self.get_family_summary(min_count)
        
        if len(family_summary) > top_n:
            top_families = family_summary.head(top_n)
            others_count = family_summary.tail(len(family_summary) - top_n)['Count_sum'].sum()
            others_percentage = (others_count / family_summary['Count_sum'].sum() * 100).round(2)
            
            # Add "Others" category
            others_row = pd.DataFrame({
                'V_Gene_Family': ['Others'],
                'Count_sum': [others_count],
                'Count_count': [family_summary.tail(len(family_summary) - top_n)['Count_count'].sum()]
            })
            plot_data = pd.concat([top_families, others_row], ignore_index=True)
        else:
            plot_data = family_summary
        
        # Calculate percentages for the plot data
        percentages = (plot_data['Count_sum'] / plot_data['Count_sum'].sum() * 100).round(2)
        
        plt.figure(figsize=figsize)
        
        # Default colors if none provided
        if colors is None:
            colors = plt.cm.Set3(range(len(plot_data)))
        
        # Create labels with both family name and count/percentage
        labels_with_counts = [f'{family}\\n({count}, {pct:.2f}%)' 
                             for family, count, pct in 
                             zip(plot_data['V_Gene_Family'], plot_data['Count_sum'], percentages)]
        
        plt.pie(percentages, labels=labels_with_counts, autopct='',
                startangle=startangle, colors=colors, textprops={'fontsize': label_fontsize})
        plt.title(f'V Gene Family Distribution (min_count >= {min_count})', 
                 fontsize=title_fontsize, pad=20)
        plt.axis('equal')
        
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, min_count: int = 0, output_prefix: str = 'vgene_analysis'):
        """
        Export analysis results to CSV files.
        
        Args:
            min_count (int): Minimum count threshold
            output_prefix (str): Prefix for output files
        """
        # Export detailed results
        detailed_df = self.analyze_v_genes(min_count)
        detailed_df.to_csv(f'{output_prefix}_detailed_min{min_count}.csv', index=False)
        
        # Export family summary
        family_summary = self.get_family_summary(min_count)
        family_summary.to_csv(f'{output_prefix}_family_summary_min{min_count}.csv', index=False)
        
        # Export gene summary
        gene_summary = self.get_gene_summary(min_count)
        gene_summary.to_csv(f'{output_prefix}_gene_summary_min{min_count}.csv', index=False)
        
        print(f"Results exported with prefix '{output_prefix}' and min_count={min_count}")

# Example usage functions
def quick_analysis(json_file_path: str, min_count: int = 100, figsize: tuple = (12, 8), 
                  bar_color: str = 'steelblue', fontsize_large: int = 14, 
                  fontsize_medium: int = 12, fontsize_small: int = 10,
                  bar_label_fontsize: int = 9):
    """
    Perform a quick analysis with customizable plot styling.
    
    Args:
        json_file_path (str): Path to JSON file
        min_count (int): Minimum count threshold
        figsize (tuple): Figure size for plots
        bar_color (str): Color for bar plots
        fontsize_large (int): Font size for titles
        fontsize_medium (int): Font size for axis labels  
        fontsize_small (int): Font size for tick labels
        bar_label_fontsize (int): Font size for count/percentage labels on bars
    """
    analyzer = VGeneAnalyzer(json_file_path)
    
    print(f"\\n=== V Gene Analysis (min_count >= {min_count}) ===")
    
    # Show family summary
    print("\\n--- V Gene Family Summary ---")
    family_summary = analyzer.get_family_summary(min_count)
    print(family_summary.head(10))
    
    # Show gene summary
    print("\\n--- Top V Genes Summary ---")
    gene_summary = analyzer.get_gene_summary(min_count)
    print(gene_summary.head(10))
    
    # Generate plots with custom styling
    analyzer.plot_family_distribution(
        min_count=min_count,
        figsize=figsize, 
        bar_color=bar_color,
        title_fontsize=fontsize_large,
        label_fontsize=fontsize_medium,
        tick_fontsize=fontsize_small,
        bar_label_fontsize=bar_label_fontsize,
        show_both_metrics=True
    )
    
    analyzer.plot_gene_distribution(
        min_count=min_count,
        figsize=figsize,
        bar_color='darkgreen',
        title_fontsize=fontsize_large,
        label_fontsize=fontsize_medium,
        tick_fontsize=fontsize_small,
        bar_label_fontsize=bar_label_fontsize,
        show_both_metrics=True
    )
    
    analyzer.create_pie_chart(
        min_count=min_count,
        figsize=figsize,
        title_fontsize=fontsize_large,
        label_fontsize=fontsize_small
    )
    
    return analyzer

def compare_thresholds(json_file_path: str, thresholds: List[int] = [0, 10, 50, 100]):
    """
    Compare analysis results across different count thresholds.
    
    Args:
        json_file_path (str): Path to JSON file
        thresholds (List[int]): List of count thresholds to compare
    """
    analyzer = VGeneAnalyzer(json_file_path)
    
    comparison_results = []
    for threshold in thresholds:
        family_summary = analyzer.get_family_summary(threshold)
        total_sequences = len(analyzer.filter_sequences(threshold))
        total_families = len(family_summary)
        
        comparison_results.append({
            'Threshold': threshold,
            'Total_Sequences': total_sequences,
            'Total_Families': total_families,
            'Top_Family': family_summary.iloc[0]['V_Gene_Family'] if len(family_summary) > 0 else 'N/A',
            'Top_Family_Count': family_summary.iloc[0]['Count_sum'] if len(family_summary) > 0 else 0
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    print("\\n=== Threshold Comparison ===")
    print(comparison_df)
    
    return comparison_df