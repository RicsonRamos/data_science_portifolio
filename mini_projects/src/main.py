from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def pollution_graphs(df, metric, group_col=['State', 'Year'], out_dir=r'./reports'):
    """
    Generates pollution-related visualizations including a boxplot and a heatmap.
    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing pollution data.
    metric : str
        The name of the column in the DataFrame that represents the pollution metric to be analyzed.
    group_col : list of str, optional
        The columns to group by for the analysis. Default is ['State', 'Year'].
    out_dir : str, optional
        The directory where the generated plots will be saved. Default is './reports'.
    Raises:
    ------
    KeyError
        If the specified metric is not found in the DataFrame columns.
    Notes:
    -----
    - The function creates a boxplot to visualize the distribution of the specified metric 
      across the specified grouping columns.
    - If both 'Year' and 'State' are present in the grouped data, a heatmap is also generated 
      to show the metric values across states and years.
    - The output plots are saved as PNG files in the specified output directory.
    """
    # Validações
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if metric not in df.columns:
        raise KeyError(f"metric '{metric}' not found in DataFrame columns")
    
    if isinstance(group_col, str):
        group_col = [group_col]

    grouped = df.groupby(group_col)[metric].mean().reset_index()

    # Converte eixo categórico para string quando necessário
    x_col = group_col[0]
    grouped[x_col] = grouped[x_col].astype(str)

    # Boxplot
    box_path = out_dir / f"{metric}_boxplot.png"
    fig, ax = plt.subplots(figsize=(15,6))
    sns.boxplot(x=x_col, y=metric, data=grouped, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_title(f'Distribuição de {metric} por {x_col}', fontsize=14)
    plt.tight_layout()
    fig.savefig(box_path, dpi=150)
    plt.close(fig)

    # Heatmap (só se Year e State estiverem presentes)
    if 'Year' in grouped.columns and 'State' in grouped.columns:
        heatmap_data = grouped.pivot(index='State', columns='Year', values=metric)
        heatmap_data = heatmap_data.fillna(0)  # ou heatmap_data.fillna(heatmap_data.mean())

        heat_path = out_dir / f"{metric}_heatmap.png"
        fig, ax = plt.subplots(figsize=(12,6))
        sns.heatmap(heatmap_data, cmap='Reds', linewidths=0.5, ax=ax, cbar_kws={'label': metric})
        ax.set_title(f'{metric} por Estado e Ano', fontsize=14)
        plt.tight_layout()
        fig.savefig(heat_path, dpi=150)
        plt.close(fig)
