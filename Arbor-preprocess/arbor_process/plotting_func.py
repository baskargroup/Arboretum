import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns

def generate_plots(csv_filename, destination_folder, min_threshold=100, max_threshold=1000):
    """
    Generate plots for category totals, species counts, and top 40 species, and save them to a specified folder.
    
    Args:
        csv_filename (str): Path to the CSV file containing species data.
        destination_folder (str): Path to the folder where the plots will be saved.
        min_threshold (int): Minimum threshold value for creating histograms of species count below the specified value.
        max_threshold (int): Maximum threshold value for creating histograms of species count above the specified value.
    """
    # Load the CSV file
    df = pd.read_csv(csv_filename)

    # Generate group by category total of the count column
    category_totals = df.groupby('category')['count'].sum().reset_index()

    # Generate the number of species in each category
    species_counts = df.groupby('category')['species'].nunique().reset_index()
    species_counts.columns = ['category', 'species_count']

    # Merge the two dataframes on the category column
    category_df = pd.merge(category_totals, species_counts, on='category')

    # Top 40 species by count
    top_40_species = df.groupby('species')['count'].sum().nlargest(40).reset_index()

    # Set Seaborn style for publication quality
    sns.set(style='white', context='paper')
    
    # Formatter with one decimal place
    formatter = ticker.EngFormatter(places=1)

    # Plotting the first figure
    fig1, axes1 = plt.subplot_mosaic([['1,1', '1,2'], ['2,1', '2,1']], figsize=(8, 6))

    # Plot 1,1: Number of species in each category
    axes1['1,2'].bar(category_df['category'], category_df['species_count'], color='skyblue')
    axes1['1,2'].set_title('Species Distribution in the Categories', fontsize=10)
    axes1['1,2'].yaxis.set_major_formatter(formatter)
    for i, value in enumerate(category_df['species_count']):
        axes1['1,2'].text(i, value-0.5*value, f'{formatter(np.ceil(value))}', ha='center', va='bottom', fontsize=8, rotation=90)
    axes1['1,2'].tick_params(axis='x', rotation=45, labelsize=8)
    sns.despine(ax=axes1['1,2'])

    # Plot 1,2: Total count in each category
    axes1['1,1'].bar(category_df['category'], category_df['count'], color='salmon')
    axes1['1,1'].set_title('Size of the Categories', fontsize=10)
    axes1['1,1'].yaxis.set_major_formatter(formatter)
    for i, value in enumerate(category_df['count']):
        axes1['1,1'].text(i, value-0.5*value, f'{formatter(np.ceil(value))}', ha='center', va='bottom', fontsize=8, rotation=90)
    axes1['1,1'].tick_params(axis='x', rotation=45, labelsize=8)
    sns.despine(ax=axes1['1,1'])

    # Plot 2,1 (and 2,2): Top 40 species spanning both cells
    axes1['2,1'].bar(top_40_species['species'], top_40_species['count'], color='lightgreen')
    axes1['2,1'].set_title('Top 40 Species', fontsize=10)
    axes1['2,1'].yaxis.set_major_formatter(formatter)
    for i, value in enumerate(top_40_species['count']):
        axes1['2,1'].text(i, value - value * 0.5, f'{formatter(np.ceil(value))}', ha='center', va='bottom', fontsize=8, rotation=90)
    axes1['2,1'].tick_params(axis='x', rotation=90, labelsize=8)
    sns.despine(ax=axes1['2,1'])

    # Adding subplot labels (a), (b), and (c)
    axes1['1,1'].text(0.5, 0.6, '(a)', ha='center', va='top', transform=axes1['1,1'].transAxes, fontsize=10)
    axes1['1,2'].text(0.5, 0.6, '(b)', ha='center', va='top', transform=axes1['1,2'].transAxes, fontsize=10)
    axes1['2,1'].text(0.5, 0.6, '(c)', ha='center', va='top', transform=axes1['2,1'].transAxes, fontsize=10)

    plt.tight_layout()

    # Save the first plot to the specified destination folder
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    plot_path1 = os.path.join(destination_folder, 'category_species_plots.png')
    plt.savefig(plot_path1)
    #plt.show()

    # Plotting the second figure with histograms using 2 by 2 subplots
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram (a): Histogram of species by count, using both axes semi-log
    axes2[0, 0].hist(df['count'], bins=50, color='purple', log=True)
    axes2[0, 0].set_xscale('log')
    axes2[0, 0].set_yscale('log')
    axes2[0, 0].set_title('Histogram of Species by Count (log Axes)')
    sns.despine(ax=axes2[0, 0])

    # Histogram (b): Histogram of species by count below min_threshold
    df_filtered_below_threshold = df[df['count'] <= min_threshold]
    axes2[0, 1].hist(df_filtered_below_threshold['count'], bins=50, color='blue')
    axes2[0, 1].set_title(f'Histogram of Species by Count (Below {min_threshold})')
    sns.despine(ax=axes2[0, 1])

    # Histogram (c): Histogram of species by count above max_threshold
    df_filtered_above_threshold = df[df['count'] > max_threshold]
    axes2[1, 0].hist(df_filtered_above_threshold['count'], bins=50, color='green')
    axes2[1, 0].set_title(f'Histogram of Species by Count (Above {max_threshold})')
    sns.despine(ax=axes2[1, 0])

    # Creating the table for subplot (d)
    thresholds = [25, 50, 75, 100, 200, 500, 1000]
    species_counts_at_thresholds = [len(df[df['count'] > t]) for t in thresholds]
    total_counts_at_thresholds = [df[df['count'] > t]['count'].sum() for t in thresholds]

    table_data = {
        'Threshold': thresholds,
        'Number of Species': species_counts_at_thresholds,
        'Total Counts': total_counts_at_thresholds
    }

    table_df = pd.DataFrame(table_data)

    # Plotting the table
    axes2[1, 1].axis('off')
    table = axes2[1, 1].table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    axes2[1, 1].set_title('Threshold Counts Table')

    plt.tight_layout()

    # Save the second plot to the specified destination folder
    plot_path2 = os.path.join(destination_folder, 'species_histograms.png')
    plt.savefig(plot_path2)
    #plt.show()

