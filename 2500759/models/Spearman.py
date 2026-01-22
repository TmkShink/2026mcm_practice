import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

def main():
    print("Starting Spearman Correlation Analysis for 2024...")
    
    # 1. Load Data
    data_dir = 'Data'
    if not os.path.exists(data_dir): data_dir = '../Data'
    
    athletes_path = os.path.join(data_dir, 'summerOly_athletes.csv')
    
    print(f"Loading athletes data from {athletes_path}...")
    df = pd.read_csv(athletes_path)
    
    # Filter 2024
    df_2024 = df[df['Year'] == 2024].copy()
    print(f"2024 Entries: {len(df_2024)}")
    
    if df_2024.empty:
        print("Error: No 2024 data found.")
        return

    # 2. Data Processing
    # We need to handle team sports (one medal per team, not per athlete)
    # Deduplicate based on Event, NOC, and Medal
    # Keep only rows with Medals (Gold, Silver, Bronze)
    df_medals_only = df_2024[df_2024['Medal'].isin(['Gold', 'Silver', 'Bronze'])].copy()
    
    # Deduplicate to get distinct Medal events per country
    df_dedup = df_medals_only.drop_duplicates(subset=['Event', 'NOC', 'Medal'])
    print(f"Unique Medal Events: {len(df_dedup)}")
    
    # 3. Aggregation
    # A. Country Totals (Gold Count, Total Count)
    # Group by NOC
    noc_stats = df_dedup.groupby('NOC')['Medal'].value_counts().unstack(fill_value=0)
    
    if 'Gold' not in noc_stats.columns: noc_stats['Gold'] = 0
    if 'Silver' not in noc_stats.columns: noc_stats['Silver'] = 0
    if 'Bronze' not in noc_stats.columns: noc_stats['Bronze'] = 0
    
    noc_stats['Total'] = noc_stats['Gold'] + noc_stats['Silver'] + noc_stats['Bronze']
    
    # Filter: Only consider countries with at least 1 medal?
    # Usually Correlation analysis includes all variability.
    # But if a country has 0 medals, it's not in this list (since we filtered medals_only).
    # We should probably stick to countries that won something to avoid massive zero-inflation?
    # The prompt implies: "strong nations ... excelling in ...".
    # Let's use the set of medal-winning nations.
    
    # B. Sport Totals per Country
    # How many medals (Total count) did Country C win in Sport S?
    # Note: The prompt mentions "correlation for gold medals... but higher for total".
    # This implies we might need Sport_Gold vs Total_Gold?
    # Or Sport_Total vs Total_Gold?
    # Usually "Correlation between a sport and Gold/Total medals" implies:
    # Does success in Sport X correlate with Overall Success (Gold/Total)?
    # Variable X: Country's Medal Count in Sport S.
    # Variable Y1: Country's Total Gold.
    # Variable Y2: Country's Total Medals.
    
    # Count medals per Sport per NOC
    sport_stats = df_dedup.groupby(['NOC', 'Sport']).size().unstack(fill_value=0)
    
    # Align indices (NOCs)
    # Ensure noc_stats and sport_stats have same NOCs
    common_nocs = noc_stats.index.intersection(sport_stats.index)
    noc_stats = noc_stats.loc[common_nocs]
    sport_stats = sport_stats.loc[common_nocs]
    
    # 4. Correlation Calculation
    results = []
    
    target_gold = noc_stats['Gold']
    target_total = noc_stats['Total']
    
    all_sports = sport_stats.columns
    print(f"Analyzing {len(all_sports)} sports...")
    
    for sport in all_sports:
        # Vector of medals in this sport for all countries
        sport_vec = sport_stats[sport]
        
        # Spearman for Gold
        corr_g, p_g = spearmanr(sport_vec, target_gold)
        
        # Spearman for Total
        corr_t, p_t = spearmanr(sport_vec, target_total)
        
        results.append({
            'Sport': sport,
            'Gold_Corr': corr_g,
            'Gold_P': p_g,
            'Total_Corr': corr_t,
            'Total_P': p_t
        })
        
    df_res = pd.DataFrame(results)
    
    # Handle NaNs (if constant input)
    df_res = df_res.fillna(0)
    
    # 5. Filtering
    # "Excluded sports with non-significant correlation coefficients"
    # Filter where p > 0.05 for *both* or *either*?
    # Usually we want significant in at least one? Or both?
    # Let's filter if BOTH are non-significant (p > 0.05).
    # Or strictly filter.
    # "Wrestling, Beach Volleyball... were excluded".
    
    # Let's apply a threshold. p < 0.05 is signif.
    # If p_g > 0.05 AND p_t > 0.05, exclude.
    mask_sig = (df_res['Gold_P'] < 0.05) | (df_res['Total_P'] < 0.05)
    df_sig = df_res[mask_sig].copy()
    
    excluded = df_res[~mask_sig]['Sport'].tolist()
    print(f"Excluded {len(excluded)} sports: {excluded}")
    
    # Save
    if not os.path.exists('processed_data'): os.makedirs('processed_data')
    output_path = 'processed_data/spearman_correlations.csv'
    df_sig.to_csv(output_path, index=False)
    print(f"Saved significant correlations to {output_path}")
    
    # 6. Plotting
    plot_heatmap_dot(df_sig)

def plot_heatmap_dot(df):
    """
    Replicate the dot-matrix heatmap.
    Rows: Gold, Total
    Cols: Sports
    Color/Size: Correlation
    """
    # Reshape for plotting
    # We want a "long" format: Sport | Type (Gold/Total) | Correlation
    
    # Sort by Total Correlation desc
    df = df.sort_values('Total_Corr', ascending=False)
    
    # Prepare data
    sports = df['Sport'].tolist()
    
    # We simply plot two scatter plots on same axis
    
    plt.figure(figsize=(15, 4))
    sns.set_style("whitegrid")
    
    # X coordinates
    x_coords = np.arange(len(sports))
    
    # Y coordinates: Gold=1, Total=0
    # Data for Gold
    y_gold = np.ones(len(sports))
    c_gold = df['Gold_Corr'].values
    s_gold = np.abs(c_gold) * 500  # Size scaling
    
    # Data for Total
    y_total = np.zeros(len(sports))
    c_total = df['Total_Corr'].values
    s_total = np.abs(c_total) * 500 
    
    # Plot Scatter
    # Use a colormap like 'Blues'
    plt.scatter(x_coords, y_gold, s=s_gold, c=c_gold, cmap='Blues', vmin=0, vmax=1, alpha=0.9, edgecolors='white')
    plt.scatter(x_coords, y_total, s=s_total, c=c_total, cmap='Blues', vmin=0, vmax=1, alpha=0.9, edgecolors='white')
    
    # Formatting
    plt.yticks([0, 1], ['Total', 'Gold'], fontsize=12, fontweight='bold')
    plt.xticks(x_coords, sports, rotation=90, fontsize=10)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.xlabel("")
    plt.title("Spearman Correlation: Sports vs Medal Counts (Paris 2024)", fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(orientation='vertical', pad=0.01)
    cbar.set_label('Correlation Coefficient')
    
    plt.ylim(-0.5, 1.5)
    plt.margins(x=0.02)
    plt.tight_layout()
    
    save_path = 'processed_data/spearman_heatmap.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
