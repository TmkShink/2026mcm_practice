import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns

def main():
    # 1. Load Data
    data_path = '../Data/summerOly_athletes.csv'
    if not os.path.exists(data_path):
        data_path = 'Data/summerOly_athletes.csv' # Try from root
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Data Filtering: Lock to 2020 Summer Olympics
    df_2020 = df[(df['Year'] == 2020) & (df['Medal'].isin(['Gold', 'Silver', 'Bronze']))].copy()
    
    if df_2020.empty:
        print("No data found for Year 2020 with medals.")
        return

    print(f"Data filtered for 2020 Olympics. {len(df_2020)} entries.")

    # 3. Deduplication: Handle Team Sports
    df_dedup = df_2020.drop_duplicates(subset=['NOC', 'Event', 'Medal']).copy()

    # 4. Weighted Calculation
    medal_weights = {'Gold': 0.5, 'Silver': 0.3, 'Bronze': 0.2}
    df_dedup['Weight'] = df_dedup['Medal'].map(medal_weights)

    # Calculate weighted score for each Country (NOC) in each Sport
    noc_sport_score = df_dedup.groupby(['NOC', 'Sport'])['Weight'].sum().reset_index()

    # 5. Matrix Construction
    matrix = noc_sport_score.pivot(index='NOC', columns='Sport', values='Weight').fillna(0)
    print(f"Matrix constructed. Shape: {matrix.shape}")

    # 6. Standardization
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    
    # 7. Run PCA
    n_components = 5
    n_components = min(n_components, min(matrix_scaled.shape))
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(matrix_scaled)
    
    # --- SAVE GENERATED DATA ---
    print("\n--- Saving Processed Data ---")
    processed_dir = 'processed_data'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"Created directory: {processed_dir}")

    # 0. Create DF for PCs
    columns_pc = [f'PC{i+1}' for i in range(n_components)]
    pc_df = pd.DataFrame(data=principal_components, columns=columns_pc, index=matrix.index)

    # 1. Scores Matrix (N x K)
    pc_df.index.name = 'NOC'
    scores_path = os.path.join(processed_dir, 'pca_scores_matrix.csv')
    pc_df.to_csv(scores_path)
    print(f"1. Scores Matrix saved to {scores_path}")

    # 2. Explained Variance Ratio
    explained_variance_df = pd.DataFrame({
        'Principal Component': columns_pc,
        'Explained Variance Ratio': pca.explained_variance_ratio_
    })
    variance_path = os.path.join(processed_dir, 'pca_explained_variance_ratio.csv')
    explained_variance_df.to_csv(variance_path, index=False)
    print(f"2. Explained Variance Ratio saved to {variance_path}")

    # 3. Loadings Matrix (M x K)
    # pca.components_ shape is (n_components, n_features). We transpose it to (M, K)
    loadings_df = pd.DataFrame(data=pca.components_.T, index=matrix.columns, columns=columns_pc)
    loadings_df.index.name = 'Sport'
    loadings_path = os.path.join(processed_dir, 'pca_loadings_matrix.csv')
    loadings_df.to_csv(loadings_path)
    print(f"3. Loadings Matrix saved to {loadings_path}")

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # 8. Visualization
    # plot_biplot_advanced(principal_components, pca, matrix.columns, matrix.index)
    plot_loadings_heatmap(loadings_df, pca.explained_variance_ratio_)
    plot_scree_plot(pca)

def plot_loadings_heatmap(loadings_df, explained_variance):
    """
    Replicates the dot-plot heatmap for PCA loadings.
    X-axis: Sports
    Y-axis: Dimensions (PCs)
    Dot size/color: Magnitude of loading
    """
    sns.set_style("whitegrid")
    
    # Transpose for easier plotting with seaborn or scatter: 
    # We want Rows=Dimensions, Cols=Sports is standard for heatmap, but dotplot usually takes long form
    # We'll convert to long format: Sport, PC, Loading
    
    loadings_long = loadings_df.reset_index().melt(id_vars='Sport', var_name='PC', value_name='Loading')
    
    # Add absolute value for size
    loadings_long['AbsLoading'] = loadings_long['Loading'].abs()
    
    # For color, we generally use the absolute value (magnitude) or squared loading (cos2)
    # The reference image has color scale 0 -> 0.76 approx.
    # We will use AbsLoading for color and Size
    
    n_sports = len(loadings_df.index)
    
    plt.figure(figsize=(max(15, n_sports * 0.3), 6)) # Dynamic width based on number of sports
    
    # Sort sports by their maximum loading across the PCs to make it look organized
    # Replicating the order might be hard without data, but assuming grouping by dominant component
    # We can assign each sport to the PC where it has max loading
    
    # Calculate sort metrics on the numeric data only (before adding new columns)
    max_pcs = loadings_df.abs().idxmax(axis=1)
    max_vals = loadings_df.abs().max(axis=1)
    
    loadings_df['MaxPC'] = max_pcs
    loadings_df['MaxVal'] = max_vals
    
    # Sort by MaxPC then by MaxVal
    loadings_df_sorted = loadings_df.sort_values(by=['MaxPC', 'MaxVal'], ascending=[True, False])
    sorted_sports = loadings_df_sorted.index.tolist()
    
    # Reorder the long dataframe
    loadings_long['Sport'] = pd.Categorical(loadings_long['Sport'], categories=sorted_sports, ordered=True)
    
    # Create the scatter plot
    ax = sns.scatterplot(
        data=loadings_long, 
        x='Sport', 
        y='PC', 
        size='AbsLoading', 
        hue='AbsLoading',
        sizes=(20, 500), # Min and max size of dots
        palette='Blues', # Matches the blue theme in the image
        edgecolor='gray',
        linewidth=0.5
    )
    
    # Customizing the axes
    plt.xticks(rotation=90)
    plt.xlabel("") 
    plt.ylabel("")
    
    # We want PC1 top. 
    ax.invert_yaxis()

    # Move X axis labels to top? Reference has them on TOP.
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=90, fontsize=10)
    
    # Add grid
    plt.grid(True, linestyle='-', alpha=0.5)

    # Legend
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., title="Loading Strength")
    
    plt.title("PCA Loadings: Contribution of Sports to Principal Components", y=1.2, fontsize=16, weight='bold')
    
    plt.tight_layout()
    output_file = 'pca_loadings_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Loadings Heatmap saved to {output_file}")

def plot_biplot_advanced(score, pca, columns, index):
    # Set style
    sns.set_style("whitegrid")
    
    # Extract data for plotting
    xs = score[:, 0]
    ys = score[:, 1]
    coeff = pca.components_
    explained_variance = pca.explained_variance_ratio_
    
    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    # --- 1. Plot Individuals (Countries) ---
    # Create the distinct look from reference: Solid circles
    
    # Use distance from origin for sizing/coloring contribution
    dist = np.sqrt(xs**2 + ys**2)
    # Normalize size: Range 50 to 500
    sizes = (dist / dist.max()) * 400 + 100 
    
    # Color map for points: Gold/Orange/Red theme
    points = ax.scatter(xs, ys, c=dist, cmap='YlOrRd', s=sizes, alpha=1.0, edgecolors='white', linewidth=1.5, zorder=2)
    
    # Label mainly the outliers (far from origin)
    top_n_indices = np.argsort(dist)[-15:] 
    
    # Use adjust_text if available, else standard text
    try:
        from adjustText import adjust_text
        use_adjust_text = True
        texts = []
    except ImportError:
        use_adjust_text = False

    for i in top_n_indices:
        # Standard matplotlib text
        txt = ax.text(xs[i], ys[i], index[i], color='black', fontsize=10, weight='bold', ha='center', va='center')
        if use_adjust_text:
            texts.append(txt)

    if use_adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # --- 2. Plot Variables (Sports/Arrows) ---
    # Scale arrows to match score range
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    coeff_max_x = np.abs(coeff[0, :]).max()
    coeff_max_y = np.abs(coeff[1, :]).max()
    
    # Adjust scaling to fit well
    scale_x = (x_range / coeff_max_x) * 0.6 
    scale_y = (y_range / coeff_max_y) * 0.6
    scale = min(scale_x, scale_y)
    
    arrow_x = coeff[0, :] * scale
    arrow_y = coeff[1, :] * scale
    
    # Magnitude for coloring arrows
    arrow_magnitudes = np.sqrt(coeff[0, :]**2 + coeff[1, :]**2)
    
    # Filter variables to avoid clutter - keep top 30
    top_var_indices = np.argsort(arrow_magnitudes)[-30:] 
    
    # Colormap for arrows: Spectral_r (Blue -> Yellow -> Red)
    cmap_arrows = plt.get_cmap('Spectral_r') 
    norm_arrows = mcolors.Normalize(vmin=arrow_magnitudes.min(), vmax=arrow_magnitudes.max())

    for i in top_var_indices:
        x_end = arrow_x[i]
        y_end = arrow_y[i]
        
        color = cmap_arrows(norm_arrows(arrow_magnitudes[i]))
        
        # Plot arrow
        ax.arrow(0, 0, x_end, y_end, color=color, alpha=0.8, 
                 head_width=x_range*0.015, linewidth=1.5, length_includes_head=True, zorder=3)
        
        # Label with line connecting if needed, or just text
        text_pos_x = x_end * 1.15
        text_pos_y = y_end * 1.15
        
        # Draw a thin line from arrow tip to text for clarity
        ax.plot([x_end, text_pos_x], [y_end, text_pos_y], color=color, alpha=0.5, linewidth=0.8, linestyle='--')
        
        ax.text(text_pos_x, text_pos_y, columns[i], color=color, 
                ha='center', va='center', fontsize=9, zorder=4)

    # --- 3. Final Polish ---
    ax.axhline(0, linestyle='--', color='black', alpha=0.4, linewidth=1)
    ax.axvline(0, linestyle='--', color='black', alpha=0.4, linewidth=1)
    
    ax.set_xlabel(f"Dim 1 ({explained_variance[0]*100:.1f}%)", fontsize=14)
    ax.set_ylabel(f"Dim 2 ({explained_variance[1]*100:.1f}%)", fontsize=14)
    
    ax.set_title("PCA Biplot (2020 Olympics)", fontsize=16, weight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_arrows, norm=norm_arrows)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, aspect=40, pad=0.02)
    cbar.set_label('Contribution (cos2)', rotation=270, labelpad=20)

    # Legend for sizes
    l1 = plt.scatter([],[], s=100, c='gray', alpha=0.5)
    l2 = plt.scatter([],[], s=250, c='gray', alpha=0.5)
    l3 = plt.scatter([],[], s=500, c='gray', alpha=0.5)
    legend = plt.legend([l1, l2, l3], ["Low", "Med", "High"], title="Impact", frameon=True, loc='upper left')
    plt.setp(legend.get_title(),fontsize='small')

    plt.tight_layout()
    
    output_file = 'pca_biplot_improved.png'
    plt.savefig(output_file, dpi=300)
    print(f"Improved Biplot saved to {output_file}") 
def plot_scree_plot(pca):
    """
    复现论文 Figure 16: Scree Plot
    展示每个主成分解释的方差比例 (柱状图) 和 累计比例 (折线图)
    """
    # 1. 准备数据
    var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)
    
    # 构造 x 轴标签 (PC1, PC2, ...)
    num_components = len(var_ratio)
    ind = np.arange(num_components)
    labels = [f'PC{i+1}' for i in range(num_components)]
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # 2. 画柱状图 (Bar Chart) - 代表单个PC的重要性
    bars = ax.bar(ind, var_ratio, color='#4c72b0', alpha=0.9, label='Individual Variance')
    
    # 3. 画折线图 (Line Chart) - 代表累计重要性
    # 论文中通常会有两个轴，这里我们用同一个轴展示比例即可 (0.0 - 1.0)
    line = ax.plot(ind, cum_var_ratio, color='black', marker='o', linestyle='-', linewidth=2, markersize=6, label='Cumulative Variance')
    
    # 4. 标注数值
    # 标注累计值 (如论文中的 "Accumulated 76.4%")
    for i, v in enumerate(cum_var_ratio):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontsize=9, fontweight='bold')
    
    # 5. 美化图表
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1) # y轴范围 0% 到 110% 以便显示标签
    
    ax.set_xlabel('Dimensions (Principal Components)', fontsize=12)
    ax.set_ylabel('Percentage of Explained Variance', fontsize=12)
    ax.set_title('Scree Plot for Principal Component Selection', fontsize=14, fontweight='bold')
    
    # 添加图例
    ax.legend(loc='center right')
    
    # 添加网格
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('pca_scree_plot.png', dpi=300)
    print("Scree Plot saved to pca_scree_plot.png")

if __name__ == "__main__":
    main()
