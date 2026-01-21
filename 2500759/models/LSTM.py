import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# ==========================================
# 1. Data Preparation (数据准备)
# ==========================================

def load_and_preprocess_data():
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

    # 路径检查 (兼容不同运行环境)
    base_path = 'Data/'
    if not os.path.exists(base_path):
        base_path = '../Data/' # 尝试上级目录
        if not os.path.exists(base_path) and os.path.exists('summerOly_medal_counts.csv'):
             base_path = '.' # 尝试当前目录
    
    print(f"Loading data from {base_path}...")
    medal_df = pd.read_csv(os.path.join(base_path, 'summerOly_medal_counts.csv'))
    
    hosts_path = os.path.join(base_path, 'summerOly_hosts.csv')
    if os.path.exists(hosts_path):
        hosts_df = pd.read_csv(hosts_path)
    else:
        # Fallback if file missing
        print("Warning: hosts file not found")
        hosts_df = pd.DataFrame(columns=['Year', 'Host'])

    # --- 1.1 清洗东道主数据 ---
    def extract_country(host_str):
        if isinstance(host_str, float) or "Cancelled" in str(host_str):
            return None
        parts = str(host_str).split(',')
        if len(parts) > 1:
            raw_country = parts[-1].strip()
            # 使用字典映射，更清晰，也容易添加新的
            mapping = {
            "United Kingdom": "Great Britain",
            "Soviet Union": "Soviet Union", 
            "West Germany": "West Germany",
            "United States": "USA",         
            "China": "China",               
            "South Korea": "South Korea",
            "Russia": "Russia"
            }
        
            return mapping.get(raw_country, raw_country)
        return None

    if not hosts_df.empty:
        hosts_df['HostCountry'] = hosts_df['Host'].apply(extract_country)
        hosts_df = hosts_df.dropna(subset=['HostCountry'])
        hosts_df['Year'] = pd.to_numeric(hosts_df['Year'], errors='coerce')
        hosts_df = hosts_df.dropna(subset=['Year'])
        hosts_df['Year'] = hosts_df['Year'].astype(int)
    else:
        hosts_df['HostCountry'] = []
        hosts_df['Year'] = []

    # --- 1.2 计算主场优势系数 alpha_c (Equation 1) ---
    avg_gold = medal_df.groupby('NOC')['Gold'].mean()
    
    if not hosts_df.empty:
        hosted_medals = medal_df.merge(hosts_df, left_on=['Year', 'NOC'], right_on=['Year', 'HostCountry'], how='inner')
        avg_gold_when_hosting = hosted_medals.groupby('NOC')['Gold'].mean()
    else:
        avg_gold_when_hosting = pd.Series()
    
    alpha_c = {}
    all_nocs = medal_df['NOC'].unique()
    
    for noc in all_nocs:
        if noc in avg_gold_when_hosting.index and noc in avg_gold.index:
            g_host = avg_gold_when_hosting[noc]
            g_avg = avg_gold[noc]
            if g_avg > 0:
                alpha = (g_host - g_avg) / g_avg
            else:
                alpha = 0.0
            alpha_c[noc] = alpha
        else:
            alpha_c[noc] = 0.0
            
    # --- 1.3 构建序列数据 ---
    processed_dataset = []
    years = sorted(medal_df['Year'].unique())
    nocs = medal_df['NOC'].unique()
    
    # 论文设定 n_steps = 3 (回顾过去3届)
    seq_len = 3
    
    print(f"Processing sequences for {len(nocs)} countries over {len(years)} olympiads...")
    
    for noc in nocs:
        country_data = medal_df[medal_df['NOC'] == noc].sort_values('Year')
        
        # 构建该国完整的时间序列
        country_seq = []
        for year in years:
            # 1. Host Status
            is_host = 0
            if not hosts_df.empty:
                host_row = hosts_df[hosts_df['Year'] == year]
                if not host_row.empty and host_row.iloc[0]['HostCountry'] == noc:
                    is_host = 1
            
            # 2. Prep Status (Equation 2: t in [t_host-8, t_host])
            is_prep = 0
            if not hosts_df.empty:
                future_hosts = hosts_df[hosts_df['HostCountry'] == noc]['Year'].values
                for h_year in future_hosts:
                    if year >= h_year - 8 and year <= h_year:
                        is_prep = 1
                        break
                    
            # 3. Medals
            row = country_data[country_data['Year'] == year]
            if not row.empty:
                gold = row.iloc[0]['Gold']
                total = row.iloc[0]['Total']
            else:
                gold = 0
                total = 0
            
            # 4. Alpha
            alpha = alpha_c.get(noc, 0.0)
            
            # Vector: [Gold, Total, Host, Prep, Alpha]
            country_seq.append([gold, total, is_host, is_prep, alpha])
            
        country_seq = np.array(country_seq)
        
        # 滑动窗口生成样本
        if len(country_seq) > seq_len:
            for i in range(seq_len, len(country_seq)):
                # Input: t-3, t-2, t-1
                x_seq = country_seq[i-seq_len:i] 
                
                # Target: t
                y_target_gold = country_seq[i][0]
                y_target_total = country_seq[i][1]
                y_target_host = country_seq[i][2]
                
                # Current year metadata (for reconstruction later)
                curr_year_gold = country_seq[i][0]
                curr_year_total = country_seq[i][1]
                curr_year_host = country_seq[i][2]
                curr_year_prep = country_seq[i][3]
                curr_year_alpha = country_seq[i][4]

                processed_dataset.append({
                    'x_medal': x_seq[:, :2],   # [Gold, Total] for Channel 1
                    'x_host': x_seq[:, 2:],    # [Host, Prep, Alpha] for Channel 2
                    'y_gold': y_target_gold,
                    'y_total': y_target_total,
                    'y_host': y_target_host,
                    'noc': noc,
                    'year': years[i],
                    # Store current year ground truth features for exporting
                    'feat_gold': curr_year_gold,
                    'feat_total': curr_year_total,
                    'feat_host': curr_year_host,
                    'feat_prep': curr_year_prep,
                    'feat_alpha': curr_year_alpha
                })

    return processed_dataset, alpha_c, hosts_df

# ==========================================
# 2. Model Definition (模型定义)
# ==========================================

class DualChannelLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(DualChannelLSTM, self).__init__()
        
        # Channel 1: 关注奖牌历史
        self.lstm_medal = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        
        # Channel 2: 关注主场环境
        self.lstm_host = nn.LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)
        
        # 特征融合层
        self.fc = nn.Linear(2 * hidden_size, 3) 
        # Output dim 3: [Gold_pred, Total_pred, Host_Logit]
        
    def forward(self, x_medal, x_host):
        _, (h_medal, _) = self.lstm_medal(x_medal)
        _, (h_host, _) = self.lstm_host(x_host)
        
        # 取最后一个时间步的 hidden state
        h_medal = h_medal[-1]
        h_host = h_host[-1]
        
        # 拼接
        combined = torch.cat((h_medal, h_host), dim=1)
        output = self.fc(combined)
        
        pred_medals = output[:, :2]  # 回归预测
        pred_host_logit = output[:, 2] # 分类Logit
        
        return pred_medals, pred_host_logit

# ==========================================
# 3. Training & Evaluation (训练与评估)
# ==========================================

class OlympicDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        y_gold = float(item['y_gold'])
        y_total = float(item['y_total'])
        
        return (
            torch.tensor(item['x_medal'], dtype=torch.float32), # x_m
            torch.tensor(item['x_host'], dtype=torch.float32),  # x_h
            torch.tensor([y_gold, y_total], dtype=torch.float32), # y_true
            torch.tensor(item['y_host'], dtype=torch.float32),    # y_host_class
            torch.tensor([y_gold, y_total], dtype=torch.float32) 
        )

def wmae_loss(y_pred, y_true, weights):
    abs_diff = torch.abs(y_true - y_pred)
    sum_weights = torch.sum(weights)
    if sum_weights < 1e-6:
        return torch.mean(abs_diff)
    loss = torch.sum(weights * abs_diff) / sum_weights
    return loss

def train_model(hidden_size, patience, dataset, epochs=1000):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    model = DualChannelLSTM(hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion_medal = nn.MSELoss()
    criterion_host = nn.BCEWithLogitsLoss()
    lambda1, lambda2 = 0.7, 0.3
    
    best_val_wmae = float('inf') 
    early_stop_counter = 0
    final_wmae_gold = 0
    final_wmae_total = 0
    
    for epoch in range(epochs):
        model.train()
        for x_m, x_h, y_m, y_h, w in train_loader:
            optimizer.zero_grad()
            pred_m, pred_h = model(x_m, x_h)
            loss_m = criterion_medal(pred_m, y_m)
            loss_h = criterion_host(pred_h, y_h)
            loss = lambda1 * loss_m + lambda2 * loss_h
            loss.backward()
            optimizer.step()
            
        model.eval()
        wmae_g_sum = 0
        wmae_t_sum = 0
        total_batches = 0
        
        with torch.no_grad():
            for x_m, x_h, y_m, y_h, w in val_loader:
                pred_m, pred_h = model(x_m, x_h)
                batch_wmae_g = wmae_loss(pred_m[:, 0], y_m[:, 0], weights=w[:, 0])
                batch_wmae_t = wmae_loss(pred_m[:, 1], y_m[:, 1], weights=w[:, 1])
                wmae_g_sum += batch_wmae_g.item()
                wmae_t_sum += batch_wmae_t.item()
                total_batches += 1
        
        avg_wmae_gold = wmae_g_sum / total_batches if total_batches > 0 else 0
        avg_wmae_total = wmae_t_sum / total_batches if total_batches > 0 else 0
        current_metric = avg_wmae_gold + avg_wmae_total 
        
        if current_metric < best_val_wmae:
            best_val_wmae = current_metric
            early_stop_counter = 0
            final_wmae_gold = avg_wmae_gold
            final_wmae_total = avg_wmae_total
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            break
            
    return final_wmae_gold, final_wmae_total

def generate_lstm_features(model, raw_data, output_path, alpha_c, df_hosts):
    """
    Generate features including predicted trends and input vectors (Gold, Total, Host, Prep, Alpha).
    """
    model.eval()
    results = []
    
    # 1. Historical Data
    with torch.no_grad():
        for item in raw_data:
            x_m = torch.tensor(item['x_medal'], dtype=torch.float32).unsqueeze(0) 
            x_h = torch.tensor(item['x_host'], dtype=torch.float32).unsqueeze(0)
            
            pred_m, pred_h = model(x_m, x_h)
            
            results.append({
                'NOC': item['noc'],
                'Year': item['year'],
                # Predicted Trends
                'LSTM_Gold_Trend': pred_m[0, 0].item(),
                'LSTM_Total_Trend': pred_m[0, 1].item(),
                'LSTM_Host_Logit': pred_h[0].item(),
                # "5 Vectors" (Current Year Attributes - Ground Truth for history)
                'Gold': item['feat_gold'],
                'Total': item['feat_total'],
                'Is_Host': item['feat_host'],
                'Is_Prep': item['feat_prep'],
                'Alpha': item['feat_alpha'],
                'Is_History': True
            })
            
    # 2. Future Prediction (2028)
    # Reconstruct input for 2028: needs vals from 2016, 2020, 2024
    noc_history = defaultdict(list)
    for item in raw_data:
        noc_history[item['noc']].append(item)
        
    for noc, items in noc_history.items():
        items.sort(key=lambda x: x['year'])
        last_item = items[-1]
        
        if last_item['year'] == 2024:
            # Last item x_medal was [2012, 2016, 2020]
            # y_target was 2024 values
            prev_x_m = last_item['x_medal'] 
            prev_x_h = last_item['x_host']
            
            val_2024_gold = last_item['y_gold']
            val_2024_total = last_item['y_total']
            
            # New Input: [2016, 2020, 2024]
            new_x_m = np.vstack([prev_x_m[1:], [val_2024_gold, val_2024_total]])
            
            # Attributes for 2024 (Source for input)
            # We need attributes for the input years: 2016, 2020, 2024
            # prev_x_h is [2012_attr, 2016_attr, 2020_attr]
            # new_x_h should be [2016_attr, 2020_attr, 2024_attr]
            
            alpha = alpha_c.get(noc, 0.0)
            
            # Construct 2024 attributes
            # Check host for 2024 explicitly? or assume last_item ground truth is correct
            # In item['feat_...'] we have current year ground truth
            attr_2024 = [last_item['feat_host'], last_item['feat_prep'], alpha]
            
            new_x_h = np.vstack([prev_x_h[1:], attr_2024])
            
            # Predict Trend for 2028
            x_m_tensor = torch.tensor(new_x_m, dtype=torch.float32).unsqueeze(0)
            x_h_tensor = torch.tensor(new_x_h, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                pred_28_m, pred_28_h = model(x_m_tensor, x_h_tensor)
                
            # Attributes for 2028 (Target Year properties)
            # Need to check Is_Host for 2028
            is_host_28 = 0
            if not df_hosts.empty:
                host_28_row = df_hosts[df_hosts['Year'] == 2028]
                if not host_28_row.empty and str(noc) in str(host_28_row.iloc[0]['HostCountry']):
                    is_host_28 = 1
            
            # Is_Prep for 2028?
            is_prep_28 = 0 
            # Simplified logic if needed
            
            results.append({
                'NOC': noc,
                'Year': 2028,
                'LSTM_Gold_Trend': pred_28_m[0, 0].item(),
                'LSTM_Total_Trend': pred_28_m[0, 1].item(),
                'LSTM_Host_Logit': pred_28_h[0].item(),
                # "5 Vectors" (Features for 2028)
                'Gold': np.nan, # Unknown
                'Total': np.nan, # Unknown
                'Is_Host': is_host_28,
                'Is_Prep': is_prep_28,
                'Alpha': alpha,
                'Is_History': False
            })

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_path, index=False)
    print(f"LSTM Features (Trends + Vectors) saved to {output_path}")

# ==========================================
# 4. Main Experiment (主程序)
# ==========================================

def main():
    print("Pre-processing data...")
    raw_data, alpha_c, df_hosts = load_and_preprocess_data()
    if len(raw_data) == 0:
        print("Error: No data loaded. Check CSV paths.")
        return

    dataset = OlympicDataset(raw_data)
    
    hidden_sizes = [32, 64] # Reduced for speed in demo
    patiences = [10]
    
    results = [] 
    
    print(f"Starting Grid Search on {len(raw_data)} samples...")
    print(f"{'Hidden':<8} {'Patience':<10} {'WMAE_Gold':<12} {'WMAE_Total':<12}")
    
    for h in hidden_sizes:
        for p in patiences:
            wg, wt = train_model(h, p, dataset)
            results.append({
                'hidden_size': h,
                'patience': p,
                'WMAE_Gold': wg,
                'WMAE_Total': wt
            })
            print(f"{h:<8} {p:<10} {wg:<12.4f} {wt:<12.4f}")

    res_df = pd.DataFrame(results)
    output_path = 'processed_data/lstm_tuning_results.csv'
    res_df.to_csv(output_path, index=False)
    
    best_config = sorted(results, key=lambda x: x['WMAE_Gold'] + x['WMAE_Total'])[0]
    print(f"\nRetraining Best Model ({best_config['hidden_size']} hidden) for Feature Generation...")
    
    final_model = DualChannelLSTM(best_config['hidden_size'])
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion_medal = nn.MSELoss()
    criterion_host = nn.BCEWithLogitsLoss()
    
    final_model.train()
    for ep in range(100):
        for x_m, x_h, y_m, y_h, w in train_loader:
            optimizer.zero_grad()
            pred_m, pred_h = final_model(x_m, x_h)
            loss = criterion_medal(pred_m, y_m) + 0.3 * criterion_host(pred_h, y_h)
            loss.backward()
            optimizer.step()
            
    # Pass extra args needed for generation
    generate_lstm_features(final_model, raw_data, 'processed_data/lstm_generated_features.csv', alpha_c, df_hosts)
    
    plot_results(res_df)

def plot_results(df):
    if df.empty: return
    plt.figure(figsize=(10, 6))
    df['config'] = df['hidden_size'].astype(str) + '-' + df['patience'].astype(str)
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df['WMAE_Total'], width, label='WMAE Total')
    plt.bar(x + width/2, df['WMAE_Gold'], width, label='WMAE Gold')
    plt.xticks(x, df['config'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('processed_data/lstm_tuning_comparison.png')

if __name__ == "__main__":
    main()
