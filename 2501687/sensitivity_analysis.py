import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 引入核心计算逻辑 (保持与 PSO.py 一致)
# ==========================================

# --- 终极修正参数 (Final Fix Parameters) ---
S1_FIXED = 7774865.0 
BETA_1 = 2.155
BETA_2 = 1.0
GAMMA_1M = 120000.0  # 环境上限
GAMMA_2M = 100000.0  # 社会上限
GAMMA_0 = 10000.0
ALPHA = 1e-4
I_0 = 0
ERI_MAX = 200000.0
BETA = 1e-4
PEAK_START = 121
PEAK_END = 270
DAYS_PEAK = 150
DAYS_OFF = 215
P_PROFIT = 85.0  
B_CONST = 2000000 

def objective_function(x):
    """
    计算目标函数值 (返回负效用)
    """
    c1, c2, I, gamma1, x1, x2 = x
    
    # --- 硬性约束 (Penalty) ---
    penalty = 0
    if c1 < c2: penalty += 1e9
    if x1 < x2: penalty += 1e9
    if I < 0: penalty += 1e9
    if I > B_CONST * P_PROFIT: penalty += 1e9
    total_tourists_est = c1 * 150 + c2 * 215
    if x1 > total_tourists_est * 0.25 * P_PROFIT: penalty += 1e9

    if penalty > 0: return -1e15 # 违反约束返回极小值

    # --- 核心计算 ---
    gamma2 = 1 - gamma1
    I_daily = I / 365.0
    
    def logistic_gain(amount, gamma_val, limit):
        denom = 1 + (limit/GAMMA_0 - 1) * np.exp(-ALPHA * (gamma_val * amount - I_0))
        return limit / denom

    G_env = logistic_gain(I, gamma1, GAMMA_1M)
    G_soc = logistic_gain(I, gamma2, GAMMA_2M)

    total_U = 0
    S1_daily = S1_FIXED / 365.0
    
    # 拥堵系数
    CONGESTION_FACTOR = 0.0028 
    # 税收阻力
    TAX_QUAD_FACTOR = 8.0e-6 

    # 补贴挂钩惩罚
    target_subsidy = -2.5 * c2 * (DAYS_OFF / 365.0) 
    subsidy_penalty = 0
    if x2 > target_subsidy: 
        subsidy_penalty = -10.0 * abs(x2 - target_subsidy)

    for t in range(1, 366):
        is_peak = (PEAK_START <= t <= PEAK_END)
        if is_peak:
            N_t = c1
            f_t = x1 / DAYS_PEAK 
            cong_coeff = CONGESTION_FACTOR
            daily_subsidy_pen = 0
        else:
            N_t = c2
            f_t = x2 / DAYS_OFF
            cong_coeff = CONGESTION_FACTOR * 5.0 
            daily_subsidy_pen = subsidy_penalty / DAYS_OFF

        # P (Profit)
        P_t = (N_t * P_PROFIT) + (f_t - I_daily)
        # E (Environment)
        E_carbon = - N_t * 12.5 
        E_eri = (1 / (1 + BETA * N_t)) * ERI_MAX
        E_inv = G_env / 365.0
        E_t = E_carbon + E_eri + E_inv
        # S (Society)
        S_pos = S1_daily
        S_neg = -1 * cong_coeff * (N_t ** 2)
        if f_t > 0:
            S_tax_pain = -1 * TAX_QUAD_FACTOR * (f_t * DAYS_PEAK) ** 2 / 365.0
        else:
            S_tax_pain = 0
        S_soc_inv = G_soc / 365.0
        
        S_t = S_pos + S_neg + S_tax_pain + S_soc_inv + daily_subsidy_pen
        total_U += (P_t + E_t + S_t)

    return -total_U # 返回的是负数 (Negative Objective Value)

# ==========================================
# 2. 敏感性分析绘图逻辑
# ==========================================

# 定义基准最优解 (Standard Value, sigma)
# 这里填入你刚才 PSO 跑出来的最优结果
best_params = [
    13993,      # c1
    1035,       # c2
    92310.46,   # I
    0.5433,     # gamma1
    195643.33,  # x1
    -5797.52    # x2
]

# 变量名称映射
var_names = ['c1', 'c2', 'I', 'gamma1', 'x1', 'x2']
var_labels = ['Sensitivity Analysis of c1', 
              'Sensitivity Analysis of c2', 
              'Sensitivity Analysis of I', 
              'Sensitivity Analysis of gamma1']

# 设置画布
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# 需要分析的前4个变量 (c1, c2, I, gamma1)
target_indices = [0, 1, 2, 3]

print("开始生成敏感性分析图表...")

for i, idx in enumerate(target_indices):
    ax = axes[i]
    current_val = best_params[idx]
    
    # --- 生成扫描区间 ---
    if idx == 3: # 如果是 gamma1
        # 论文图中 gamma1 是从 0 到 1 的完整曲线 (U型)
        x_range = np.linspace(0.01, 0.99, 50) 
    else:
        # 其他变量：论文通过 Monte Carlo 选取 sigma +/- 5%
        # 这里用 linspace 模拟 "101 values at equal intervals"
        lower_bound = current_val * 0.95
        upper_bound = current_val * 1.05
        x_range = np.linspace(lower_bound, upper_bound, 101)
    
    y_values = []
    
    # --- 计算每个点的效用 ---
    for val in x_range:
        # 复制一份参数，避免修改原列表
        temp_params = best_params.copy()
        # 修改当前变量的值
        temp_params[idx] = val
        
        # 计算效用 (得到的是负值)
        # 注意：论文图的 Y 轴是 "Negative Objective Value"
        # 我们的 objective_function 返回的就是 -U，正是图上需要的
        y_val = objective_function(temp_params)
        y_values.append(y_val)
    
    # --- 绘图 ---
    ax.plot(x_range, y_values, 'o-', markersize=3, linewidth=1)
    ax.set_title(var_labels[i])
    ax.set_xlabel(var_names[idx])
    ax.set_ylabel('Negative Objective Value')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 标记最优值的位置
    ax.axvline(current_val, color='r', linestyle='--', alpha=0.5, label='Optimal')

plt.tight_layout()
plt.show()

print("绘图完成。请检查是否复现了论文中的曲线形态。")