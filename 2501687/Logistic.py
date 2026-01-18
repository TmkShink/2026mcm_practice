import numpy as np
import matplotlib.pyplot as plt

def logistic_model(Investment, alpha, Gamma_max, Gamma_init):
    # 对应论文公式 (10)
    # I_10 设为 0
    exponent = -alpha * (Investment - 0)
    denominator = 1 + (Gamma_max / Gamma_init - 1) * np.exp(exponent)
    return Gamma_max / denominator

# 论文给出的参数 [cite: 392-398]
alpha = 1e-4
Gamma_m = 1e5
Gamma_0 = 1e4

# 生成投资金额数据 (从 0 到 10万美元)
I = np.linspace(0, 100000, 100)
Returns = logistic_model(I, alpha, Gamma_m, Gamma_0)

# 画图
plt.plot(I, Returns)
plt.title("Logistic Model of Government Investment")
plt.xlabel("Investment ($)")
plt.ylabel("Benefit Value")
plt.grid(True)
plt.show()