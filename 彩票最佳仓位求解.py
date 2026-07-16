import numpy as np

def calculate_kelly_numpy(outcomes):
    """
    使用 NumPy 计算多结局广义凯利公式的最佳下注比例
    
    :param outcomes: 二维 array-like 对象，格式为 [(概率, 净收益率), ...]
                     例如：[(0.01, 50.0), (0.05, 10.0), (0.94, -1.0)]
    :return: 最佳下注资金比例 f*
    """
    # 转换为 NumPy 浮点数组
    data = np.array(outcomes, dtype=np.float64)
    
    # 1. 结构检查
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("输入数据格式必须为二维，且每行包含(概率, 净收益率)两项。")
        
    probs = data[:, 0]
    returns = data[:, 1]
    
    # 2. 概率和检查（使用 np.isclose 避免浮点数精度误差导致判定失败）
    prob_sum = np.sum(probs)
    if not np.isclose(prob_sum, 1.0, atol=1e-5):
        raise ValueError(f"拒绝计算：所有结局的概率之和必须为 1.0（当前和为: {prob_sum:.6f}）。")
        
    # 3. 期望收益率检查
    expected_return = np.sum(probs * returns)
    if expected_return <= 0:
        print(f"提示：当前游戏的期望收益率为负或零 ({expected_return:.4f})，根据凯利公式，不应下注（最佳比例为 0）。")
        return 0.0

    # 4. 确定搜索区间上限 (max_f)
    # 如果有可能会亏损 (returns < 0)，f 的大小不能导致账户归零或穿仓。
    # 也就是说，对于所有的负收益 R_i，必须满足 1 + f * R_i > 0，即 f < -1 / R_i
    neg_returns = returns[returns < 0]
    if len(neg_returns) > 0:
        # 寻找最严苛的穿仓限制，并稍微留出安全裕度 (1e-9)
        max_f = -1.0 / np.min(neg_returns) - 1e-9
    else:
        max_f = 1.0 # 若无负收益，理论上可加杠杆，但实际限制在 100% 仓位
        
    # 限制最大下注比例不超过 1.0 (不考虑加杠杆负债买彩票的情况)
    max_f = min(max_f, 1.0)
    
    # 5. 二分法寻找导数为 0 的根
    low = 0.0
    high = max_f
    
    # 迭代 100 次，精度可达 1e-16 以上
    for _ in range(100):
        mid = (low + high) / 2.0
        # 计算当前 f 下的导数值：sum( p_i * R_i / (1 + f * R_i) )
        deriv = np.sum((probs * returns) / (1.0 + mid * returns))
        
        # 由于是凹函数，导数单调递减
        # 如果导数大于 0，说明还在极值点左侧，需要调大 f
        if deriv > 0:
            low = mid
        else:
            high = mid
            
    return round(low, 6)

# ==================== 测试运行 ====================
if __name__ == "__main__":
    # 案例一：正常的正期望彩票
    # 1%概率赢50倍，5%概率赢10倍，94%概率不中(-1)
    lottery_data = [
        (0.001, 500.0),
        (0.005, 100.0),
        (0.994, -1.0)
    ]
    
    try:
        f_star = calculate_kelly_numpy(lottery_data)
        print(f"案例一最佳下注比例: {f_star * 100:.4f}%")
    except ValueError as e:
        print(f"计算出错: {e}")

    # 案例二：概率和不为 1，触发拒绝机制
    invalid_data = [
        (0.1, 5.0),
        (0.8, -1.0) # 概率和为 0.9
    ]
    
    print("\n尝试运行不合规数据，请检查概率和是否为1")
    try:
        calculate_kelly_numpy(invalid_data)
    except ValueError as e:
        print(f"成功捕获异常: {e}")