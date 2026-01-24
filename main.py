"""定投策略模拟分析：比较有止盈和无止盈策略的效果"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List, Optional


class InvestmentStrategy:
    """投资策略基类"""

    def __init__(self, name: str, monthly_investment: float = 300.0,
                 investment_interval: int = 30):
        """
        初始化投资策略

        Parameters
        ----------
        name : str
            策略名称
        monthly_investment : float
            每月投资金额
        investment_interval : int
            投资间隔天数
        """
        self.name = name
        self.monthly_investment = monthly_investment
        self.investment_interval = investment_interval

        # 投资记录: (时间, 总成本, 净资产, 持有份额)
        self.records: List[Tuple[float, float, float, float]] = [(0, 0, 0, 0)]

        # 当前状态
        self.total_cost = 0.0  # 总投入成本
        self.current_shares = 0.0  # 当前持有份额
        self.cost_in_system = 0.0  # 系统内持仓成本（用于止盈计算）
        self.realized_profit = 0.0  # 已实现净利润


class NoStopProfitStrategy(InvestmentStrategy):
    """无止盈策略"""

    def __init__(self, monthly_investment: float = 300.0,
                 investment_interval: int = 30):
        super().__init__("无止盈策略", monthly_investment, investment_interval)

    def invest(self, time_point: float, price: float) -> None:
        """执行投资操作"""
        if int(round(time_point)) % self.investment_interval == 0:
            investment_amount = self.monthly_investment * self.investment_interval / 30

            self.total_cost += investment_amount
            self.current_shares += investment_amount / price

            current_asset = self.current_shares * price - self.total_cost
            self.records.append((time_point, self.total_cost, current_asset, self.current_shares))


class StopProfitStrategy(InvestmentStrategy):
    """有止盈策略"""

    def __init__(self, monthly_investment: float = 300.0,
                 investment_interval: int = 30,
                 take_profit_rate: float = 0.50,
                 sell_ratio: float = 0.5):
        """
        初始化止盈策略

        Parameters
        ----------
        take_profit_rate : float
            止盈阈值，当累计收益率达到该值时触发止盈
        sell_ratio : float
            卖出比例，触发止盈时卖出持仓的比例
        """
        super().__init__("有止盈策略", monthly_investment, investment_interval)
        self.take_profit_rate = take_profit_rate
        self.sell_ratio = sell_ratio

    def invest(self, time_point: float, price: float) -> None:
        """执行投资操作（先买入，后检查止盈）"""
        if int(round(time_point)) % self.investment_interval == 0:
            # 执行定投（买入）
            investment_amount = self.monthly_investment * self.investment_interval / 30
            self.total_cost += investment_amount
            self.cost_in_system += investment_amount
            self.current_shares += investment_amount / price

            # 检查并执行止盈
            if self.cost_in_system > 0 and self.current_shares > 0:
                cumulative_return = (self.current_shares * price - self.cost_in_system) / self.cost_in_system

                if cumulative_return >= self.take_profit_rate:
                    shares_to_sell = self.current_shares * self.sell_ratio
                    sell_proceeds = shares_to_sell * price
                    cost_of_shares_sold = self.cost_in_system * self.sell_ratio
                    net_profit = sell_proceeds - cost_of_shares_sold

                    # 更新已实现利润和系统内持仓
                    self.realized_profit += net_profit
                    self.current_shares -= shares_to_sell
                    self.cost_in_system -= cost_of_shares_sold

            # 计算当前净资产：系统内净值 + 已实现净利
            current_asset = (self.current_shares * price - self.cost_in_system +
                             self.realized_profit)
            self.records.append((time_point, self.cost_in_system,
                                 current_asset, self.current_shares))


def load_price_data(file_path: str, date_column: str = 'date',
                    price_column: Optional[str] = None) -> np.ndarray:
    """
    从CSV文件加载价格数据

    Parameters
    ----------
    file_path : str
        CSV文件路径
    date_column : str
        日期列名
    price_column : str, optional
        价格列名，如果为None则自动查找

    Returns
    -------
    np.ndarray
        价格序列
    """
    df = pd.read_csv(file_path, parse_dates=[date_column], dayfirst=False)

    # 确定价格列
    if price_column and price_column in df.columns:
        price_series = df[price_column].astype(float).values
    elif 'close' in df.columns:
        price_series = df['close'].astype(float).values
    else:
        numeric_columns = df.select_dtypes(include=[float, int]).columns
        if len(numeric_columns) == 0:
            raise ValueError("CSV文件中没有找到数值型的价格列")
        price_series = df[numeric_columns[0]].astype(float).values

    return price_series


def plot_results(time_points: np.ndarray, price_series: np.ndarray,
                 strategy1: InvestmentStrategy, strategy2: InvestmentStrategy) -> None:
    """
    绘制分析结果图表

    Parameters
    ----------
    time_points : np.ndarray
        时间点序列
    price_series : np.ndarray
        价格序列
    strategy1 : InvestmentStrategy
        策略1（通常为无止盈策略）
    strategy2 : InvestmentStrategy
        策略2（通常为有止盈策略）
    """
    # 准备数据
    times1 = np.array([record[0] for record in strategy1.records])
    assets1 = np.array([record[2] for record in strategy1.records])
    shares1 = np.array([record[3] for record in strategy1.records])

    times2 = np.array([record[0] for record in strategy2.records])
    assets2 = np.array([record[2] for record in strategy2.records])
    shares2 = np.array([record[3] for record in strategy2.records])

    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 价格曲线
    axes[0].plot(time_points, price_series, linewidth=1)
    axes[0].set_title("价格曲线", fontsize=14)
    axes[0].set_xlabel("时间（天）")
    axes[0].set_ylabel("价格")
    axes[0].grid(True, alpha=0.3)

    # 净资产对比
    axes[1].plot(times1, assets1, 'r-', linewidth=2, label=strategy1.name)
    axes[1].plot(times2, assets2, 'b-', linewidth=2, label=strategy2.name)
    axes[1].set_title("净资产对比", fontsize=14)
    axes[1].set_xlabel("时间（天）")
    axes[1].set_ylabel("净资产")
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # 计算并显示最终收益
    final_return1 = assets1[-1] / strategy1.total_cost * 100 if strategy1.total_cost > 0 else 0
    final_return2 = assets2[-1] / strategy2.total_cost * 100 if strategy2.total_cost > 0 else 0
    axes[1].text(0.02, 0.98, f'{strategy1.name}: {final_return1:.1f}%\n{strategy2.name}: {final_return2:.1f}%',
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 持有份额对比
    axes[2].plot(times1, shares1, 'r-', linewidth=2, label=strategy1.name)
    axes[2].plot(times2, shares2, 'b-', linewidth=2, label=strategy2.name)
    axes[2].set_title("持有份额对比", fontsize=14)
    axes[2].set_xlabel("时间（天）")
    axes[2].set_ylabel("持有份额")
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print("\n=== 投资策略统计 ===")
    print(f"{strategy1.name}:")
    print(f"  最终净资产: {assets1[-1]:.2f}")
    print(f"  总投入成本: {strategy1.total_cost:.2f}")
    print(f"  最终收益率: {final_return1:.2f}%")
    print(f"  最终持有份额: {shares1[-1]:.2f}")

    print(f"\n{strategy2.name}:")
    print(f"  最终净资产: {assets2[-1]:.2f}")
    print(f"  总投入成本: {strategy2.total_cost:.2f}")
    print(f"  已实现利润: {strategy2.realized_profit:.2f}")
    print(f"  最终收益率: {final_return2:.2f}%")
    print(f"  最终持有份额: {shares2[-1]:.2f}")


def main():
    """主函数：执行定投策略模拟"""
    # 参数配置
    MONTHLY_INVESTMENT = 300.0  # 每月投资金额
    INVESTMENT_INTERVAL = 30  # 投资间隔天数

    # 止盈策略参数
    TAKE_PROFIT_RATE = 0.50  # 止盈阈值（50%）
    SELL_RATIO = 0.5  # 卖出比例（50%）

    # 数据文件路径
    DATA_FILE_PATH = "sse_data.csv"

    try:
        # 加载价格数据
        print("正在加载价格数据...")
        price_series = load_price_data(DATA_FILE_PATH)

        # 创建时间轴（假设每天一个数据点）
        time_points = np.arange(0, len(price_series), 1)

        print(f"数据加载完成，共 {len(price_series)} 个数据点")
        print(f"价格范围: {price_series.min():.2f} - {price_series.max():.2f}")

        # 初始化投资策略
        strategy_no_stop = NoStopProfitStrategy(
            monthly_investment=MONTHLY_INVESTMENT,
            investment_interval=INVESTMENT_INTERVAL
        )

        strategy_with_stop = StopProfitStrategy(
            monthly_investment=MONTHLY_INVESTMENT,
            investment_interval=INVESTMENT_INTERVAL,
            take_profit_rate=TAKE_PROFIT_RATE,
            sell_ratio=SELL_RATIO
        )

        # 执行投资模拟
        print("\n正在执行投资模拟...")
        for i, (t, price) in enumerate(zip(time_points, price_series)):
            strategy_no_stop.invest(t, price)
            strategy_with_stop.invest(t, price)

            # 显示进度
            if i % 100 == 0 and i > 0:
                print(f"  已处理 {i}/{len(time_points)} 个时间点")

        print("投资模拟完成！")

        # 绘制结果
        plot_results(time_points, price_series, strategy_no_stop, strategy_with_stop)

    except FileNotFoundError:
        print(f"错误：找不到数据文件 '{DATA_FILE_PATH}'")
        print("请确保文件路径正确，或使用示例数据")

        # 使用示例数据（正弦波）作为替代
        print("\n使用示例数据继续运行...")
        example_time_points = np.arange(0, 365 * 2, 1)
        example_price_series = 100 + 20 * np.sin(2 * np.pi / 100 * example_time_points)

        # 初始化策略并执行模拟
        strategy_no_stop = NoStopProfitStrategy(MONTHLY_INVESTMENT, INVESTMENT_INTERVAL)
        strategy_with_stop = StopProfitStrategy(MONTHLY_INVESTMENT, INVESTMENT_INTERVAL,
                                                TAKE_PROFIT_RATE, SELL_RATIO)

        for t, price in zip(example_time_points, example_price_series):
            strategy_no_stop.invest(t, price)
            strategy_with_stop.invest(t, price)

        plot_results(example_time_points, example_price_series,
                     strategy_no_stop, strategy_with_stop)

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()