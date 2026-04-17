import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth
def fetch_data_advanced():
    chrome_options = Options()
    # 如果依然失败，请将下行改为 False，然后您可以手动在弹出的窗口里点验证码
    chrome_options.add_argument("--headless") 
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    print("🚀 正在启动防检测浏览器...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    # 抹除 Selenium 痕迹
    stealth(driver,
            languages=["zh-CN", "zh", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
    )
    url = "https://sc.macromicro.me/charts/99946/world-market-cap-to-gdp"
    print(f"🔗 正在访问: {url}")
    
    try:
        driver.get(url)
        
        # 等待侧边栏数据加载出来 (最多等 20 秒)
        print("⏳ 正在等待数据渲染（正在处理 Cloudflare 验证）...")
        wait = WebDriverWait(driver, 20)
        
        # 这里的 Selector 是关键，我们尝试一个更通用的路径
        sidebar_selector = ".chart-sidebar-info"
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sidebar_selector)))
        except:
            print("⚠️ 自动等待超时，可能是触发了验证码。如果使用了 Headless 模式，请尝试关闭它。")
        # 核心抓取逻辑：优先抓取侧边栏列表
        script = """
        let data = [];
        let items = document.querySelectorAll('.chart-sidebar-list li');
        items.forEach(item => {
            let name = item.querySelector('.text-start')?.innerText || '';
            let val = item.querySelector('.text-end')?.innerText || '';
            if(name) data.append({'国家/地区': name.trim(), '最新比值': val.trim()});
        });
        return data;
        """
        # 注意：上面的 append 在 JS 里应为 push
        script = script.replace(".append", ".push") 
        
        ratios = driver.execute_script(script)
        if not ratios:
            # 备选方案：直接抓取 DOM
            items = driver.find_elements(By.CSS_SELECTOR, ".chart-sidebar-list li")
            for item in items:
                name = item.find_element(By.CLASS_NAME, "text-start").text
                val = item.find_element(By.CLASS_NAME, "text-end").text
                ratios.append({'国家/地区': name, '最新比值': val})
        if ratios:
            print("\n✅ 抓取成功！")
            df = pd.DataFrame(ratios)
            print(df.to_string(index=False))
            df.to_csv("world_buffett_index.csv", index=False, encoding='utf-8-sig')
            print(f"\n📁 数据已保存至: world_buffett_index.csv")
        else:
            print("❌ 未能提取到数据，可能是页面结构已更新或被验证码阻挡。")
            
    except Exception as e:
        print(f"💥 运行时错误: {e}")
    finally:
        driver.quit()
if __name__ == "__main__":
    fetch_data_advanced()

# import requests
# import json
# import pandas as pd
# from datetime import datetime

# def fetch_macromicro_data(chart_id=99946):
#     """
#     尝试从 MacroMicro (M平方) 爬取巴菲特指数数据。
#     """
#     url = f"https://www.macromicro.me/charts/data/{chart_id}"
    
#     # 模拟浏览器请求头，防止被拦截
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#         "Referer": f"https://sc.macromicro.me/charts/{chart_id}/world-market-cap-to-gdp",
#         "Accept": "application/json, text/javascript, */*; q=0.01",
#         "X-Requested-With": "XMLHttpRequest"
#     }

#     print(f"正在尝试从 {url} 获取数据...")
    
#     try:
#         response = requests.get(url, headers=headers, timeout=15)
        
#         if response.status_code == 200:
#             data = response.json()
#             # 解析数据
#             chart_key = f"c:{chart_id}"
#             if "data" in data and chart_key in data["data"]:
#                 chart_info = data["data"][chart_key]
#                 series = chart_info.get("series", [])
#                 names = chart_info.get("names", [])
                
#                 if not series or not names:
#                     return None, "数据格式异常：未找到序列或名称"
                
#                 # 获取最新的一组数据
#                 latest_data = series[-1]
#                 date = latest_data[0]
#                 values = latest_data[1:]
                
#                 results = []
#                 for name, val in zip(names, values):
#                     if val is not None:
#                         results.append({
#                             "国家/地区": name,
#                             "最新比值 (%)": round(float(val), 2),
#                             "日期": date
#                         })
                
#                 return results, None
#             else:
#                 return None, "返回的数据中未发现目标图表 ID"
#         elif response.status_code == 403:
#             return None, "被 Cloudflare 拦截 (403)。建议在浏览器中访问一次该网站以解除限制，或使用 Selenium。"
#         else:
#             return None, f"请求失败，状态码: {response.status_code}"
            
#     except Exception as e:
#         return None, f"发生错误: {str(e)}"

# def list_crawlable_data():
#     """
#     列出该网页能够爬取的数据项
#     """
#     info = """
#     === 能够爬取的数据项列表 ===
#     1. 图表元数据:
#        - 标题: 世界-巴菲特指数 (市值/GDP)
#        - 描述: 关于巴菲特指数的定义及估值标准 (如 75%~90% 为合理区间)
#        - 数据来源: 各国交易所、世界银行等
       
#     2. 核心数值数据 (本脚本已实现):
#        - 各国/地区最新的市值与 GDP 比值 (%)
#        - 历史时间序列数据 (日期 + 各国比值)
       
#     3. 相关经济指标:
#        - 各种相关的 ETF 列表 (名称、价格、手续费、回报率)
#        - 侧边栏的辅助指标快照
#     ===========================
#     """
#     print(info)

# if __name__ == "__main__":
#     # 1. 列出可爬取项
#     list_crawlable_data()
    
#     # 2. 执行爬取
#     results, error = fetch_macromicro_data(99946)
    
#     if results:
#         print(f"\n成功获取最新比值数据 (更新于: {results[0]['日期']}):")
#         df = pd.DataFrame(results)
#         print(df.to_string(index=False))
        
#         # 保存到 CSV
#         filename = f"buffett_indicator_{datetime.now().strftime('%Y%m%d')}.csv"
#         df.to_csv(filename, index=False, encoding='utf-8-sig')
#         print(f"\n数据已保存至: {filename}")
#     else:
#         print(f"\n爬取失败: {error}")
        
#         print("\n[参考数据 - 页面快照值]:")
#         mock_data = [
#             {"国家/地区": "中国台湾", "最新比值 (%)": "爬取失败"},
#             {"国家/地区": "美国", "最新比值 (%)": "爬取失败"},
#             {"国家/地区": "日本", "最新比值 (%)": "爬取失败"},
#             {"国家/地区": "中国", "最新比值 (%)": "爬取失败"},
#             {"国家/地区": "德国", "最新比值 (%)": "爬取失败"}
#         ]
#         for item in mock_data:
#             print(f"{item['国家/地区']}: {item['最新比值 (%)']}")
