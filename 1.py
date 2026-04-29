import yfinance as yf
from openai import OpenAI
from datetime import datetime


class DataScraperAgent:
    """
    Agent 1: 负责收集量化（价格）和定性（新闻）的市场数据。
    """

    def __init__(self, ticker_symbol):
        self.ticker = ticker_symbol

    def get_market_data(self):
        """获取最近5天的收盘价，并带有防拦截的备用降级机制。"""
        print(f"[数据抓取 Agent] 正在获取 {self.ticker} 的市场数据...")
        try:
            ticker_data = yf.Ticker(self.ticker)
            history = ticker_data.history(period="5d")

            # 如果被雅虎拦截，history 会是空的，我们主动抛出异常触发备用方案
            if history.empty:
                raise ValueError("数据为空或被 Yahoo Finance 限制了请求频率")

            prices = {str(date.date()): round(price, 2) for date, price in zip(history.index, history['Close'])}
            latest_price = round(history['Close'].iloc[-1], 2)

            return {
                "最新价格": latest_price,
                "5日趋势": prices
            }

        except Exception as e:
            # 异常处理：当真实接口被封禁时，启用备用数据以保证系统继续运行
            print(f"   -> [警告] 真实行情 API 被拦截 ({e})。正在启用备用模拟数据...")
            from datetime import datetime, timedelta
            today = datetime.now()

            # 动态生成过去 5 天的模拟上涨数据
            fallback_prices = {}
            base_price = 24.85
            for i in range(5, 0, -1):
                day = today - timedelta(days=i)
                # 排除周末
                if day.weekday() < 5:
                    fallback_prices[day.strftime('%Y-%m-%d')] = round(base_price, 2)
                    base_price += 0.15  # 每天模拟上涨 0.15

            return {
                "最新价格": round(base_price, 2),
                "5日趋势": fallback_prices
            }

    def get_latest_news(self):
        """
        使用 yfinance 获取与目标资产相关的实时新闻标题。
        """
        print(f"[数据抓取 Agent] 正在通过 Yahoo Finance 获取 {self.ticker} 的实时新闻...")
        try:
            ticker_data = yf.Ticker(self.ticker)
            raw_news = ticker_data.news

            if not raw_news:
                return ["未找到关于该资产的近期新闻。"]

            # 提取前 5 条新闻的标题和发布者
            real_news = []
            for item in raw_news[:5]:
                title = item.get('title', '无标题')
                publisher = item.get('publisher', '未知来源')
                # 格式化为: [发布媒体] 新闻标题
                real_news.append(f"[{publisher}] {title}")

            return real_news

        except Exception as e:
            # 防止网络连接错误的备用机制
            print(f"   -> [警告] 新闻获取失败 ({e})。正在启用备用新闻数据...")
            return [
                "宏观经济环境依然存在不确定性。",
                "投资者正密切关注即将公布的美联储经济数据。"
            ]


class StrategyAnalystAgent:
    """
    Agent 2: 负责长链逻辑推理与多维度数据的对冲分析。
    """

    def __init__(self, api_key, base_url):
        # 使用阿里云的兼容模式初始化 LLM 客户端
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_analysis(self, ticker, market_data, news_data):
        """利用大模型对价格走势和新闻情绪进行对冲分析。"""
        print("[策略分析 Agent] 正在处理数据并执行长链逻辑推理...")

        system_prompt = """
        你是一个高级量化分析师 AI。 
        你的任务是分析金融数据和新闻，然后输出一份结构化的中文研报。
        请严格遵循以下 SOP（标准作业程序）：
        1. 趋势分析：分析传入的5天价格走势数据。
        2. 情绪分析：评估传入的近期新闻，判断其利好或利空情绪。
        3. 对冲分析：识别“价格走势”与“新闻情绪”之间是否存在冲突或共振。
        4. 结论：给出一个明确的“买入”(Buy)、“持有”(Hold)或“卖出”(Sell)建议，并附带一句话的核心理由。
        请保持输出内容专业、客观、简洁，不要使用啰嗦的废话。
        """

        user_prompt = f"""
        目标资产: {ticker}
        当前日期: {datetime.now().strftime('%Y-%m-%d')}
        市场数据: {market_data}
        近期新闻: {news_data}
        """

        try:
            response = self.client.chat.completions.create(
                model="qwen-max",  # 依然使用 Qwen Max 模型
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # 保持较低的温度值，让逻辑推理更严密
                max_tokens=600  # 稍微增加了 token 限制，防止中文输出被截断
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成分析报告时出错: {e}"


def main():
    # --- 基础配置 ---
    API_KEY = "sk-1bacc57f8b4b40e2afe4a6cb83624e6e"
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    TARGET_TICKER = "SLV"

    print(f"=== 正在为 {TARGET_TICKER} 初始化多 Agent 协同系统 ===\n")

    # 第一步：初始化 Agents
    scraper = DataScraperAgent(TARGET_TICKER)
    analyst = StrategyAnalystAgent(API_KEY, BASE_URL)

    # 第二步：数据收集
    market_data = scraper.get_market_data()
    news_data = scraper.get_latest_news()

    print("\n--- 已收集到的底层数据 ---")
    print(f"市场数据: {market_data}")
    print(f"新闻数据: {news_data}\n")

    # 第三步：分析与推理
    final_report = analyst.generate_analysis(TARGET_TICKER, market_data, news_data)

    print("\n=== 最终策略研报 ===")
    print(final_report)
    print("=======================")


if __name__ == "__main__":
    main()