import logging
from logging.handlers import RotatingFileHandler
import os
import json
import time
import pandas as pd
import numpy as np
import okx.Trade as Trade
import okx.MarketData as MarketData
import okx.Account as Account
import okx.PublicData as PublicData
import traceback
import uuid

# 日志配置
logging.basicConfig(
    handlers=[RotatingFileHandler('spot_trading_bot.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CONFIG_FILE = 'okx_config.json'

# ==========================================
# 内置交易参数
# ==========================================
SYMBOL = 'BTC-USDT'           # 现货交易对
TIMEFRAME = '1D'             # K线周期（日线）
RSI_PERIOD = 14              # RSI周期
RSI_MA_PERIOD = 10           # RSI均线周期
TRADE_QTY = 0.00001            # 每次固定交易的BTC数量
# ==========================================

trade_client = None
market_client = None
account_client = None
public_client = None
state = {SYMBOL: {'current_price': 0.0, 'latest_rsi': None, 'latest_rsi_ma': None}}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except Exception as e:
            logging.error(f"加载配置文件失败: {str(e)}")
    return {}

def init_okx(api_key, api_secret, passphrase, flag='0'):
    global trade_client, market_client, account_client, public_client
    try:
        trade_client = Trade.TradeAPI(api_key, api_secret, passphrase, use_server_time=False, flag=flag)
        market_client = MarketData.MarketAPI(flag=flag)
        account_client = Account.AccountAPI(api_key, api_secret, passphrase, use_server_time=False, flag=flag)
        public_client = PublicData.PublicAPI(flag=flag)
        response = account_client.get_account_balance()
        if response.get('code') != '0':
            raise Exception(f"账户余额检查失败: {response.get('msg', '未知错误')}")
        logging.info(f"OKX 现货 API 初始化成功，flag={flag}")
        return True
    except Exception as e:
        logging.error(f"OKX API 初始化失败: {str(e)}\n{traceback.format_exc()}")
        return False

def get_price(symbol):
    for attempt in range(3):
        try:
            ticker = market_client.get_ticker(instId=symbol)
            if ticker.get('code') != '0':
                raise Exception(f"获取行情失败: {ticker.get('msg', '未知错误')}")
            price = float(ticker['data'][0]['last'])
            state[symbol]['current_price'] = price
            return price
        except Exception as e:
            logging.error(f"获取价格失败: {symbol} (尝试 {attempt + 1}/3): {str(e)}")
            time.sleep(2)
    return None

def get_klines(symbol, interval, limit=100):
    for attempt in range(5):
        try:
            klines = market_client.get_candlesticks(instId=symbol, bar=interval, limit=str(limit))
            if not klines.get('data'):
                return None
            df = pd.DataFrame(klines['data'], columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
            df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
            df = df.sort_values('ts').reset_index(drop=True)
            if df.empty:
                return None
            df = df.dropna(subset=['close'])
            df = df[df['close'] > 0]
            
            min_required = RSI_PERIOD + RSI_MA_PERIOD
            if len(df) < min_required:
                logging.warning(f"K线数据不足: 现有 {len(df)} 条，计算需要 {min_required} 条")
                return None
            return df
        except Exception as e:
            logging.error(f"获取K线错误: {symbol} (尝试 {attempt + 1}/5): {str(e)}")
            time.sleep(2)
    return None

def calculate_rsi_and_ma(df, rsi_period=14, ma_period=20, current_price=None):
    try:
        if current_price is not None:
            df = df.copy()
            df.iloc[-1, df.columns.get_loc('close')] = current_price

        close = df['close'].values
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gains[:rsi_period])
        avg_loss = np.mean(losses[:rsi_period])
        
        rsi = np.zeros(len(close))
        rsi[:rsi_period] = np.nan
        rsi[rsi_period] = 100 - (100 / (1 + (avg_gain / (avg_loss if avg_loss != 0 else 1e-10))))
        
        for i in range(rsi_period + 1, len(close)):
            avg_gain = (avg_gain * (rsi_period - 1) + gains[i - 1]) / rsi_period
            avg_loss = (avg_loss * (rsi_period - 1) + losses[i - 1]) / rsi_period
            rs = avg_gain / (avg_loss if avg_loss != 0 else 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
            
        rsi_series = pd.Series(rsi, index=df.index)
        rsi_ma_series = rsi_series.rolling(window=ma_period).mean()
        
        logging.info(f"指标计算完成 - 当前 RSI: {rsi_series.iloc[-1]:.2f} | RSI_MA: {rsi_ma_series.iloc[-1]:.2f}")
        return rsi_series.iloc[-1], rsi_ma_series.iloc[-1]
    except Exception as e:
        logging.error(f"RSI及均线计算错误: {str(e)}")
        return None, None

def get_symbol_info(symbol):
    try:
        info = public_client.get_instruments(instType='SPOT', instId=symbol)
        if info.get('code') != '0':
            raise Exception(f"获取交易对信息失败: {info.get('msg')}")
        min_qty = float(info['data'][0]['minSz'])
        lot_sz = float(info['data'][0].get('lotSz', min_qty))
        return min_qty, lot_sz
    except Exception as e:
        logging.error(f"获取交易对精度失败: {symbol}, {str(e)}")
        return 0.00001, 0.00001

def place_spot_order(side):
    try:
        min_qty, lot_sz = get_symbol_info(SYMBOL)
        quantity = max(round(TRADE_QTY / lot_sz) * lot_sz, min_qty)
        decimal_places = int(-np.log10(lot_sz)) if lot_sz < 1 else 0
        sz_str = f"{quantity:.{decimal_places}f}"

        order_params = {
            'instId': SYMBOL,
            'tdMode': 'cash',
            'side': side.lower(),
            'ordType': 'market',
            'sz': sz_str,
            'clOrdId': str(uuid.uuid4()).replace('-', '')[:32]
        }
        
        # 市价买单以BTC数量为单位时，必须指定tgtCcy为base_ccy
        if side.lower() == 'buy':
            order_params['tgtCcy'] = 'base_ccy'
            
        order = trade_client.place_order(**order_params)
        if order['code'] == '0':
            logging.info(f"下单成功 - 方向: {side}, 数量: {sz_str} BTC")
            return order['data'][0]['ordId']
        else:
            logging.error(f"下单失败: {order['msg']}")
            return None
    except Exception as e:
        logging.error(f"下单异常: {str(e)}")
        return None

def get_spot_balances():
    for attempt in range(3):
        try:
            balance = account_client.get_account_balance()
            if balance.get('code') != '0':
                raise Exception(f"获取余额失败: {balance.get('msg')}")
            
            usdt = btc = 0.0
            if balance.get('data') and balance['data'][0].get('details'):
                details = balance['data'][0]['details']
                usdt_asset = next((asset for asset in details if asset['ccy'] == 'USDT'), None)
                btc_asset = next((asset for asset in details if asset['ccy'] == 'BTC'), None)
                if usdt_asset:
                    usdt = float(usdt_asset.get('availEq', 0.0) or 0.0)
                if btc_asset:
                    btc = float(btc_asset.get('availEq', 0.0) or 0.0)
            return usdt, btc
        except Exception as e:
            logging.error(f"获取币币余额失败 (尝试 {attempt + 1}/3): {str(e)}")
            time.sleep(2)
    return 0.0, 0.0

def execute_trading_logic():
    try:
        df_klines = get_klines(SYMBOL, TIMEFRAME, limit=100)
        if df_klines is None: return False
            
        price = get_price(SYMBOL)
        if not price: return False
            
        latest_rsi, latest_rsi_ma = calculate_rsi_and_ma(df_klines, rsi_period=RSI_PERIOD, ma_period=RSI_MA_PERIOD, current_price=price)
        if latest_rsi is None or latest_rsi_ma is None: return False
            
        state[SYMBOL]['latest_rsi'] = latest_rsi
        state[SYMBOL]['latest_rsi_ma'] = latest_rsi_ma
        
        usdt_balance, btc_balance = get_spot_balances()
        logging.info(f"账户余额 - USDT: {usdt_balance:.2f} | BTC: {btc_balance:.6f}")
        
        # 策略核心逻辑
        if latest_rsi < latest_rsi_ma:
            # 防止重复买入：只有当持有的BTC小于交易阈值时才买入
            if btc_balance < TRADE_QTY:
                required_usdt = TRADE_QTY * price
                if usdt_balance >= required_usdt:
                    logging.info(f"触发买入信号: RSI({latest_rsi:.2f}) < MA({latest_rsi_ma:.2f})")
                    place_spot_order('buy')
                else:
                    logging.warning(f"余额不足，取消买入。需要USDT: {required_usdt:.2f}, 当前: {usdt_balance:.2f}")
            else:
                logging.info("满足买入信号，但当前已持有BTC，跳过交易")
                
        elif latest_rsi > latest_rsi_ma:
            # 只有持有足够BTC时才触发卖出
            if btc_balance >= TRADE_QTY:
                logging.info(f"触发卖出信号: RSI({latest_rsi:.2f}) > MA({latest_rsi_ma:.2f})")
                place_spot_order('sell')
            else:
                logging.info("满足卖出信号，但当前未持有足够BTC，跳过交易")
        else:
            logging.info("信号持平，无操作")
            
        return True
    except Exception as e:
        logging.error(f"执行交易逻辑异常: {str(e)}")
        return False

def main():
    try:
        config = load_config()
        # 优先读取本地配置文件，没有则读取 GitHub Actions 注入的环境变量
        api_key = config.get('api_key') or os.getenv('OKX_API_KEY')
        api_secret = config.get('api_secret') or os.getenv('OKX_API_SECRET')
        passphrase = config.get('passphrase') or os.getenv('OKX_PASSPHRASE')
        
        if not all([api_key, api_secret, passphrase]):
            raise ValueError("API密钥未配置，请检查配置文件或环境变量")

        if init_okx(api_key, api_secret, passphrase, flag='0'):
            execute_trading_logic()
    except Exception as e:
        logging.error(f"主程序错误: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()