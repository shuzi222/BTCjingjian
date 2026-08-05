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
SYMBOL = 'BTC-USDT-SWAP'      # 永续合约交易对
TIMEFRAME = '1H'              # K线周期（一小时线）
RSI_PERIOD = 14               # RSI 周期
RSI_MA_PERIOD = 14            # RSI 均线周期
LEVERAGE = 3                  # 杠杆倍数
MARGIN_MODE = 'cross'         # 保证金模式: cross(全仓) / isolated(逐仓)
# ---------- 开多参数 ----------
BUY_QTY = 0.0001              # 每次开多的 BTC 数量
BUY_TAKE_PROFIT = 2           # 开多止盈百分比
BUY_STOP_LOSS = 2             # 开多止损百分比
# ---------- 平多参数 ----------
SELL_QTY = 0.0001             # 每次平多的 BTC 数量
# ---------- 信号阈值 ----------
RSI_BUY_THRESHOLD = 4         # RSI 需低于 RSI_MA 多少才开多（正数=更大缓冲）
RSI_SELL_THRESHOLD = 4        # RSI 需高于 RSI_MA 多少才平多（正数=更大缓冲）
# ---------- 通用开关 ----------
ENABLE_TP_SL = True           # 开多时是否附带止盈止损单
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
        logging.info(f"OKX 合约 API 初始化成功，flag={flag}")
        return True
    except Exception as e:
        logging.error(f"OKX API 初始化失败: {str(e)}\n{traceback.format_exc()}")
        return False


def set_leverage():
    try:
        response = account_client.set_leverage(
            instId=SYMBOL, lever=str(LEVERAGE), mgnMode=MARGIN_MODE
        )
        if response.get('code') == '0':
            logging.info(f"设置杠杆成功: {SYMBOL}, 杠杆={LEVERAGE}x, 模式={MARGIN_MODE}")
            return True
        else:
            raise Exception(f"设置杠杆失败: {response.get('msg', '未知错误')}")
    except Exception as e:
        logging.error(f"设置杠杆失败: {SYMBOL}, {str(e)}")
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
            df = pd.DataFrame(klines['data'],
                              columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
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

        logging.info(f"指标计算完成 - RSI: {rsi_series.iloc[-1]:.2f} | RSI_MA: {rsi_ma_series.iloc[-1]:.2f}"
                     f"{' [实时价格]' if current_price is not None else ' [收盘价]'}")
        return rsi_series.iloc[-1], rsi_ma_series.iloc[-1]
    except Exception as e:
        logging.error(f"RSI及均线计算错误: {str(e)}")
        return None, None


def get_symbol_info(symbol):
    try:
        info = public_client.get_instruments(instType='SWAP', instId=symbol)
        if info.get('code') != '0':
            raise Exception(f"获取交易对信息失败: {info.get('msg', '未知错误')}")
        ct_val = float(info['data'][0]['ctVal'])
        min_qty = float(info['data'][0]['minSz'])
        tick_sz = float(info['data'][0]['tickSz'])
        lot_sz = float(info['data'][0].get('lotSz', min_qty))
        return ct_val, min_qty, tick_sz, lot_sz
    except Exception as e:
        logging.error(f"获取交易对信息失败: {symbol}, {str(e)}")
        return 0.01, 0.01, 0.01, 0.01


def place_order(side):
    """side='buy'→开多(按BUY_QTY, 可选止盈止损), side='sell'→平多(按SELL_QTY, 不超过持仓)"""
    try:
        ct_val, min_qty, tick_sz, lot_sz = get_symbol_info(SYMBOL)

        if side == 'buy':
            # BTC 数量 → 合约张数
            quantity_in_contracts = BUY_QTY / ct_val
            quantity_in_contracts = max(round(quantity_in_contracts / lot_sz) * lot_sz, min_qty)
            if quantity_in_contracts < min_qty:
                logging.warning(f"下单失败: 张数 {quantity_in_contracts:.2f} < 最小 {min_qty}")
                return None

            order_params = {
                'instId': SYMBOL, 'tdMode': MARGIN_MODE,
                'side': 'buy', 'posSide': 'long',
                'ordType': 'market', 'sz': str(round(quantity_in_contracts, 2)),
                'clOrdId': str(uuid.uuid4()).replace('-', '')[:32]
            }

            if ENABLE_TP_SL:
                current_price = state[SYMBOL]['current_price']
                tp_price = round(current_price * (1 + BUY_TAKE_PROFIT / 100), -int(np.log10(tick_sz)))
                sl_price = round(current_price * (1 - BUY_STOP_LOSS / 100), -int(np.log10(tick_sz)))
                algo_order = {
                    'tpTriggerPx': str(tp_price), 'tpOrdPx': '-1',
                    'slTriggerPx': str(sl_price), 'slOrdPx': '-1',
                    'tpOrdKind': 'condition', 'slTriggerPxType': 'last', 'tpTriggerPxType': 'last'
                }
                order_params['attachAlgoOrds'] = [algo_order]
        else:
            # 平多：按 SELL_QTY 平仓，不超过实际持仓
            _, long_qty, _ = get_position()
            if long_qty <= 0:
                logging.info("无多头仓位，跳过平仓")
                return None
            sell_contracts = min(SELL_QTY / ct_val, long_qty / ct_val)
            quantity_in_contracts = max(round(sell_contracts / lot_sz) * lot_sz, min_qty)
            if quantity_in_contracts < min_qty:
                logging.warning(f"下单失败: 张数 {quantity_in_contracts:.2f} < 最小 {min_qty}")
                return None
            order_params = {
                'instId': SYMBOL, 'tdMode': MARGIN_MODE,
                'side': 'sell', 'posSide': 'long',
                'ordType': 'market', 'sz': str(round(quantity_in_contracts, 2)),
                'clOrdId': str(uuid.uuid4()).replace('-', '')[:32]
            }

        order = trade_client.place_order(**order_params)
        if order['code'] == '0':
            action = '开多' if side == 'buy' else '平多'
            logging.info(f"{action}成功: {order['data'][0]['ordId']}")
            return order['data'][0]['ordId']
        else:
            logging.error(f"下单失败: {order['msg']}")
            return None
    except Exception as e:
        logging.error(f"下单异常: {str(e)}")
        return None


def get_position():
    """获取多头持仓（张数、均价）"""
    for attempt in range(3):
        try:
            positions = account_client.get_positions(instType='SWAP', instId=SYMBOL)
            if positions.get('code') != '0':
                raise Exception(f"获取持仓失败: {positions.get('msg')}")

            if positions.get('data'):
                for pos in positions['data']:
                    if pos['instId'] == SYMBOL and pos['posSide'] == 'long':
                        qty = float(pos['pos']) if pos['pos'] else 0.0
                        avg_price = float(pos['avgPx']) if pos['avgPx'] else 0.0
                        return qty, avg_price
            return 0.0, 0.0
        except Exception as e:
            logging.error(f"获取持仓失败 (尝试 {attempt + 1}/3): {str(e)}")
            time.sleep(2)
    return 0.0, 0.0


def get_balance():
    for attempt in range(3):
        try:
            balance = account_client.get_account_balance()
            if balance.get('code') != '0':
                raise Exception(f"获取余额失败: {balance.get('msg')}")

            usdt = 0.0
            if balance.get('data') and balance['data'][0].get('details'):
                usdt_asset = next(
                    (asset for asset in balance['data'][0]['details'] if asset['ccy'] == 'USDT'),
                    {'availEq': '0'}
                )
                usdt = float(usdt_asset['availEq']) if usdt_asset['availEq'] else 0.0
            return usdt
        except Exception as e:
            logging.error(f"获取余额失败 (尝试 {attempt + 1}/3): {str(e)}")
            time.sleep(2)
    return 0.0


def execute_trading_logic():
    try:
        df_klines = get_klines(SYMBOL, TIMEFRAME, limit=100)
        if df_klines is None:
            return False

        price = get_price(SYMBOL)
        if not price:
            return False

        latest_rsi, latest_rsi_ma = calculate_rsi_and_ma(
            df_klines, rsi_period=RSI_PERIOD, ma_period=RSI_MA_PERIOD, current_price=price
        )
        if latest_rsi is None or latest_rsi_ma is None:
            return False

        state[SYMBOL]['latest_rsi'] = latest_rsi
        state[SYMBOL]['latest_rsi_ma'] = latest_rsi_ma

        usdt_balance = get_balance()
        long_qty, long_avg_price = get_position()

        logging.info(
            f"账户状态 - USDT: {usdt_balance:.2f} | "
            f"多头: {long_qty:.2f}张@{long_avg_price:.2f} | "
            f"RSI: {latest_rsi:.2f} | RSI_MA: {latest_rsi_ma:.2f}"
        )

        # 策略核心逻辑：RSI 交叉 + 阈值缓冲
        buy_gap = latest_rsi_ma - latest_rsi    # RSI 低于均线的幅度（正数=偏低）
        sell_gap = latest_rsi - latest_rsi_ma   # RSI 高于均线的幅度（正数=偏高）

        if buy_gap > RSI_BUY_THRESHOLD:
            # RSI 低于均线超过阈值 → 开多
            required_usdt = BUY_QTY * price / LEVERAGE
            if usdt_balance >= required_usdt:
                logging.info(f"触发开多信号: RSI({latest_rsi:.2f}) < MA({latest_rsi_ma:.2f}), "
                             f"差值={buy_gap:.2f} > 阈值({RSI_BUY_THRESHOLD})")
                place_order('buy')
            else:
                logging.warning(f"余额不足，取消开多。需要保证金: {required_usdt:.2f} USDT, 当前: {usdt_balance:.2f}")

        elif sell_gap > RSI_SELL_THRESHOLD:
            # RSI 高于均线超过阈值 → 平多
            if long_qty > 0:
                logging.info(f"触发平多信号: RSI({latest_rsi:.2f}) > MA({latest_rsi_ma:.2f}), "
                             f"差值={sell_gap:.2f} > 阈值({RSI_SELL_THRESHOLD})")
                place_order('sell')
            else:
                logging.info("满足平多信号，但当前无多头仓位，跳过")

        else:
            logging.info(f"信号未触发 | RSI: {latest_rsi:.2f} | MA: {latest_rsi_ma:.2f} | "
                         f"买差: {buy_gap:.2f}(需>{RSI_BUY_THRESHOLD}) | 卖差: {sell_gap:.2f}(需>{RSI_SELL_THRESHOLD})")

        return True
    except Exception as e:
        logging.error(f"执行交易逻辑异常: {str(e)}\n{traceback.format_exc()}")
        return False


def main():
    try:
        config = load_config()
        api_key = config.get('api_key') or os.getenv('OKX_API_KEY')
        api_secret = config.get('api_secret') or os.getenv('OKX_API_SECRET')
        passphrase = config.get('passphrase') or os.getenv('OKX_PASSPHRASE')

        if not all([api_key, api_secret, passphrase]):
            raise ValueError("API密钥未配置，请检查配置文件或环境变量")

        if init_okx(api_key, api_secret, passphrase, flag='0'):
            set_leverage()
            execute_trading_logic()
            logging.info("程序执行完毕")
        else:
            logging.error("OKX API 初始化失败，程序退出")
    except Exception as e:
        logging.error(f"主程序错误: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()