import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =========================
# TOKEN / API
# =========================
TOKEN = "8691843872:AAEMhCuon4Y4ZW7DSL-4az2dHH8JhKcOsec"
NEWS_API_KEY = "79e6a3398b3743a389f99a0e321bfa97"

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FILE = os.path.join(BASE_DIR, "signals_history.json")
LOG_FILE = os.path.join(BASE_DIR, "bot.log")

PRIMARY_INTERVAL = "5m"      # основной таймфрейм для сигналов
TREND_INTERVAL = "15m"       # подтверждение тренда
LIMIT = 1000

AUTO_CHECK_SECONDS = 60
MODEL_RETRAIN_SECONDS = 900
REQUEST_TIMEOUT = 15

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

MIN_CONFIDENCE = 68.0
MIN_ACCURACY = 58.0
MIN_RR = 1.2
MIN_TREND_DIFF_PCT = 0.03
COOLDOWN_CANDLES = 1

auto_signal_running = False
auto_task = None
model_cache = {}
news_cache = {}

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# FILE STORAGE
# =========================
def load_signals():
    if not os.path.exists(SIGNALS_FILE):
        return []
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Ошибка чтения signals_history.json: %s", e)
        return []


def save_signals(signals):
    try:
        with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump(signals, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Ошибка записи signals_history.json: %s", e)


def add_signal_record(record):
    signals = load_signals()
    signals.append(record)
    save_signals(signals)


def update_signal_result(signal_id, result, next_close):
    signals = load_signals()
    updated = False

    for signal in signals:
        if signal["signal_id"] == signal_id:
            signal["result"] = result
            signal["next_close"] = next_close
            updated = True
            break

    if updated:
        save_signals(signals)


def signal_exists(signal_id):
    signals = load_signals()
    return any(signal["signal_id"] == signal_id for signal in signals)


def recent_same_symbol_signal_exists(symbol, entry_time, candles=COOLDOWN_CANDLES):
    signals = load_signals()
    threshold_ms = candles * interval_to_minutes(PRIMARY_INTERVAL) * 60 * 1000
    for s in reversed(signals):
        if s["symbol"] == symbol and abs(entry_time - s["entry_time"]) < threshold_ms:
            return True
    return False


# =========================
# HELPERS
# =========================
def safe_request(url, params=None, retries=3, delay=2):
    """Делает HTTP-запрос с несколькими попытками в случае неудачи."""
    for i in range(retries):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Вызовет исключение для кодов 4xx/5xx
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(
                "HTTP ошибка (попытка %d/%d): %s | url=%s", i + 1, retries, e, url
            )
            if i < retries - 1:
                time.sleep(delay)
    
    logger.error("Не удалось выполнить HTTP-запрос после %d попыток.", retries)
    return None


def interval_to_minutes(interval: str) -> int:
        return None


def interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    raise ValueError(f"Неподдерживаемый interval: {interval}")


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def round_price(symbol: str, price: float) -> float:
    if "BTC" in symbol:
        return round(price, 2)
    if "ETH" in symbol:
        return round(price, 2)
    return round(price, 4)


# =========================
# MARKET DATA
# =========================
def get_data(symbol="BTCUSDT", interval="5m", limit=LIMIT):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    data = safe_request(url, params=params)
    if not data or not isinstance(data, list):
        # Возвращаем пустой DataFrame вместо ошибки, чтобы избежать падения
        logger.warning("Получены пустые данные для %s %s", symbol, interval)
        return pd.DataFrame()
        raise ValueError(f"Не удалось получить market data для {symbol} {interval}")

    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    df["time"] = df["time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    return df


def add_indicators(df):
    df = df.copy()

    macd_obj = ta.trend.MACD(df["close"])
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"] = macd_obj.macd_diff()

    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_high"] - df["bb_low"]

    atr_indicator = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr_indicator.average_true_range()

    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    adx_obj = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx_obj.adx()

    df["price_change"] = df["close"].pct_change()
    df["candle_range"] = df["high"] - df["low"]
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]

    df["ema20_50_diff_pct"] = ((df["ema20"] - df["ema50"]) / df["close"]) * 100
    df["ema50_200_diff_pct"] = ((df["ema50"] - df["ema200"]) / df["close"]) * 100

    return df

# =========================
# CANDLE PATTERNS
# =========================
def is_bullish(candle):
    return float(candle["close"]) > float(candle["open"])


def is_bearish(candle):
    return float(candle["close"]) < float(candle["open"])


def candle_body(candle):
    return abs(float(candle["close"]) - float(candle["open"]))


def candle_range(candle):
    return float(candle["high"]) - float(candle["low"])


def upper_wick(candle):
    return float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))


def lower_wick(candle):
    return min(float(candle["open"]), float(candle["close"])) - float(candle["low"])


def is_small_body(candle, max_body_ratio=0.35):
    rng = candle_range(candle)
    if rng <= 0:
        return False
    return candle_body(candle) / rng <= max_body_ratio


def is_big_body(candle, min_body_ratio=0.55):
    rng = candle_range(candle)
    if rng <= 0:
        return False
    return candle_body(candle) / rng >= min_body_ratio


def detect_hammer(c1, trend="bearish"):
    rng = candle_range(c1)
    body = candle_body(c1)
    uw = upper_wick(c1)
    lw = lower_wick(c1)

    if rng <= 0:
        return False

    return (
        trend == "bearish"
        and body <= rng * 0.5
        and lw >= body * 2
        and uw <= body * 0.5
    )


def detect_inverted_hammer(c1, trend="bearish"):
    rng = candle_range(c1)
    body = candle_body(c1)
    uw = upper_wick(c1)
    lw = lower_wick(c1)

    if rng <= 0:
        return False

    return (
        trend == "bearish"
        and body <= rng * 0.5
        and uw >= body * 2
        and lw <= body * 0.5
    )


def detect_shooting_star(c1, trend="bullish"):
    rng = candle_range(c1)
    body = candle_body(c1)
    uw = upper_wick(c1)
    lw = lower_wick(c1)

    if rng <= 0:
        return False

    return (
        trend == "bullish"
        and body <= rng * 0.5
        and uw >= body * 2
        and lw <= body * 0.5
    )


def detect_hanging_man(c1, trend="bullish"):
    rng = candle_range(c1)
    body = candle_body(c1)
    uw = upper_wick(c1)
    lw = lower_wick(c1)

    if rng <= 0:
        return False

    return (
        trend == "bullish"
        and body <= rng * 0.5
        and lw >= body * 2
        and uw <= body * 0.5
    )


def detect_bullish_engulfing(c1, c2, trend="bearish"):
    return (
        trend == "bearish"
        and is_bearish(c1)
        and is_bullish(c2)
        and float(c2["open"]) <= float(c1["close"])
        and float(c2["close"]) >= float(c1["open"])
    )


def detect_bearish_engulfing(c1, c2, trend="bullish"):
    return (
        trend == "bullish"
        and is_bullish(c1)
        and is_bearish(c2)
        and float(c2["open"]) >= float(c1["close"])
        and float(c2["close"]) <= float(c1["open"])
    )


def detect_bullish_harami(c1, c2, trend="bearish"):
    return (
        trend == "bearish"
        and is_bearish(c1)
        and is_bullish(c2)
        and candle_body(c2) <= candle_body(c1) * 0.45
        and float(c2["open"]) >= min(float(c1["open"]), float(c1["close"]))
        and float(c2["close"]) <= max(float(c1["open"]), float(c1["close"]))
    )


def detect_bearish_harami(c1, c2, trend="bullish"):
    return (
        trend == "bullish"
        and is_bullish(c1)
        and is_bearish(c2)
        and candle_body(c2) <= candle_body(c1) * 0.45
        and float(c2["open"]) <= max(float(c1["open"]), float(c1["close"]))
        and float(c2["close"]) >= min(float(c1["open"]), float(c1["close"]))
    )


def detect_morning_star(c1, c2, c3, trend="bearish"):
    first_mid = (float(c1["open"]) + float(c1["close"])) / 2
    return (
        trend == "bearish"
        and is_bearish(c1)
        and is_big_body(c1)
        and is_small_body(c2, 0.4)
        and is_bullish(c3)
        and is_big_body(c3, 0.45)
        and float(c3["close"]) >= first_mid
    )


def detect_evening_star(c1, c2, c3, trend="bullish"):
    first_mid = (float(c1["open"]) + float(c1["close"])) / 2
    return (
        trend == "bullish"
        and is_bullish(c1)
        and is_big_body(c1)
        and is_small_body(c2, 0.4)
        and is_bearish(c3)
        and is_big_body(c3, 0.45)
        and float(c3["close"]) <= first_mid
    )


def detect_three_white_soldiers(c1, c2, c3, trend="bearish"):
    return (
        trend == "bearish"
        and is_bullish(c1) and is_bullish(c2) and is_bullish(c3)
        and candle_body(c1) > 0 and candle_body(c2) > 0 and candle_body(c3) > 0
        and float(c2["close"]) > float(c1["close"])
        and float(c3["close"]) > float(c2["close"])
    )


def detect_three_black_crows(c1, c2, c3, trend="bullish"):
    return (
        trend == "bullish"
        and is_bearish(c1) and is_bearish(c2) and is_bearish(c3)
        and candle_body(c1) > 0 and candle_body(c2) > 0 and candle_body(c3) > 0
        and float(c2["close"]) < float(c1["close"])
        and float(c3["close"]) < float(c2["close"])
    )


def detect_pin_bar(c1):
    body = candle_body(c1)
    uw = upper_wick(c1)
    lw = lower_wick(c1)
    rng = candle_range(c1)

    if rng <= 0:
        return None

    # bullish pin bar
    if lw >= body * 2.5 and uw <= body:
        return "BULLISH_PIN_BAR"

    # bearish pin bar
    if uw >= body * 2.5 and lw <= body:
        return "BEARISH_PIN_BAR"

    return None


def detect_inside_bar(c1, c2):
    return (
        float(c2["high"]) <= float(c1["high"])
        and float(c2["low"]) >= float(c1["low"])
    )


def get_simple_trend(df, lookback=5):
    if len(df) < lookback + 2:
        return "SIDEWAYS"

    closes = df["close"].tail(lookback).tolist()

    if closes[-1] > closes[0]:
        return "BULLISH"
    elif closes[-1] < closes[0]:
        return "BEARISH"
    return "SIDEWAYS"


def detect_candle_patterns(df):
    """
    Возвращает:
    {
        "bullish": [список бычьих паттернов],
        "bearish": [список медвежьих паттернов],
        "neutral": [список нейтральных паттернов],
        "score": число
    }
    """
    result = {
        "bullish": [],
        "bearish": [],
        "neutral": [],
        "score": 0
    }

    if len(df) < 5:
        return result

    trend = get_simple_trend(df, lookback=5)
    trend_lower = "bullish" if trend == "BULLISH" else "bearish" if trend == "BEARISH" else "sideways"

    c1 = df.iloc[-1]
    c2 = df.iloc[-2]
    c3 = df.iloc[-3]
    c4 = df.iloc[-4]

    # 1-candle
    if detect_hammer(c1, trend_lower):
        result["bullish"].append("HAMMER")
        result["score"] += 2

    if detect_inverted_hammer(c1, trend_lower):
        result["bullish"].append("INVERTED_HAMMER")
        result["score"] += 1

    if detect_shooting_star(c1, trend_lower):
        result["bearish"].append("SHOOTING_STAR")
        result["score"] -= 2

    if detect_hanging_man(c1, trend_lower):
        result["bearish"].append("HANGING_MAN")
        result["score"] -= 1

    pin = detect_pin_bar(c1)
    if pin == "BULLISH_PIN_BAR":
        result["bullish"].append(pin)
        result["score"] += 1
    elif pin == "BEARISH_PIN_BAR":
        result["bearish"].append(pin)
        result["score"] -= 1

    # 2-candle
    if detect_bullish_engulfing(c2, c1, trend_lower):
        result["bullish"].append("BULLISH_ENGULFING")
        result["score"] += 2

    if detect_bearish_engulfing(c2, c1, trend_lower):
        result["bearish"].append("BEARISH_ENGULFING")
        result["score"] -= 2

    if detect_bullish_harami(c2, c1, trend_lower):
        result["bullish"].append("BULLISH_HARAMI")
        result["score"] += 1

    if detect_bearish_harami(c2, c1, trend_lower):
        result["bearish"].append("BEARISH_HARAMI")
        result["score"] -= 1

    if detect_inside_bar(c2, c1):
        result["neutral"].append("INSIDE_BAR")

    # 3-candle
    if detect_morning_star(c3, c2, c1, trend_lower):
        result["bullish"].append("MORNING_STAR")
        result["score"] += 3

    if detect_evening_star(c3, c2, c1, trend_lower):
        result["bearish"].append("EVENING_STAR")
        result["score"] -= 3

    if detect_three_white_soldiers(c3, c2, c1, trend_lower):
        result["bullish"].append("THREE_WHITE_SOLDIERS")
        result["score"] += 3

    if detect_three_black_crows(c3, c2, c1, trend_lower):
        result["bearish"].append("THREE_BLACK_CROWS")
        result["score"] -= 3

    return result

def create_target(df):
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df


FEATURES = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_diff",
    "volume",
    "ema20",
    "ema50",
    "ema200",
    "bb_high",
    "bb_low",
    "bb_width",
    "atr",
    "adx",
    "stoch_k",
    "stoch_d",
    "price_change",
    "candle_range",
    "body_size",
    "upper_wick",
    "lower_wick",
    "volume_ratio",
    "ema20_50_diff_pct",
    "ema50_200_diff_pct",
]

# =========================
# MODEL
# =========================
def train_model_time_series(df):
    if len(df) < 150:
        raise ValueError("Недостаточно данных для обучения модели")

    X = df[FEATURES]
    y = df["target"]

    split_index = int(len(df) * 0.8)
    if split_index <= 50 or split_index >= len(df):
        raise ValueError("Некорректный split для обучения")

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test) * 100
    return model, accuracy


def get_or_train_model(symbol, df):
    global model_cache

    key = f"{symbol}_{PRIMARY_INTERVAL}"
    if df.empty:
        logger.warning("get_or_train_model: получен пустой DataFrame для %s", key)
        raise ValueError(f"Недостаточно данных для анализа {symbol}")

    last_candle_time = int(df["time"].iloc[-1])
    current_ts = time.time()

    cached = model_cache.get(key)
    should_retrain = True

    if cached:
        trained_recently = (current_ts - cached["trained_at"]) < MODEL_RETRAIN_SECONDS
        same_candle = cached["last_train_candle"] == last_candle_time
        if trained_recently and same_candle:
            should_retrain = False

    if should_retrain:
        logger.info("Переобучение модели для %s", key)
        model, accuracy = train_model_time_series(df)
        model_cache[key] = {
            "model": model,
            "accuracy": accuracy,
            "trained_at": current_ts,
            "last_train_candle": last_candle_time,
        }

    return model_cache[key]["model"], model_cache[key]["accuracy"]


def predict_last(model, df):
    last = df[FEATURES].iloc[-1:]
    pred = model.predict(last)[0]
    prob = model.predict_proba(last)[0]

    confidence = max(prob) * 100
    direction = "UP" if pred == 1 else "DOWN"
    return direction, confidence


# =========================
# NEWS
# =========================
def get_news(symbol):
    """
    Получает новости для символа, используя кэш, чтобы избежать
    превышения лимитов API.
    """
    if not NEWS_API_KEY:
        return []

    mapping = {
        "BTCUSDT": "bitcoin OR BTC",
        "ETHUSDT": "ethereum OR ETH",
        "SOLUSDT": "solana OR SOL"
    }
    query = mapping.get(symbol, "crypto")

    # Проверка кэша
    cached_news = news_cache.get(symbol)
    if cached_news and (time.time() - cached_news["timestamp"]) < 600: # 10 минут
        return cached_news["articles"]

    logger.info("Запрос свежих новостей для %s", symbol)

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY,
    }

    res = safe_request(url, params=params)
    if not res or res.get("status") != "ok":
        return []
    
    articles = res.get("articles", [])
    # Сохраняем в кэш
    news_cache[symbol] = {"articles": articles, "timestamp": time.time()}
    
    return res.get("articles", [])


def analyze_news(symbol):
    articles = get_news(symbol)

    weighted_positive = {
        "surge": 1, "rally": 1, "bullish": 1, "rise": 1, "growth": 1,
        "adoption": 1, "gain": 1, "approval": 2, "record": 1,
        "breakout": 2, "inflow": 2, "institutional": 2, "upgrade": 1,
        "partnership": 1, "launch": 1, "etf": 2
    }
    weighted_negative = {
        "crash": 2, "drop": 1, "hack": 2, "ban": 2, "fall": 1,
        "loss": 1, "bearish": 1, "lawsuit": 2, "liquidation": 2,
        "dump": 2, "selloff": 2, "fear": 1, "panic": 1,
        "exploit": 2, "outflow": 2, "fraud": 2
    }

    score = 0
    headlines = []

    for article in articles:
        title = (article.get("title") or "").lower()
        original_title = article.get("title", "No title")
        headlines.append(original_title)

        for word, weight in weighted_positive.items():
            if word in title:
                score += weight

        for word, weight in weighted_negative.items():
            if word in title:
                score -= weight

    if score >= 3:
        sentiment = "POSITIVE"
    elif score <= -3:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    return sentiment, score, headlines


# =========================
# ORDER BOOK + MOMENTUM
# =========================
def get_order_book_signal(symbol="BTCUSDT", limit=100):
    url = "https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}

    data = safe_request(url, params=params)
    if not data:
        return "NEUTRAL", 0, 0, 0, 0

    bids = data.get("bids", [])
    asks = data.get("asks", [])

    if not bids or not asks:
        return "NEUTRAL", 0, 0, 0, 0

    bid_volume = sum(float(bid[1]) for bid in bids)
    ask_volume = sum(float(ask[1]) for ask in asks)
    total = bid_volume + ask_volume
    imbalance = (bid_volume - ask_volume) / total if total else 0

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = best_ask - best_bid

    if imbalance > 0.12:
        signal = "BUY PRESSURE"
    elif imbalance < -0.12:
        signal = "SELL PRESSURE"
    else:
        signal = "NEUTRAL"

    return signal, imbalance, bid_volume, ask_volume, spread


def get_market_momentum(df):
    last_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]

    volume_spike = bool(last_volume > avg_volume * 1.35) if not pd.isna(avg_volume) else False
    price_change = float(df["close"].pct_change().iloc[-1])
    strong_move = abs(price_change) > 0.0015
    direction = "UP" if price_change > 0 else "DOWN"

    return volume_spike, strong_move, direction, price_change


def get_trend_strength(df):
    last = df.iloc[-1]
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    close = float(last["close"])
    adx = float(last["adx"])

    diff_20_50_pct = abs((ema20 - ema50) / close) * 100
    diff_50_200_pct = abs((ema50 - ema200) / close) * 100

    if ema20 > ema50 > ema200:
        trend = "BULLISH"
    elif ema20 < ema50 < ema200:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    is_strong_trend = diff_20_50_pct >= MIN_TREND_DIFF_PCT and adx >= 18
    return trend, is_strong_trend, diff_20_50_pct, diff_50_200_pct, adx


def get_higher_timeframe_confirmation(symbol):
    df = get_data(symbol, TREND_INTERVAL)
    df = add_indicators(df)
    df.dropna(inplace=True)

    if df.empty:
        logger.warning("Не удалось получить данные для HTF-подтверждения по %s", symbol)
        return "UNKNOWN", False, 0

    if len(df) < 50:
        return "UNKNOWN", False, 0

    trend, strong_trend, _, _, adx = get_trend_strength(df)
    return trend, strong_trend, adx


def get_support_resistance(df, lookback=20):
    recent = df.tail(lookback)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    return support, resistance


def calculate_trade_levels(symbol, prediction, entry, atr, support, resistance):
    if prediction == "UP":
        sl = min(entry - atr * 1.2, support - atr * 0.2)
        tp = max(entry + atr * 2.0, resistance)
        risk = entry - sl
        reward = tp - entry
    else:
        sl = max(entry + atr * 1.2, resistance + atr * 0.2)
        tp = min(entry - atr * 2.0, support)
        risk = sl - entry
        reward = entry - tp

    rr = reward / risk if risk > 0 else 0

    return round_price(symbol, sl), round_price(symbol, tp), round(rr, 2)


# =========================
# FINAL LOGIC
# =========================
def combine(prediction, news, confidence, accuracy, order_book_signal,
            volume_spike, strong_move, move_direction,
            trend, strong_trend, htf_trend, htf_strong, rr, patterns_score):
    if (
        prediction == "UP"
        and trend == "BULLISH"
        and htf_trend == "BULLISH"
        and strong_trend
        and htf_strong
        and order_book_signal == "BUY PRESSURE"
        and move_direction == "UP"
        and confidence >= MIN_CONFIDENCE
        and accuracy >= MIN_ACCURACY
        and (volume_spike or strong_move)
        and news in ["POSITIVE", "NEUTRAL"]
        and rr >= MIN_RR
        and patterns_score >= 1
    ):
        return "🔥 STRONG BUY"

    if (
        prediction == "DOWN"
        and trend == "BEARISH"
        and htf_trend == "BEARISH"
        and strong_trend
        and htf_strong
        and order_book_signal == "SELL PRESSURE"
        and move_direction == "DOWN"
        and confidence >= MIN_CONFIDENCE
        and accuracy >= MIN_ACCURACY
        and (volume_spike or strong_move)
        and news in ["NEGATIVE", "NEUTRAL"]
        and rr >= MIN_RR
        and patterns_score <= -1
    ):
        return "🔥 STRONG SELL"

    if strong_move or volume_spike:
        return "⚠️ VOLATILE"

    return "⚠️ MIXED"


def should_send_signal(confidence, accuracy, overall, trend, strong_trend, rr):
    if confidence < MIN_CONFIDENCE:
        return False
    if accuracy < MIN_ACCURACY:
        return False
    if overall not in ["🔥 STRONG BUY", "🔥 STRONG SELL"]:
        return False
    if trend == "SIDEWAYS":
        return False
    if not strong_trend:
        return False
    if rr < MIN_RR:
        return False
    return True


# =========================
# RESULT CHECK
# =========================
async def wait_for_closed_next_candle(symbol, entry_time_ms, timeout_seconds=900):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            df = get_data(symbol, PRIMARY_INTERVAL)
            future = df[df["time"] > entry_time_ms]
            if not future.empty:
                next_row = future.iloc[0]
                next_close_time_ms = int(next_row["close_time"])

                if int(time.time() * 1000) >= next_close_time_ms:
                    return float(next_row["close"])
        except Exception as e:
            logger.warning("Ошибка wait_for_closed_next_candle %s: %s", symbol, e)

        await asyncio.sleep(5)

    return None


async def check_signal_result(app, chat_id, signal_id, symbol, prediction, entry_close, entry_time):
    try:
        next_close = await wait_for_closed_next_candle(symbol, entry_time, timeout_seconds=900)

        if next_close is None:
            await app.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Не удалось проверить результат для {symbol}: следующая свеча не закрылась вовремя."
            )
            return

        if prediction == "UP":
            result = "WIN" if next_close > entry_close else "LOSS"
        else:
            result = "WIN" if next_close < entry_close else "LOSS"

        update_signal_result(signal_id, result, next_close)

        emoji = "✅" if result == "WIN" else "❌"
        text = (
            f"📍 Signal result\n\n"
            f"📊 {symbol}\n"
            f"Prediction: {prediction}\n"
            f"Entry close: {round_price(symbol, entry_close)}\n"
            f"Next close: {round_price(symbol, next_close)}\n\n"
            f"{emoji} {result}"
        )
        await app.bot.send_message(chat_id=chat_id, text=text)

    except Exception as e:
        logger.exception("Ошибка проверки результата сигнала: %s", e)
        await app.bot.send_message(chat_id=chat_id, text=f"❌ Error checking signal: {e}")


# =========================
# SIGNAL CORE
# =========================
def build_signal(symbol):
    df = get_data(symbol, PRIMARY_INTERVAL)
    df = add_indicators(df)
    df = create_target(df)
    
    if df.empty or len(df) < 50:
        logger.warning("Недостаточно данных для build_signal по %s", symbol)
        raise ValueError(f"Недостаточно данных для анализа {symbol}")

    patterns = detect_candle_patterns(df)

    model, accuracy = get_or_train_model(symbol, df)
    prediction, confidence = predict_last(model, df)

    next_time = (
        datetime.fromtimestamp(int(df["time"].iloc[-1]) / 1000)
        + timedelta(minutes=interval_to_minutes(PRIMARY_INTERVAL))
    ).strftime("%H:%M:%S")

    news_sentiment, score, headlines = analyze_news(symbol)
    order_book_signal, imbalance, bid_volume, ask_volume, spread = get_order_book_signal(symbol)
    volume_spike, strong_move, move_direction, price_change = get_market_momentum(df)

    trend, strong_trend, diff_20_50_pct, diff_50_200_pct, adx = get_trend_strength(df)
    htf_trend, htf_strong, htf_adx = get_higher_timeframe_confirmation(symbol)

    last = df.iloc[-1]
    entry_close = float(last["close"])
    entry_time = int(last["time"])
    atr = float(last["atr"])

    support, resistance = get_support_resistance(df, lookback=20)
    stop_loss, take_profit, rr = calculate_trade_levels(
        symbol=symbol,
        prediction=prediction,
        entry=entry_close,
        atr=atr,
        support=support,
        resistance=resistance
    )

    overall = combine(
        prediction,
        news_sentiment,
        confidence,
        accuracy,
        order_book_signal,
        volume_spike,
        strong_move,
        move_direction,
        trend,
        strong_trend,
        htf_trend,
        htf_strong,
        rr,
        patterns["score"]
    )

    signal_id = f"{symbol}_{entry_time}_{prediction}"

    return {
        "symbol": symbol,
        "prediction": prediction,
        "confidence": float(confidence),
        "accuracy": float(accuracy),
        "patterns_bullish": patterns["bullish"],
        "patterns_bearish": patterns["bearish"],
        "patterns_neutral": patterns["neutral"],
        "patterns_score": patterns["score"],
        "next_time": next_time,
        "news_sentiment": news_sentiment,
        "news_score": score,
        "headlines": headlines,
        "order_book_signal": order_book_signal,
        "imbalance": float(imbalance),
        "bid_volume": float(bid_volume),
        "ask_volume": float(ask_volume),
        "spread": float(spread),
        "volume_spike": volume_spike,
        "strong_move": strong_move,
        "move_direction": move_direction,
        "price_change": float(price_change),
        "trend": trend,
        "strong_trend": strong_trend,
        "diff_20_50_pct": float(diff_20_50_pct),
        "diff_50_200_pct": float(diff_50_200_pct),
        "adx": float(adx),
        "htf_trend": htf_trend,
        "htf_strong": htf_strong,
        "htf_adx": float(htf_adx) if isinstance(htf_adx, (float, int)) else 0.0,
        "support": round_price(symbol, support),
        "resistance": round_price(symbol, resistance),
        "entry_close": entry_close,
        "entry_time": entry_time,
        "atr": round_price(symbol, atr),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rr": rr,
        "overall": overall,
        "signal_id": signal_id,
    }


def format_signal_text(data, auto_mode=False):
    emoji = "📈" if data["prediction"] == "UP" else "📉"
    header = "🚀 AUTO SIGNAL" if auto_mode else f"📊 {data['symbol']} PRO SIGNAL"
    news_lines = "\n".join([f"• {h}" for h in data["headlines"][:3]]) if data["headlines"] else "• No recent headlines"
    bullish_patterns = ", ".join(data["patterns_bullish"]) if data["patterns_bullish"] else "None"
    bearish_patterns = ", ".join(data["patterns_bearish"]) if data["patterns_bearish"] else "None"
    neutral_patterns = ", ".join(data["patterns_neutral"]) if data["patterns_neutral"] else "None"
    text = (
        f"{header}\n\n"
        f"📊 {data['symbol']}\n"
        f"🕒 TF: {PRIMARY_INTERVAL} | HTF: {TREND_INTERVAL}\n"
        f"⏰ Next candle: {data['next_time']}\n"
        f"{emoji} Prediction: {data['prediction']}\n"
        f"🎯 Confidence: {round(data['confidence'], 2)}%\n"
        f"🧠 Accuracy: {round(data['accuracy'], 2)}%\n\n"
        f"📍 Entry: {round_price(data['symbol'], data['entry_close'])}\n"
        f"🛑 Stop Loss: {data['stop_loss']}\n"
        f"🎯 Take Profit: {data['take_profit']}\n"
        f"⚖️ Risk/Reward: {data['rr']}\n\n"
        f"📐 Trend {PRIMARY_INTERVAL}: {data['trend']} | ADX: {round(data['adx'], 2)}\n"
        f"📈 Trend {TREND_INTERVAL}: {data['htf_trend']} | ADX: {round(data['htf_adx'], 2)}\n"
        f"📏 EMA20/50 diff: {round(data['diff_20_50_pct'], 4)}%\n"
        f"📏 EMA50/200 diff: {round(data['diff_50_200_pct'], 4)}%\n\n"
        f"📰 News: {data['news_sentiment']} (score: {data['news_score']})\n"
        f"📚 Order book: {data['order_book_signal']}\n"
        f"⚖️ Imbalance: {round(data['imbalance'], 4)}\n"
        f"💰 Bid vol: {round(data['bid_volume'], 2)} | Ask vol: {round(data['ask_volume'], 2)}\n"
        f"↔️ Spread: {round(data['spread'], 4)}\n"
        f"📊 Volume spike: {data['volume_spike']}\n"
        f"⚡ Strong move: {data['strong_move']}\n"
        f"📉 Price move: {data['move_direction']} ({round(data['price_change'] * 100, 2)}%)\n"
        f"🧱 Support: {data['support']} | Resistance: {data['resistance']}\n\n"
        f"🕯 Bullish patterns: {bullish_patterns}\n"
        f"🕯 Bearish patterns: {bearish_patterns}\n"
        f"🕯 Neutral patterns: {neutral_patterns}\n"
        f"🧮 Pattern score: {data['patterns_score']}\n\n"
        f"📌 {data['overall']}"
    )

    if not auto_mode:
        text += f"\n\nLatest headlines:\n{news_lines}"

    return text


async def send_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str):
    try:
        data = build_signal(symbol)

        if not signal_exists(data["signal_id"]):
            add_signal_record({
                "signal_id": data["signal_id"],
                "symbol": data["symbol"],
                "prediction": data["prediction"],
                "confidence": round(data["confidence"], 2),
                "accuracy": round(data["accuracy"], 2),
                "entry_close": data["entry_close"],
                "entry_time": data["entry_time"],
                "result": None,
                "next_close": None,
                "overall": data["overall"],
                "created_at": now_str()
            })

            asyncio.create_task(
                check_signal_result(
                    context.application,
                    update.effective_chat.id,
                    data["signal_id"],
                    data["symbol"],
                    data["prediction"],
                    data["entry_close"],
                    data["entry_time"]
                )
            )

        await update.message.reply_text(format_signal_text(data, auto_mode=False))

    except Exception as e:
        logger.exception("send_signal error: %s", e)
        await update.message.reply_text(f"❌ Ошибка: {e}")


# =========================
# COMMANDS
# =========================
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    signals = load_signals()

    if not signals:
        await update.message.reply_text("📊 Статистика пока пустая.")
        return

    finished = [s for s in signals if s["result"] in ["WIN", "LOSS"]]

    total = len(finished)
    wins = sum(1 for s in finished if s["result"] == "WIN")
    losses = sum(1 for s in finished if s["result"] == "LOSS")
    win_rate = (wins / total * 100) if total > 0 else 0

    parts = []
    for sym in SYMBOLS:
        sym_signals = [s for s in finished if s["symbol"] == sym]
        sym_total = len(sym_signals)
        sym_wins = sum(1 for s in sym_signals if s["result"] == "WIN")
        sym_losses = sum(1 for s in sym_signals if s["result"] == "LOSS")
        sym_rate = (sym_wins / sym_total * 100) if sym_total > 0 else 0

        parts.append(
            f"{sym}\n"
            f"Signals: {sym_total}\n"
            f"Wins: {sym_wins}\n"
            f"Losses: {sym_losses}\n"
            f"Win rate: {round(sym_rate, 2)}%"
        )

    text = (
        f"📊 BOT STATS\n\n"
        f"Total finished signals: {total}\n"
        f"✅ Wins: {wins}\n"
        f"❌ Losses: {losses}\n"
        f"🎯 Win rate: {round(win_rate, 2)}%\n\n"
        + "\n\n".join(parts)
    )

    await update.message.reply_text(text)


async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    signals = load_signals()

    if not signals:
        await update.message.reply_text("📜 История пока пустая.")
        return

    last_signals = signals[-10:]
    last_signals.reverse()

    lines = []
    for signal in last_signals:
        result = signal["result"] if signal["result"] else "PENDING"
        lines.append(
            f"{signal['symbol']}\n"
            f"Prediction: {signal['prediction']}\n"
            f"Confidence: {signal['confidence']}%\n"
            f"Accuracy: {signal['accuracy']}%\n"
            f"Overall: {signal.get('overall', 'N/A')}\n"
            f"Result: {result}\n"
            f"Time: {signal['created_at']}"
        )

    text = "📜 LAST SIGNALS\n\n" + "\n\n".join(lines)
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 PRO COMMANDS\n\n"
        "/start — start bot\n"
        "/help — commands list\n"
        "/auto — start auto signals\n"
        "/stop — stop auto mode\n"
        "/stats — signal statistics\n"
        "/history — last signals\n\n"
        "Buttons:\n"
        "BTC / ETH / SOL — manual pro signal"
    )
    await update.message.reply_text(text)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["BTC", "ETH", "SOL"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    await update.message.reply_text(
        "🤖 PRO Trend Signal Bot\n\n"
        "Таймфрейм сигнала: 5m\n"
        "Подтверждение тренда: 15m\n\n"
        "Команды:\n"
        "/auto — авто-сигналы\n"
        "/stop — остановить авто-сигналы\n"
        "/stats — статистика\n"
        "/history — последние сигналы\n"
        "/help — помощь",
        reply_markup=reply_markup
    )


async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        txt = update.message.text.strip().upper()

        if "BTC" in txt:
            await send_signal(update, context, "BTCUSDT")
        elif "ETH" in txt:
            await send_signal(update, context, "ETHUSDT")
        elif "SOL" in txt:
            await send_signal(update, context, "SOLUSDT")
        else:
            await update.message.reply_text("Выбери кнопку: BTC / ETH / SOL")

    except Exception as e:
        logger.exception("handle error: %s", e)
        await update.message.reply_text(f"❌ Ошибка: {e}")


# =========================
# AUTO MODE
# =========================
async def auto_loop(app, chat_id):
    global auto_signal_running

    while auto_signal_running:
        try:
            for symbol in SYMBOLS:
                try:
                    data = build_signal(symbol)

                    if recent_same_symbol_signal_exists(symbol, data["entry_time"], candles=COOLDOWN_CANDLES):
                        continue

                    if signal_exists(data["signal_id"]):
                        continue

                    if not should_send_signal(
                        data["confidence"],
                        data["accuracy"],
                        data["overall"],
                        data["trend"],
                        data["strong_trend"],
                        data["rr"]
                    ):
                        continue

                    add_signal_record({
                        "signal_id": data["signal_id"],
                        "symbol": data["symbol"],
                        "prediction": data["prediction"],
                        "confidence": round(data["confidence"], 2),
                        "accuracy": round(data["accuracy"], 2),
                        "entry_close": data["entry_close"],
                        "entry_time": data["entry_time"],
                        "result": None,
                        "next_close": None,
                        "overall": data["overall"],
                        "created_at": now_str()
                    })

                    asyncio.create_task(
                        check_signal_result(
                            app,
                            chat_id,
                            data["signal_id"],
                            data["symbol"],
                            data["prediction"],
                            data["entry_close"],
                            data["entry_time"]
                        )
                    )

                    await app.bot.send_message(chat_id=chat_id, text=format_signal_text(data, auto_mode=True))

                except Exception as inner_e:
                    logger.exception("Ошибка по символу %s: %s", symbol, inner_e)
                    await app.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка по {symbol}: {inner_e}")

        except Exception as e:
            logger.exception("AUTO ERROR: %s", e)
            await app.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка авто-режима: {e}")

        await asyncio.sleep(AUTO_CHECK_SECONDS)


async def auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_signal_running, auto_task

    if auto_signal_running:
        await update.message.reply_text("⚠️ Уже включено")
        return

    auto_signal_running = True
    chat_id = update.effective_chat.id

    await update.message.reply_text(
        "🚀 Проф авто-режим включен.\n"
        "Сканирую BTC / ETH / SOL.\n"
        "TF: 5m, подтверждение: 15m.\n"
        "Отправляю только сильные сигналы."
    )

    auto_task = asyncio.create_task(auto_loop(context.application, chat_id))


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_signal_running, auto_task

    auto_signal_running = False

    if auto_task:
        auto_task.cancel()
        auto_task = None

    await update.message.reply_text("🛑 Остановлено")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Update %s caused error: %s", update, context.error)


# =========================
# MAIN
# =========================
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("auto", auto))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.add_error_handler(error_handler)

    logger.info("Bot running...")
    print("Bot running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()