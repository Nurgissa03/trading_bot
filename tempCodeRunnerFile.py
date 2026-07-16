import os
import json
import time
import math
import asyncio
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd
import ta
from dotenv import load_dotenv
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
# ENV / CONFIG
# =========================
TOKEN = "8691843872:AAEMhCuon4Y4ZW7DSL-4az2dHH8JhKcOsec"
NEWS_API_KEY = "79e6a3398b3743a389f99a0e321bfa97"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FILE = os.path.join(BASE_DIR, "signals_history.json")
LOG_FILE = os.path.join(BASE_DIR, "bot.log")

AUTO_CHECK_SECONDS = 60          # как часто проверять авто-сигналы
MODEL_RETRAIN_SECONDS = 900      # переобучать модель раз в 15 минут
INTERVAL = "1m"
LIMIT = 400
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

MIN_CONFIDENCE = 70.0
MIN_ACCURACY = 55.0              # реальнее для time-series split
MIN_TREND_DIFF_PCT = 0.03        # ema20/ema50 разница минимум 0.03%
COOLDOWN_CANDLES = 1             # не слать подряд одинаковый свежий сигнал
REQUEST_TIMEOUT = 15

auto_signal_running = False
auto_task = None

# Кэш модели: {symbol: {"model": ..., "accuracy": ..., "trained_at": ..., "last_train_candle": ...}}
model_cache = {}

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
    threshold_ms = candles * 60 * 1000
    for s in reversed(signals):
        if s["symbol"] == symbol and abs(entry_time - s["entry_time"]) < threshold_ms:
            return True
    return False

# =========================
# HELPERS
# =========================
def safe_request(url, params=None):
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning("HTTP ошибка: %s | url=%s", e, url)
        return None


def interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    raise ValueError(f"Неподдерживаемый interval: {interval}")


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================
# MARKET DATA
# =========================
def get_data(symbol="BTCUSDT"):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT
    }

    data = safe_request(url, params=params)
    if not data or not isinstance(data, list):
        raise ValueError(f"Не удалось получить market data для {symbol}")

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

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()
    df["macd_diff"] = ta.trend.MACD(df["close"]).macd_diff()

    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_high"] - df["bb_low"]

    atr_indicator = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr_indicator.average_true_range()

    df["price_change"] = df["close"].pct_change()
    df["candle_range"] = df["high"] - df["low"]
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]

    # тренд-фичи
    df["ema20_50_diff_pct"] = ((df["ema20"] - df["ema50"]) / df["close"]) * 100
    df["ema50_200_diff_pct"] = ((df["ema50"] - df["ema200"]) / df["close"]) * 100

    return df


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
    "price_change",
    "candle_range",
    "volume_ratio",
    "ema20_50_diff_pct",
    "ema50_200_diff_pct",
]

# =========================
# MODEL
# =========================
def train_model_time_series(df):
    if len(df) < 220:
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
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test) * 100
    return model, accuracy


def get_or_train_model(symbol, df):
    global model_cache

    last_candle_time = int(df["time"].iloc[-1])
    current_ts = time.time()

    cached = model_cache.get(symbol)
    should_retrain = True

    if cached:
        trained_recently = (current_ts - cached["trained_at"]) < MODEL_RETRAIN_SECONDS
        same_candle = cached["last_train_candle"] == last_candle_time
        if trained_recently and same_candle:
            should_retrain = False

    if should_retrain:
        logger.info("Переобучение модели для %s", symbol)
        model, accuracy = train_model_time_series(df)
        model_cache[symbol] = {
            "model": model,
            "accuracy": accuracy,
            "trained_at": current_ts,
            "last_train_candle": last_candle_time,
        }

    return model_cache[symbol]["model"], model_cache[symbol]["accuracy"]


def predict_last(model, df):
    last = df[FEATURES].iloc[-1:]
    pred = model.predict(last)[0]
    prob = model.predict_proba(last)[0]

    confidence = max(prob) * 100
    direction = "UP" if pred == 1 else "DOWN"
    return direction, confidence


def get_next_time(df):
    last_time = int(df["time"].iloc[-1])
    dt = datetime.fromtimestamp(last_time / 1000)
    minutes = interval_to_minutes(INTERVAL)
    return (dt + timedelta(minutes=minutes)).strftime("%H:%M:%S")


# =========================
# NEWS
# =========================
def get_news(symbol):
    if not NEWS_API_KEY:
        return []

    mapping = {
        "BTCUSDT": "bitcoin OR BTC",
        "ETHUSDT": "ethereum OR ETH",
        "SOLUSDT": "solana OR SOL"
    }
    query = mapping.get(symbol, "crypto")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY,
    }

    res = safe_request(url, params=params)
    if not res or res.get("status") != "ok":
        return []

    return res.get("articles", [])


def analyze_news(symbol):
    articles = get_news(symbol)

    positive = ["surge", "rally", "bullish", "rise", "growth", "adoption", "gain", "approval", "record"]
    negative = ["crash", "drop", "hack", "ban", "fall", "loss", "bearish", "lawsuit", "liquidation"]

    score = 0
    headlines = []

    for article in articles:
        title = (article.get("title") or "").lower()
        original_title = article.get("title", "No title")
        headlines.append(original_title)

        for word in positive:
            if word in title:
                score += 1
        for word in negative:
            if word in title:
                score -= 1

    if score > 1:
        sentiment = "POSITIVE"
    elif score < -1:
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
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) else 0

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = best_ask - best_bid

    if imbalance > 0.10:
        signal = "BUY PRESSURE"
    elif imbalance < -0.10:
        signal = "SELL PRESSURE"
    else:
        signal = "NEUTRAL"

    return signal, imbalance, bid_volume, ask_volume, spread


def get_market_momentum(df):
    last_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]

    volume_spike = bool(last_volume > avg_volume * 1.4) if not pd.isna(avg_volume) else False
    price_change = float(df["close"].pct_change().iloc[-1])
    strong_move = abs(price_change) > 0.001  # 0.1% для 1m
    direction = "UP" if price_change > 0 else "DOWN"

    return volume_spike, strong_move, direction, price_change


def get_trend_strength(df):
    last = df.iloc[-1]
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    close = float(last["close"])

    diff_20_50_pct = abs((ema20 - ema50) / close) * 100
    diff_50_200_pct = abs((ema50 - ema200) / close) * 100

    if ema20 > ema50 > ema200:
        trend = "BULLISH"
    elif ema20 < ema50 < ema200:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    is_strong_trend = diff_20_50_pct >= MIN_TREND_DIFF_PCT
    return trend, is_strong_trend, diff_20_50_pct, diff_50_200_pct


# =========================
# FINAL LOGIC
# =========================
def combine(prediction, news, confidence, accuracy, order_book_signal,
            volume_spike, strong_move, move_direction, trend, strong_trend):
    if (
        prediction == "UP"
        and trend == "BULLISH"
        and strong_trend
        and order_book_signal == "BUY PRESSURE"
        and move_direction == "UP"
        and confidence >= MIN_CONFIDENCE
        and accuracy >= MIN_ACCURACY
        and (volume_spike or strong_move)
        and news in ["POSITIVE", "NEUTRAL"]
    ):
        return "🔥 STRONG BUY"

    if (
        prediction == "DOWN"
        and trend == "BEARISH"
        and strong_trend
        and order_book_signal == "SELL PRESSURE"
        and move_direction == "DOWN"
        and confidence >= MIN_CONFIDENCE
        and accuracy >= MIN_ACCURACY
        and (volume_spike or strong_move)
        and news in ["NEGATIVE", "NEUTRAL"]
    ):
        return "🔥 STRONG SELL"

    if strong_move or volume_spike:
        return "⚠️ VOLATILE"

    return "⚠️ MIXED"


def should_send_signal(confidence, accuracy, overall, trend, strong_trend):
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
    return True


# =========================
# SIGNAL RESULT CHECK
# =========================
async def wait_for_closed_next_candle(symbol, entry_time_ms, timeout_seconds=180):
    """
    Ждём появления следующей закрытой свечи после entry_time_ms.
    """
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            df = get_data(symbol)
            future = df[df["time"] > entry_time_ms]
            if not future.empty:
                next_row = future.iloc[0]
                next_close_time_ms = int(next_row["close_time"])

                # если свеча уже точно закрыта
                if int(time.time() * 1000) >= next_close_time_ms:
                    return float(next_row["close"])

        except Exception as e:
            logger.warning("Ошибка wait_for_closed_next_candle %s: %s", symbol, e)

        await asyncio.sleep(3)

    return None


async def check_signal_result(app, chat_id, signal_id, symbol, prediction, entry_close, entry_time):
    try:
        next_close = await wait_for_closed_next_candle(symbol, entry_time, timeout_seconds=180)

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
            f"Entry close: {round(entry_close, 4)}\n"
            f"Next close: {round(next_close, 4)}\n\n"
            f"{emoji} {result}"
        )
        await app.bot.send_message(chat_id=chat_id, text=text)

    except Exception as e:
        logger.exception("Ошибка проверки результата сигнала: %s", e)
        await app.bot.send_message(chat_id=chat_id, text=f"❌ Error checking signal: {e}")


# =========================
# SIGNAL GENERATION CORE
# =========================
def build_signal(symbol):
    df = get_data(symbol)
    df = add_indicators(df)
    df = create_target(df)

    model, accuracy = get_or_train_model(symbol, df)
    prediction, confidence = predict_last(model, df)
    next_time = get_next_time(df)

    news_sentiment, score, headlines = analyze_news(symbol)
    order_book_signal, imbalance, bid_volume, ask_volume, spread = get_order_book_signal(symbol)
    volume_spike, strong_move, move_direction, price_change = get_market_momentum(df)
    trend, strong_trend, diff_20_50_pct, diff_50_200_pct = get_trend_strength(df)

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
        strong_trend
    )

    entry_close = float(df["close"].iloc[-1])
    entry_time = int(df["time"].iloc[-1])

    signal_id = f"{symbol}_{entry_time}_{prediction}"

    return {
        "df": df,
        "symbol": symbol,
        "prediction": prediction,
        "confidence": float(confidence),
        "accuracy": float(accuracy),
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
        "overall": overall,
        "entry_close": entry_close,
        "entry_time": entry_time,
        "signal_id": signal_id,
    }


def format_signal_text(data, auto_mode=False):
    emoji = "📈" if data["prediction"] == "UP" else "📉"
    prefix = "🚀 AUTO SIGNAL" if auto_mode else f"📊 {data['symbol']}"
    news_lines = "\n".join([f"• {h}" for h in data["headlines"][:3]]) if data["headlines"] else "• No recent headlines"

    if auto_mode:
        head = f"{prefix}\n\n📊 {data['symbol']}\n"
    else:
        head = f"{prefix}\n\n"

    text = (
        f"{head}"
        f"⏰ Next: {data['next_time']}\n"
        f"{emoji} Tech: {data['prediction']}\n"
        f"🎯 Confidence: {round(data['confidence'], 2)}%\n"
        f"🧠 Accuracy: {round(data['accuracy'], 2)}%\n\n"
        f"📰 News: {data['news_sentiment']} (score: {data['news_score']})\n"
        f"📚 Order book: {data['order_book_signal']}\n"
        f"⚖️ Imbalance: {round(data['imbalance'], 4)}\n"
        f"💰 Bid vol: {round(data['bid_volume'], 2)} | Ask vol: {round(data['ask_volume'], 2)}\n"
        f"↔️ Spread: {round(data['spread'], 4)}\n"
        f"📊 Volume spike: {data['volume_spike']}\n"
        f"⚡ Strong move: {data['strong_move']}\n"
        f"📈 Move: {data['move_direction']} ({round(data['price_change'] * 100, 2)}%)\n"
        f"📐 Trend: {data['trend']}\n"
        f"📏 EMA20/50 diff: {round(data['diff_20_50_pct'], 4)}%\n"
        f"📏 EMA50/200 diff: {round(data['diff_50_200_pct'], 4)}%\n\n"
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
        "🤖 COMMANDS\n\n"
        "/start — start bot\n"
        "/help — commands list\n"
        "/auto — start auto signals\n"
        "/stop — stop auto mode\n"
        "/stats — signal statistics\n"
        "/history — last signals\n\n"
        "Buttons:\n"
        "BTC / ETH / SOL — manual signal"
    )
    await update.message.reply_text(text)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["BTC", "ETH", "SOL"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    await update.message.reply_text(
        "🤖 Advanced Trading Bot\n\n"
        "Выбери монету.\n\n"
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
                        data["strong_trend"]
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
        "🚀 Авто включен.\n"
        "Проверяю BTC / ETH / SOL каждую минуту.\n"
        "Дубликаты не отправляются.\n"
        "Результаты и статистика сохраняются."
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