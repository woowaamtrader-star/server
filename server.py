from flask import Flask, request, jsonify
from google import genai
from google.genai import types
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- 1. ตั้งค่า API Key สำหรับ Gemini ---
# ห้าม hardcode key ในไฟล์ ให้ตั้งผ่าน environment variable แทน
# เช่น บน Render: Settings -> Environment -> เพิ่ม GEMINI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY ไม่ถูกตั้งค่า กรุณาตั้ง environment variable ก่อนรัน "
        "(อย่า hardcode key ในโค้ด โดยเฉพาะถ้าจะ push ขึ้น git repo)"
    )

# ใช้ google-genai (SDK ใหม่) แทน google-generativeai ตัวเก่าที่หมดอายุการซัพพอร์ตไปแล้ว
# (end-of-life 31 ส.ค. 2025) - SDK เก่าไม่รองรับโมเดลรุ่นใหม่ๆ อย่างถูกต้อง ทำให้เจอ 404 ค้าง
client = genai.Client(api_key=API_KEY)

# --- 2. เลือกโมเดล AI ที่ใช้งาน ---
# ตั้งชื่อโมเดลผ่าน env var เพื่อให้สลับรุ่นได้โดยไม่ต้องแก้โค้ด/deploy ใหม่
# เวลา Google ประกาศ deprecate รุ่นถัดไป (ดูรายชื่อรุ่นล่าสุดได้ที่ ai.google.dev/gemini-api/docs/models)
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.6-flash")

# --- 2b. Groq (fallback) - ใช้เฉพาะตอน Gemini โควต้าเต็ม (429) เท่านั้น ---
# ถ้าไม่ตั้ง GROQ_API_KEY ไว้ ระบบจะไม่ fallback แค่คืนค่า Hold เหมือนเดิมตอน Gemini ชนโควต้า
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
# หมายเหตุ: เดิมใช้ openai/gpt-oss-120b แต่ free tier มี TPD (token/วัน) แค่ 200,000
# หารด้วย ~2,000 token/request ของเรา เหลือโควต้าจริงแค่ ~100 ครั้ง/วันเท่านั้น
# llama-3.1-8b-instant มี TPD 500,000 และ RPD 14,400 - เหมาะกับ fallback ที่ต้องรับ volume สูง
# กว่ามาก แลกกับความฉลาดที่ลดลง แต่พอใช้งานได้ในฐานะตัวสำรอง (ไม่ใช่ตัวหลัก)
groq_client = None
if GROQ_API_KEY:
    from openai import OpenAI  # Groq ใช้ API รูปแบบเดียวกับ OpenAI SDK
    # max_retries=0: ปิด auto-retry ของ SDK เพราะเวลาเจอ 429 มันจะรอตาม Retry-After
    # (เจอจริงคือ 57 วินาที) ซึ่งนานกว่า timeout ของ EA (20 วินาที) มาก
    # ถ้าปล่อย default retry ไว้ EA จะ timeout ค้างก่อน server จะตอบกลับด้วยซ้ำ
    groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY, max_retries=0)
    logger.info("Groq fallback enabled (model=%s)", GROQ_MODEL_NAME)
else:
    logger.info("GROQ_API_KEY not set - Groq fallback disabled, will return Hold when Gemini quota is exhausted")


def describe_rsi_zone(rsi: float) -> str:
    if rsi >= 70:
        return "Overbought zone (>=70)"
    if rsi <= 30:
        return "Oversold zone (<=30)"
    if rsi >= 55:
        return "Upper-neutral, leaning bullish momentum"
    if rsi <= 45:
        return "Lower-neutral, leaning bearish momentum"
    return "Neutral zone, no clear momentum bias"


def describe_trend(price: float, ma_fast: float, ma_slow: float) -> str:
    if ma_fast <= 0 or ma_slow <= 0:
        return "Unknown (insufficient MA data)"
    if ma_fast > ma_slow and price > ma_fast:
        return "Uptrend (price above both MAs, fast MA above slow MA)"
    if ma_fast < ma_slow and price < ma_fast:
        return "Downtrend (price below both MAs, fast MA below slow MA)"
    return "Choppy / no clear trend (price and MAs mixed)"


def describe_bb_position(price: float, bb_upper: float, bb_lower: float) -> str:
    if bb_upper <= 0 or bb_lower <= 0 or bb_upper <= bb_lower:
        return "Unknown (Bollinger Band data unavailable)"
    band_width = bb_upper - bb_lower
    if price >= bb_upper:
        return "At/above upper Bollinger Band - stretched to the upside"
    if price <= bb_lower:
        return "At/below lower Bollinger Band - stretched to the downside"
    position_pct = (price - bb_lower) / band_width * 100
    return f"Inside bands, at {position_pct:.0f}% of the band range (0%=lower, 100%=upper)"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"signal": 0, "confidence": 0.0, "reason": "no data received"})

        symbol = data.get("symbol", "XAUUSD")
        ask = float(data.get("ask", 0.0))
        bid = float(data.get("bid", 0.0))
        rsi = float(data.get("rsi", 0.0))
        ma = float(data.get("ma", 0.0))          # fast MA (backward-compatible field name)
        atr = float(data.get("atr", 0.0))

        # --- ฟิลด์เสริม (optional) เพื่อให้ AI มีบริบทมากขึ้น ---
        # ถ้า EA ฝั่ง MQL5 ยังไม่ได้ส่งค่าพวกนี้มา จะ default เป็น 0 และถูกอธิบายว่า "unknown" ในพรอมต์
        ma_slow = float(data.get("ma_slow", 0.0))
        bb_upper = float(data.get("bb_upper", 0.0))
        bb_lower = float(data.get("bb_lower", 0.0))
        atr_baseline = float(data.get("atr_baseline", 0.0))  # ATR เฉลี่ยปกติ เพื่อเทียบความผันผวนปัจจุบัน
        session = data.get("session", "unknown")             # เช่น "London", "NewYork", "Asia"
        spread_points = float(data.get("spread_points", 0.0))

        mid_price = (ask + bid) / 2 if ask and bid else 0.0
        trend_desc = describe_trend(mid_price, ma, ma_slow)
        rsi_desc = describe_rsi_zone(rsi)
        bb_desc = describe_bb_position(mid_price, bb_upper, bb_lower)

        volatility_desc = "unknown"
        if atr_baseline > 0:
            ratio = atr / atr_baseline
            if ratio >= 2.0:
                volatility_desc = f"Volatility spike: current ATR is {ratio:.1f}x the baseline - likely news-driven, exercise caution"
            elif ratio <= 0.6:
                volatility_desc = f"Volatility compressed: current ATR is only {ratio:.1f}x baseline - low movement, breakout risk both ways"
            else:
                volatility_desc = f"Volatility normal range ({ratio:.1f}x baseline)"

        prompt = f"""You are a disciplined, risk-aware technical analyst for {symbol} on a short-term (M15) timeframe.
Analyze the market snapshot below and decide whether the setup favors BUY, SELL, or HOLD.

MARKET SNAPSHOT
- Symbol: {symbol}
- Session: {session}
- Ask / Bid: {ask} / {bid} (spread: {spread_points} points)
- RSI({rsi}): {rsi_desc}
- Trend (fast MA {ma} vs slow MA {ma_slow}): {trend_desc}
- Bollinger Band position: {bb_desc}
- Volatility state (ATR {atr}): {volatility_desc}

DECISION RULES
1. Only favor BUY or SELL when at least two of the signals above (trend, RSI zone, BB position) agree in the same direction.
2. If volatility is a "spike" (news-driven), be more conservative - prefer HOLD unless the setup is very clear, since spreads and slippage are likely elevated.
3. If trend or Bollinger Band data is "unknown" due to missing inputs, treat that signal as neutral and rely on the remaining data - do not invent values.
4. Confidence should reflect genuine agreement between signals, not just the presence of a signal:
   - 0.0-0.3: weak/conflicting signals, should map to HOLD in practice
   - 0.3-0.6: some agreement, moderate conviction
   - 0.6-1.0: strong agreement across multiple signals
5. Never assume information you were not given (news events, order book, higher timeframe structure). Base the decision only on the data provided.

Respond with ONLY a JSON object (no markdown, no extra text) with exactly these keys:
- "signal": 1 for Buy, -1 for Sell, 0 for Hold
- "confidence": float between 0.0 and 1.0
- "reason": one short sentence (max ~20 words) explaining the key factor(s) behind the decision
"""

        response_text = None
        provider_used = "gemini"

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,  # ลดความเพ้อ ให้วิเคราะห์ตามตัวเลขตลาดจริง
                    response_mime_type="application/json",
                ),
            )
            response_text = response.text.strip()

        except Exception as gemini_err:
            is_quota_error = "429" in str(gemini_err) or "RESOURCE_EXHAUSTED" in str(gemini_err)

            if is_quota_error and groq_client is not None:
                logger.warning("Gemini quota exhausted (429) - falling back to Groq (%s)", GROQ_MODEL_NAME)
                provider_used = "groq_fallback"
                try:
                    groq_response = groq_client.chat.completions.create(
                        model=GROQ_MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )
                    response_text = groq_response.choices[0].message.content.strip()
                except Exception as groq_err:
                    # ทั้ง Gemini และ Groq โควต้าเต็มพร้อมกัน - คืน Hold ทันที ไม่ต้อง retry ต่อ
                    logger.warning("Groq fallback also failed: %s", groq_err)
                    return jsonify({
                        "signal": 0, "confidence": 0.0,
                        "reason": "Both Gemini and Groq unavailable (rate limited)",
                        "provider": "none",
                    })
            else:
                # ไม่ใช่ quota error หรือไม่ได้ตั้ง Groq ไว้ - โยน error ต่อให้ except ด้านล่างจัดการ
                raise

        result_json = json.loads(response_text)

        signal = int(result_json.get("signal", 0))
        confidence = float(result_json.get("confidence", 0.0))
        reason = str(result_json.get("reason", ""))

        # sanity-clamp ค่าที่ AI ส่งกลับมา กันเคส AI ตอบนอกช่วงที่กำหนด
        signal = max(-1, min(1, signal))
        confidence = max(0.0, min(1.0, confidence))

        logger.info(
            "provider=%s symbol=%s signal=%s confidence=%.2f reason=%s",
            provider_used, symbol, signal, confidence, reason,
        )

        return jsonify({
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "provider": provider_used,
        })

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini JSON response: %s", e)
        return jsonify({"signal": 0, "confidence": 0.0, "reason": "AI response parse error"})
    except Exception as e:
        is_quota_error = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
        if is_quota_error:
            logger.warning("Gemini quota exhausted and no Groq fallback configured (set GROQ_API_KEY to enable)")
            return jsonify({"signal": 0, "confidence": 0.0, "reason": "Gemini quota exhausted (429), no fallback configured"})
        logger.exception("Error calling AI provider")
        # กรณีเกิดข้อผิดพลาดอื่นๆ ส่งค่า 0 (Hold) กลับไปก่อนเพื่อความปลอดภัย
        return jsonify({"signal": 0, "confidence": 0.0, "reason": f"server error: {e}"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
