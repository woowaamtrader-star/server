from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)

# กำหนดค่า Client สำหรับ Gemini API (ใส่ API Key ของคุณตรงนี้)
client = genai.Client(api_key="AQ.Ab8RN6IYUF152LVuJTs8ywaZknSlEMe_rvQ_sVk4Daeqnlq2ZA")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symbol = data.get('symbol')
    ask = data.get('ask')
    bid = data.get('bid')
    rsi = data.get('rsi')
    ma = data.get('ma')
    atr = data.get('atr')

    # สร้าง Prompt วิเคราะห์ตลาดทองคำ
    prompt = f"""
    You are an expert XAUUSD M15 trader and machine learning model.
    Current market data: Symbol={symbol}, Ask={ask}, Bid={bid}, RSI={rsi}, MA={ma}, ATR={atr}.
    Based on trend-following and volatility management, should we BUY, SELL, or HOLD?
    Return ONLY a JSON format with keys: 'signal' (1 for Buy, -1 for Sell, 0 for Hold) and 'confidence' (0.0 to 1.0).
    """

    try:
        # อัปเดตใช้รุ่น gemini-3.6-flash ตามคำแนะนำของ API
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )

        # ตัวอย่างผลลัพธ์จำลองโครงสร้าง JSON กลับไปให้ MT5
        result = {"signal": 1, "confidence": 0.85}
        return jsonify(result)

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({"signal": 0, "confidence": 0.0})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
