from flask import Flask, request, jsonify
import google.generativeai as genai
import os

app = Flask(__name__)

# --- 1. ตั้งค่า API Key สำหรับ Gemini ---
# นำ API Key ของคุณมาใส่แทนที่ข้อความด้านล่างนี้
API_KEY = "AQ.Ab8RN6IYUF152LVuJTs8ywaZknSlEMe_rvQ_sVk4Daeqnlq2ZA"
genai.configure(api_key=API_KEY)

# --- 2. เลือกโมเดล AI ที่ใช้งาน ---
# แนะนำใช้ gemini-1.5-flash หรือรุ่นที่เสถียรสำหรับงานวิเคราะห์
generation_config = {
    "temperature": 0.2, # ลดความเพ้อ ให้วิเคราะห์ตามตัวเลขตลาดจริง
    "response_mime_type": "application/json", # สั่งให้ตอบกลับเป็น JSON เสมอ
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับข้อมูลที่ส่งมาจาก MT5 EA
        data = request.json
        if not data:
            return jsonify({"signal": 0, "confidence": 0.0})

        symbol = data.get('symbol', 'XAUUSD')
        ask = data.get('ask', 0.0)
        bid = data.get('bid', 0.0)
        rsi = data.get('rsi', 0.0)
        ma = data.get('ma', 0.0)
        atr = data.get('atr', 0.0)

        # สร้าง Prompt สำหรับสั่งให้ AI วิเคราะห์
        prompt = f"""
        You are an expert {symbol} M15 trader and machine learning model.
        Current market data: Symbol={symbol}, Ask={ask}, Bid={bid}, RSI={rsi}, MA={ma}, ATR={atr}.
        Based on trend-following and volatility management, should we BUY, SELL, or HOLD?
        Return ONLY a JSON format with keys: 'signal' (1 for Buy, -1 for Sell, 0 for Hold) and 'confidence' (float between 0.0 and 1.0).
        """

        # ส่งข้อมูลให้ Gemini ประมวลผล
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # แปลงผลลัพธ์และส่งกลับไปยัง MT5
        import json
        result_json = json.loads(response_text)
        
        signal = int(result_json.get("signal", 0))
        confidence = float(result_json.get("confidence", 0.0))

        return jsonify({"signal": signal, "confidence": confidence})

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # กรณีเกิดข้อผิดพลาด ส่งค่า 0 (Hold) กลับไปก่อนเพื่อความปลอดภัย
        return jsonify({"signal": 0, "confidence": 0.0})

if __name__ == '__main__':
    # ดึง Port จาก Render อัตโนมัติ หรือใช้ 5000 สำหรับรันเทสในเครื่อง
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
