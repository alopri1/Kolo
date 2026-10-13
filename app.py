from flask import Flask, jsonify, request, send_from_directory
import technical

app = Flask(__name__, static_folder="static")

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

@app.get("/api/analyze")
def analyze():
    sym = request.args.get("symbol", "EURUSD=X").strip()
    itv = request.args.get("interval", "1h")
    if itv not in ("5m", "15m", "1h", "4h", "1d"):
        itv = "1h"
    try:
        return jsonify(technical.run(sym, itv))
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
