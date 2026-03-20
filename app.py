from flask import Flask, render_template, request
from predictor import analyze_stock

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form.get("ticker", "TCS.NS").upper()
        try:
            prediction_days = int(request.form.get("prediction_days", 7))
        except ValueError:
            prediction_days = 7
            
        try:
            lookback_days = int(request.form.get("lookback_days", 365))
        except ValueError:
            lookback_days = 365
        
        result = analyze_stock(ticker, prediction_days, lookback_days)
        if "error" in result:
            return render_template("index.html", error=result["error"], ticker=ticker, prediction_days=prediction_days, lookback_days=lookback_days)
            
        return render_template("index.html", result=result, ticker=ticker, prediction_days=prediction_days, lookback_days=lookback_days)

    return render_template("index.html", ticker="TCS.NS", prediction_days=7, lookback_days=365)

if __name__ == "__main__":
    app.run(debug=True)
