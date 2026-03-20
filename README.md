# 📈 BazaarVani

BazaarVani is an **AI-Powered Market Intelligence Dashboard** that predicts future stock trajectories using Machine Learning. Built with Python (Flask), scikit-learn, and Plotly, this web application fetches real-time market data, performs sentiment analysis on recent financial news, and utilizes AI optimization to project upcoming stock prices.

## ✨ Features
- **AI Stock Projections**: Analyzes historical data to automatically select the best-performing regression model (Linear Regression, Random Forest, Gradient Boosting, etc.) to project future stock trajectories.
- **Interactive Visualization**: Stunning, responsive Plotly charts.
- **Sentiment Analysis**: Integrates with Finnhub, NewsAPI, and VADER to evaluate market sentiment automatically.
- **Modern UI**: A completely custom, lightweight styling and dashboard layout meant for lightning-fast lookups.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Priyanshu-Yadav19/BazaarVani.git
cd BazaarVani
```

### 2. Set up the Environment
It is highly recommended to use a Python virtual environment:
```bash
python -m venv myven
myven\Scripts\activate      # On Windows
source myven/bin/activate    # On macOS/Linux
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. API Keys Configuration
BazaarVani relies on external APIs (Polygon, Finnhub, NewsAPI, and Alpha Vantage) to fetch live financial details seamlessly. 
1. Create a file exactly named `.env` in the root folder.
2. Copy the contents of `.env.example` over to your new `.env` file.
3. Replace the placeholder text with your actual API keys:

```env
POLYGON_API_KEY=your_polygon_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
NEWSAPI_KEY=your_newsapi_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

### 4. Run the Application
Start the Flask development server:
```bash
python app.py
```
Open your web browser and navigate to `http://127.0.0.1:5000` to begin your analysis!


