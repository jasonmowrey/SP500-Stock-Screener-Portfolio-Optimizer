from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, plotting
import webbrowser
from threading import Timer
import cvxpy as cp
import warnings

# Suppress specific FutureWarning from CVXPY
warnings.filterwarnings("ignore", category=FutureWarning, module='cvxpy.reductions.solvers.solving_chain')

# Initialize app
app = Flask(__name__)
Bootstrap(app)

file_path = 'final_stock_data.csv'
stock_data = pd.read_csv(file_path)
unique_tickers = stock_data['ticker'].unique()

# Define the app
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_tickers = request.form.getlist('tickers')
        
        if not selected_tickers:
            return render_template('index.html', tickers=unique_tickers, error="Please select at least one ticker.")

        end_date = datetime.today()
        start_date = end_date - timedelta(days=730)
        
        stock_prices = yf.download(selected_tickers, start=start_date, end=end_date)['Adj Close']
        if stock_prices.empty:
            return render_template('index.html', tickers=unique_tickers, error="Failed to fetch data for selected tickers.")

        mu = expected_returns.mean_historical_return(stock_prices)
        S = risk_models.sample_cov(stock_prices)

        # Optimize for maximum Sharpe ratio
        ef = EfficientFrontier(mu, S, solver=cp.CLARABEL)
        sharpe_weights = ef.max_sharpe()
        sharpe_perf = ef.portfolio_performance(verbose=True)

        # Optimize for minimum volatility
        ef = EfficientFrontier(mu, S, solver=cp.CLARABEL)

        min_vol_weights = ef.min_volatility()
        min_vol_perf = ef.portfolio_performance(verbose=True)

        # Optimize for maximum return
        ef = EfficientFrontier(mu, S, solver=cp.CLARABEL)
        target_return = mu.max() * 0.99  # Use 99% of the maximum return to ensure it is feasible
        max_ret_weights = ef.efficient_return(target_return=target_return)
        max_ret_perf = ef.portfolio_performance(verbose=True)

        # Generate plots for the portfolio weights
        plot_urls = {}
        for label, weights, perf in zip(["Sharpe", "Min Vol", "Max Ret"],
                                        [sharpe_weights, min_vol_weights, max_ret_weights],
                                        [sharpe_perf, min_vol_perf, max_ret_perf]):
            fig, ax = plt.subplots()
            tickers = list(weights.keys())
            values = list(weights.values())
            ax.bar(tickers, values, color='blue')
            ax.set_title(f'Portfolio Weights for {label} Portfolio')
            ax.set_ylabel('Weight')
            ax.set_xlabel('Ticker')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close(fig)  # Close the figure to free resources
            img.seek(0)
            plot_urls[label] = base64.b64encode(img.getvalue()).decode('utf8')

        # Generate the efficient frontier plot again (as previous step)
        fig, ax = plt.subplots()
        ef_for_plotting = EfficientFrontier(mu, S, solver=cp.CLARABEL)
        plotting.plot_efficient_frontier(ef_for_plotting, ax=ax, show_assets=False)
        ax.scatter(*sharpe_perf[:2], color='black', marker='*', s=100, label='Max Sharpe Ratio')
        ax.scatter(*min_vol_perf[:2], color='blue', marker='o', s=100, label='Min Volatility')
        ax.scatter(*max_ret_perf[:2], color='red', marker='^', s=100, label='Max Return')
        ax.set_title('Efficient Frontier with Highlighted Portfolios')
        ax.legend()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close(fig)
        img.seek(0)
        plot_url_ef = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('results.html', plot_url_ef=plot_url_ef, plot_urls=plot_urls,
                               performances=[sharpe_perf, min_vol_perf, max_ret_perf])

    return render_template('index.html', tickers=unique_tickers)

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()  # Wait for 1 second before opening the web browser
    app.run(debug=False)
