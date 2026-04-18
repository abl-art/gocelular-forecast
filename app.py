import os
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import psycopg2

app = Flask(__name__)
CORS(app)

DB_URL = os.environ.get("GOCELULAR_DB_URL")


def get_sales_data():
    """Fetch daily sales from GOcelular DB, grouped by model."""
    conn = psycopg2.connect(DB_URL)
    try:
        query = """
            SELECT
                o.order_created_at::date AS ds,
                COALESCE(so.product_name, 'Desconocido') AS modelo,
                COUNT(*)::int AS y
            FROM gocuotas_orders o
            LEFT JOIN store_orders so ON so.id::text = o.store_order_id
            WHERE o.order_delivered_at IS NOT NULL
                AND o.order_discarded_at IS NULL
                AND o.client_id::text IN ('1', '2026134', '2461631', '5495277')
                AND o.order_created_at >= '2026-03-23'
                AND so.product_name IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1
        """
        df = pd.read_sql(query, conn)
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    finally:
        conn.close()


def get_total_sales():
    """Fetch total daily sales (all models combined)."""
    conn = psycopg2.connect(DB_URL)
    try:
        query = """
            SELECT
                o.order_created_at::date AS ds,
                COUNT(*)::int AS y
            FROM gocuotas_orders o
            JOIN store_orders so ON so.id::text = o.store_order_id
            WHERE o.order_delivered_at IS NOT NULL
                AND o.order_discarded_at IS NULL
                AND o.client_id::text IN ('1', '2026134', '2461631', '5495277')
                AND o.order_created_at >= '2026-03-23'
            GROUP BY 1
            ORDER BY 1
        """
        df = pd.read_sql(query, conn)
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    finally:
        conn.close()


def get_stock_by_model():
    """Fetch current stock from GOcelular DB."""
    conn = psycopg2.connect(DB_URL)
    try:
        query = """
            SELECT
                COALESCE(sp.display_name, dm.model_code, 'Desconocido') AS modelo,
                COUNT(*)::int AS stock
            FROM inventory_items ii
            LEFT JOIN device_models dm ON dm.model_code = ii.model_code
            LEFT JOIN store_products sp ON sp.model_code = ii.model_code
            WHERE ii.status = 'available'
            GROUP BY 1
            ORDER BY 2 DESC
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


def seasonal_forecast(df, days=90, events=None):
    """Simple seasonal forecast using weekly pattern + trend."""
    if len(df) < 7:
        return None

    # Fill missing dates
    date_range = pd.date_range(df["ds"].min(), df["ds"].max())
    df = df.set_index("ds").reindex(date_range, fill_value=0).reset_index()
    df.columns = ["ds", "y"]

    # Calculate weekly seasonality (avg by day of week)
    df["dow"] = df["ds"].dt.dayofweek
    weekly_pattern = df.groupby("dow")["y"].mean().to_dict()

    # Calculate trend using last 14 days vs previous 14 days
    recent = df.tail(14)["y"].mean()
    previous = df.tail(28).head(14)["y"].mean() if len(df) >= 28 else recent
    daily_trend = (recent - previous) / 14 if previous > 0 else 0

    # Generate forecast
    last_date = df["ds"].max()
    overall_mean = df["y"].mean()
    forecasts = []

    for i in range(1, days + 1):
        future_date = last_date + timedelta(days=i)
        dow = future_date.weekday()
        seasonal = weekly_pattern.get(dow, overall_mean)
        trend_adj = daily_trend * i
        yhat = max(0, seasonal + trend_adj)

        # Apply event multiplier if matching month
        if events:
            month_key = future_date.strftime("%Y-%m")
            if month_key in events:
                yhat *= events[month_key]

        # Confidence interval (wider as we go further)
        std = df["y"].std() * (1 + i * 0.02)
        forecasts.append({
            "ds": future_date.strftime("%Y-%m-%d"),
            "yhat": round(yhat, 1),
            "yhat_lower": round(max(0, yhat - 1.5 * std), 1),
            "yhat_upper": round(yhat + 1.5 * std, 1),
        })

    # Build result with actuals + forecast
    actuals = []
    for _, row in df.iterrows():
        actuals.append({
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "y": float(row["y"]),
            "yhat": None,
            "yhat_lower": None,
            "yhat_upper": None,
        })

    return actuals + forecasts


def forecast_model(df_model, days=30, events=None):
    """Forecast for a single model."""
    if len(df_model) < 3:
        return None

    # Fill missing dates
    date_range = pd.date_range(df_model["ds"].min(), df_model["ds"].max())
    df_filled = df_model.set_index("ds").reindex(date_range, fill_value=0).reset_index()
    df_filled.columns = ["ds", "y"]

    # Weekly pattern
    df_filled["dow"] = df_filled["ds"].dt.dayofweek
    weekly_pattern = df_filled.groupby("dow")["y"].mean().to_dict()

    # Trend
    recent = df_filled.tail(7)["y"].mean()

    last_date = df_filled["ds"].max()
    total = 0
    daily = []

    for i in range(1, days + 1):
        future_date = last_date + timedelta(days=i)
        dow = future_date.weekday()
        yhat = max(0, weekly_pattern.get(dow, recent))

        # Apply event multiplier if matching month
        if events:
            month_key = future_date.strftime("%Y-%m")
            if month_key in events:
                yhat *= events[month_key]

        daily.append({
            "ds": future_date.strftime("%Y-%m-%d"),
            "yhat": round(yhat, 1),
        })

        total += yhat

    return {
        "forecast": round(total),
        "daily": daily,
    }


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/debug/db")
def debug_db():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM gocuotas_orders LIMIT 1")
        count = cur.fetchone()[0]
        conn.close()
        return jsonify({"db": "ok", "orders_count": count})
    except Exception as e:
        return jsonify({"db": "error", "message": str(e)}), 500


@app.route("/forecast/total", methods=["GET", "POST"])
def api_forecast_total():
    try:
        events = None
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            days = body.get("days", 90)
            events = body.get("events")
        else:
            days = request.args.get("days", 90, type=int)
        df = get_total_sales()
        result = seasonal_forecast(df, days, events=events)
        if result is None:
            return jsonify({"error": "Not enough data"}), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast/models", methods=["GET", "POST"])
def api_forecast_models():
    try:
        events = None
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            days = body.get("days", 30)
            events = body.get("events")
        else:
            days = request.args.get("days", 30, type=int)
        df = get_sales_data()
        if len(df) < 3:
            return jsonify({})

        model_counts = df.groupby("modelo")["y"].sum().sort_values(ascending=False)
        top_models = model_counts.head(15).index.tolist()

        results = {}
        for modelo in top_models:
            model_df = df[df["modelo"] == modelo][["ds", "y"]].copy()
            fc = forecast_model(model_df, days, events=events)
            if fc:
                results[modelo] = fc

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast/compras", methods=["GET", "POST"])
def api_forecast_compras():
    try:
        events = None
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            days = body.get("days", 15)
            events = body.get("events")
        else:
            days = request.args.get("days", 15, type=int)
        df = get_sales_data()

        model_counts = df.groupby("modelo")["y"].sum().sort_values(ascending=False)
        top_models = model_counts.head(15).index.tolist()

        model_forecasts = {}
        for modelo in top_models:
            model_df = df[df["modelo"] == modelo][["ds", "y"]].copy()
            fc = forecast_model(model_df, days, events=events)
            if fc:
                model_forecasts[modelo] = fc

        try:
            stock_df = get_stock_by_model()
            stock_map = dict(zip(stock_df["modelo"], stock_df["stock"]))
        except Exception:
            stock_map = {}

        recommendations = []
        for modelo, data in model_forecasts.items():
            forecast_total = data["forecast"]
            current_stock = stock_map.get(modelo, 0)
            deficit = max(0, forecast_total - current_stock)

            recommendations.append({
                "modelo": modelo,
                "forecast": forecast_total,
                "stock_actual": current_stock,
                "comprar": deficit,
            })

        recommendations.sort(key=lambda x: x["comprar"], reverse=True)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
