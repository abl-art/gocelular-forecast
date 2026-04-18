import os
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from prophet import Prophet
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
                COALESCE(dm.model_code, ii.model_code, 'Desconocido') AS modelo,
                COUNT(*)::int AS stock
            FROM inventory_items ii
            LEFT JOIN device_models dm ON dm.model_code = ii.model_code
            WHERE ii.status = 'available'
            GROUP BY 1
            ORDER BY 2 DESC
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


def forecast_total(days=90):
    """Run Prophet on total daily sales."""
    df = get_total_sales()
    if len(df) < 7:
        return None

    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode="multiplicative",
    )
    m.fit(df)

    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)

    # Merge actuals
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result = result.merge(df, on="ds", how="left")
    result["ds"] = result["ds"].dt.strftime("%Y-%m-%d")

    return result.to_dict(orient="records")


def forecast_by_model(days=30):
    """Run Prophet per model (top models only)."""
    df = get_sales_data()
    if len(df) < 7:
        return {}

    # Only forecast models with enough data
    model_counts = df.groupby("modelo")["y"].sum().sort_values(ascending=False)
    top_models = model_counts.head(15).index.tolist()

    results = {}
    for modelo in top_models:
        model_df = df[df["modelo"] == modelo][["ds", "y"]].copy()

        # Fill missing dates with 0
        date_range = pd.date_range(model_df["ds"].min(), model_df["ds"].max())
        model_df = model_df.set_index("ds").reindex(date_range, fill_value=0).reset_index()
        model_df.columns = ["ds", "y"]

        if len(model_df) < 7:
            continue

        try:
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                seasonality_mode="multiplicative",
            )
            m.fit(model_df)

            future = m.make_future_dataframe(periods=days)
            fc = m.predict(future)

            # Get forecast for future period only
            future_fc = fc[fc["ds"] > model_df["ds"].max()][["ds", "yhat"]].copy()
            future_fc["yhat"] = future_fc["yhat"].clip(lower=0).round(1)
            future_fc["ds"] = future_fc["ds"].dt.strftime("%Y-%m-%d")

            # Sum forecast for next 15 and 30 days
            total_15 = future_fc.head(15)["yhat"].sum()
            total_30 = future_fc.head(30)["yhat"].sum()

            results[modelo] = {
                "forecast_15d": round(total_15),
                "forecast_30d": round(total_30),
                "daily": future_fc.to_dict(orient="records"),
            }
        except Exception:
            continue

    return results


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/forecast/total")
def api_forecast_total():
    days = request.args.get("days", 90, type=int)
    result = forecast_total(days)
    if result is None:
        return jsonify({"error": "Not enough data"}), 400
    return jsonify(result)


@app.route("/forecast/models")
def api_forecast_models():
    days = request.args.get("days", 30, type=int)
    result = forecast_by_model(days)
    return jsonify(result)


@app.route("/forecast/compras")
def api_forecast_compras():
    """Calculate purchase recommendations: forecast 15d - current stock."""
    days = 15
    model_forecast = forecast_by_model(days)

    try:
        stock_df = get_stock_by_model()
        stock_map = dict(zip(stock_df["modelo"], stock_df["stock"]))
    except Exception:
        stock_map = {}

    recommendations = []
    for modelo, data in model_forecast.items():
        forecast_15d = data["forecast_15d"]
        current_stock = stock_map.get(modelo, 0)
        deficit = max(0, forecast_15d - current_stock)

        recommendations.append({
            "modelo": modelo,
            "forecast_15d": forecast_15d,
            "stock_actual": current_stock,
            "comprar": deficit,
        })

    # Sort by comprar descending
    recommendations.sort(key=lambda x: x["comprar"], reverse=True)
    return jsonify(recommendations)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
