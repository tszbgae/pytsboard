import os
import pandas as pd
import numpy as np
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from statsmodels.tsa.seasonal import seasonal_decompose

from darts import TimeSeries
from darts.models import (
    LightGBMModel, 
    XGBModel, 
    RandomForest, 
    Prophet, 
    ARIMA
)
from darts.metrics import mape, rmse, mae
# ... import your other dependencies (pandas, darts, etc.)

# 1. FIND THE STATIC FOLDER RELATIVE TO THIS FILE
# This ensures it works no matter where the user installs the package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app) # Optional in production, but keeps things safe

# --- HELPER: Downsampling ---
def smart_downsample(df, target_points=2000):
    if len(df) <= target_points:
        return df
    
    chunk_size = len(df) // target_points
    
    # numeric_only=False allows us to try to aggregate everything, 
    # but mean() will fail on strings. Since we encoded strings to ints 
    # BEFORE calling this, mean() will work on them (which is acceptable for ordinal trends).
    downsampled = df.groupby(np.arange(len(df)) // chunk_size).mean(numeric_only=True)
    
    # Restore Date Column (Representative date for the chunk)
    for col in df.columns:
        if col.lower() in ['date', 'timestamp', 'time', 'year']:
            original_dates = df[col].iloc[::chunk_size].reset_index(drop=True)
            min_len = min(len(downsampled), len(original_dates))
            downsampled = downsampled.iloc[:min_len]
            downsampled[col] = original_dates.iloc[:min_len].values
            break
            
    return downsampled

# ... [PASTE ALL YOUR EXISTING API ROUTES HERE] ...
# (/api/analyze, /api/decompose, /api/predict, etc.)

# 2. SERVE REACT FRONTEND
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # Fallback to index.html for React Router (SPA behavior)
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        options = json.loads(request.form.get('options', '{}'))
        selected_columns = options.get('selectedColumns', [])

        # 1. LOAD FULL DATAFRAME
        df = pd.read_csv(file)
        
        # Filter Columns
        if selected_columns:
            existing_cols = [c for c in selected_columns if c in df.columns]
            df = df[existing_cols]

        # 2. DATE PARSING
        # We must identify the date column to exclude it from Ordinal Encoding
        date_col_name = None
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Check if it looks like a date
                    pd.to_datetime(df[col], errors='raise')
                    df[col] = pd.to_datetime(df[col])
                    date_col_name = col
                except:
                    pass # Not a date

        # 3. ORDINAL ENCODING (The new requirement)
        # We track mappings to send to frontend: {'City': {'New York': 0, 'Chicago': 1}}
        encoding_map = {}
        
        for col in df.columns:
            # If it's a string (object) and NOT the date column
            if df[col].dtype == 'object' and col != date_col_name:
                # Convert to category and then to codes (0, 1, 2...)
                df[col] = df[col].astype('category')
                
                # Save the mapping (Category Name -> Integer)
                mapping = dict(enumerate(df[col].cat.categories))
                # Flip it to be { "Name": 0 } for easier lookup if needed
                encoding_map[col] = {v: k for k, v in mapping.items()}
                
                # Apply the encoding
                df[col] = df[col].cat.codes

        # 4. PREPARE FULL DATA (For "Other Analysis")
        # Handle NaNs and Dates for JSON
        full_df = df.copy()
        if date_col_name:
            full_df[date_col_name] = full_df[date_col_name].dt.strftime('%Y-%m-%d')
        full_df = full_df.where(pd.notnull(full_df), None)
        full_data_json = full_df.to_dict(orient='records')

        # 5. PREPARE DOWNSAMPLED DATA (For Graphing)
        if len(df) > 2000:
            print(f"Downsampling from {len(df)} to 2000 points...")
            df_down = smart_downsample(df, target_points=2000)
        else:
            df_down = df

        # Final cleanup for downsampled version
        if date_col_name and date_col_name in df_down:
             df_down[date_col_name] = df_down[date_col_name].dt.strftime('%Y-%m-%d')
        df_down = df_down.where(pd.notnull(df_down), None)
        downsampled_json = df_down.to_dict(orient='records')

        # 6. RETURN BOTH
        return jsonify({
            "status": "success",
            "columns": list(df.columns),
            "data": downsampled_json,      # Small DF (Graphing)
            "full_data": full_data_json,   # Big DF (Analysis)
            "mappings": encoding_map,      # Reference for what 0,1,2 mean
            "rows": len(full_df)
        })

    except Exception as e:
        print(f"CRITICAL SERVER ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/decompose', methods=['POST'])
def decompose_data():
    # ... (Keep your existing decompose_data function here) ...
    # Note: Since the frontend will now send encoded numbers for string cols,
    # the existing decompose logic will handle them fine as numbers!
    try:
        req_data = request.json
        series_data = req_data.get('data', [])
        column_name = req_data.get('column', 'value')
        
        df = pd.DataFrame(series_data)
        
        # DOWNSAMPLE
        if len(df) > 1000:
             df = smart_downsample(df, target_points=1000)

        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        df = df.dropna(subset=[column_name])

        period = req_data.get('period', 12)
        if len(df) < (period * 2): period = 2

        decomposition = seasonal_decompose(df[column_name], model='additive', period=period, extrapolate_trend='freq')

        result = {
            "observed": decomposition.observed.tolist(),
            "trend": decomposition.trend.tolist(),
            "seasonal": decomposition.seasonal.tolist(),
            "residual": decomposition.resid.tolist()
        }
        return jsonify({"status": "success", "results": result})

    except Exception as e:
        print(f"Decomposition Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_data():
    try:
        req = request.json
        full_data = req.get('data', [])
        target_col = req.get('targetColumn')
        feature_cols = req.get('featureColumns', [])
        date_col = req.get('dateColumn')
        horizon = int(req.get('horizon', 12))
        model_name = req.get('model', 'lightgbm').lower()

        if not full_data or not target_col:
            return jsonify({"error": "Missing data or target column"}), 400

        # 1. Prepare DataFrame
        df = pd.DataFrame(full_data)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col])

        # 2. Create TimeSeries (Target)
        try:
            if date_col and date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(by=date_col)
                series = TimeSeries.from_dataframe(df, time_col=date_col, value_cols=target_col, fill_missing_dates=True, freq=None)
            else:
                series = TimeSeries.from_dataframe(df, value_cols=target_col)
        except:
            series = TimeSeries.from_dataframe(df, value_cols=target_col)

        # 3. Handle Covariates (Features)
        past_covariates = None
        if feature_cols:
            for f in feature_cols:
                df[f] = pd.to_numeric(df[f], errors='coerce')
            df = df.fillna(0)
            try:
                if date_col:
                    past_covariates = TimeSeries.from_dataframe(df, time_col=date_col, value_cols=feature_cols, fill_missing_dates=True, freq=None)
                else:
                    past_covariates = TimeSeries.from_dataframe(df, value_cols=feature_cols)
            except:
                pass

        # 4. CONFIGURE MODEL
        lags = 12
        
        # Define which models allow past_covariates
        supports_past_covariates = model_name in ['lightgbm', 'xgboost', 'randomforest']
        
        # If model doesn't support them, ensure we pass None
        model_covariates = past_covariates if supports_past_covariates else None

        if model_name == 'arima':
            model_class = ARIMA(p=12, d=1, q=2)
        elif model_name == 'prophet':
            model_class = Prophet()
        elif model_name == 'randomforest':
            model_class = RandomForest(lags=lags, lags_past_covariates=lags if model_covariates else None)
        elif model_name == 'xgboost':
            model_class = XGBModel(lags=lags, lags_past_covariates=lags if model_covariates else None)
        else:
            # LightGBM (Default)
            model_class = LightGBMModel(lags=lags, lags_past_covariates=lags if model_covariates else None)

        # 5. SPLIT & BACKTEST
        split_idx = int(len(series) * 0.75)
        train, val = series[:split_idx], series[split_idx:]

        # Fit & Predict (Conditionally passing covariates)
        if model_covariates:
            model_class.fit(train, past_covariates=model_covariates)
            pred_val = model_class.predict(len(val), past_covariates=model_covariates)
        else:
            model_class.fit(train)
            pred_val = model_class.predict(len(val))

        scores = {
            "mape": round(float(mape(val, pred_val)), 2),
            "rmse": round(float(rmse(val, pred_val)), 2),
            "mae": round(float(mae(val, pred_val)), 2)
        }

        # 6. FUTURE FORECAST
        # Re-instantiate to avoid state conflicts
        if model_name == 'arima':
            model_full = ARIMA(p=12, d=1, q=2)
        elif model_name == 'prophet':
            model_full = Prophet()
        elif model_name == 'randomforest':
            model_full = RandomForest(lags=lags, lags_past_covariates=lags if model_covariates else None)
        elif model_name == 'xgboost':
            model_full = XGBModel(lags=lags, lags_past_covariates=lags if model_covariates else None)
        else:
            model_full = LightGBMModel(lags=lags, lags_past_covariates=lags if model_covariates else None)

        if model_covariates:
            model_full.fit(series, past_covariates=model_covariates)
            pred_future = model_full.predict(horizon, past_covariates=model_covariates)
        else:
            model_full.fit(series)
            pred_future = model_full.predict(horizon)

        # 7. FORMAT RESULTS
        import numpy as np
        def to_list(ts): return ts.values().flatten().tolist()

        full_vals = to_list(series)
        pred_vals_list = to_list(pred_val)
        future_vals_list = to_list(pred_future)
        
        validation_graph = []
        offset = len(train)
        for i, val in enumerate(full_vals):
            point = {"index": i, "Actual": val, "Predicted": None, "split": "Train"}
            if i >= offset:
                pred_idx = i - offset
                if pred_idx < len(pred_vals_list): point["Predicted"] = pred_vals_list[pred_idx]
                point["split"] = "Test"
            validation_graph.append(point)

        future_graph = []
        for i, val in enumerate(full_vals):
            future_graph.append({"index": i, "Actual": val, "Predicted": None, "isFuture": False})
        last_idx = len(full_vals)
        for i, val in enumerate(future_vals_list):
            future_graph.append({"index": last_idx + i, "Actual": None, "Predicted": val, "isFuture": True})

        return jsonify({
            "status": "success",
            "metrics": scores,
            "validation_data": validation_graph,
            "future_data": future_graph
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# 3. ENTRY POINT FUNCTION
def start_dashboard():
    # --- DEBUG PRINTS ---
    print(f"📂 Current Directory: {os.getcwd()}")
    print(f"📂 App Base Dir: {BASE_DIR}")
    print(f"📂 Static Folder Target: {STATIC_DIR}")
    
    if os.path.exists(os.path.join(STATIC_DIR, 'index.html')):
        print("✅ SUCCESS: index.html found!")
    else:
        print("❌ FAILURE: index.html is MISSING from the installed package.")
        print(f"   Contents of {STATIC_DIR}:")
        try:
            print(os.listdir(STATIC_DIR))
        except:
            print("   (Directory does not exist)")
    # --------------------

    print(f"🚀 Dashboard launching on http://127.0.0.1:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == '__main__':
    start_dashboard()