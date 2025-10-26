from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Import prediction model functions
from model import (
    load_and_preprocess_data,
    predict_crop_price,
)

app = Flask(__name__, static_url_path='/static')


# Load data at startup
try:
    df = load_and_preprocess_data()
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    df = None

@app.route('/')
def index():
    # Get list of unique commodities and states for dropdowns
    commodities = []
    states = []
    
    if df is not None:
        commodities = sorted(df['commodity'].unique())
        states = sorted(df['state'].unique())
    
    return render_template('page.html',
                          commodities=commodities,
                          states=states)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        commodity = request.form['commodity']
        state = request.form['state']
        time_unit = request.form['time_unit']
        steps_ahead = int(request.form['steps_ahead'])
        
        # Validate inputs
        if time_unit not in ["week", "month"]:
            return jsonify({"error": "Invalid time unit. Choose 'week' or 'month'."})
        
        if steps_ahead <= 0:
            return jsonify({"error": "Number of steps must be positive."})
            
        # Run prediction model
        models, best_model, price_series, original_steps, gap_size = predict_crop_price(
            df, commodity, state, steps_ahead, time_unit
        )
        
        if models is None or best_model is None:
            return jsonify({"error": "Prediction failed. Check if data exists for the selected commodity and state."})
        
        
        # Get predictions for display
        predictions = {}
        best_pred = None
        
        for model_name, preds in models.items():
            if preds is not None and len(preds) > 0:
                if isinstance(preds, pd.Series):
                    last_pred = preds.iloc[-1]
                else:
                    last_pred = preds[-1]
                price_per_kg = float(last_pred) / 100  # Convert to per kg
                predictions[model_name] = round(price_per_kg, 2)
        
        if best_model and models[best_model] is not None and len(models[best_model]) > 0:
            if isinstance(models[best_model], pd.Series):
                best_pred_price = models[best_model].iloc[-1]
            else:
                best_pred_price = models[best_model][-1]
            best_pred = round(float(best_pred_price) / 100, 2)
        
        # Return results
        return jsonify({
            "success": True,
            "predictions": predictions,
            "best_model": best_model,
            "best_prediction": best_pred
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
