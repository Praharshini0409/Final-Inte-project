from flask import Flask, request, jsonify, render_template
from datetime import datetime
import xgboost_newfinal_code as backend

app = Flask(__name__, 
            template_folder='updated_inte_pro/inte/inte/templates',
            static_folder='updated_inte_pro/inte/inte/static')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

@app.route('/drying-info')
def drying_info():
    return render_template('drying_info.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# API Endpoints
@app.route('/api/predict_drying_time', methods=['POST'])
def predict_drying_time():
    data = request.json
    try:
        user_datetime = f"{data['startDate']} {data['startTime']}:00"
        thickness = float(data['thickness'])
        user_temp = float(data.get('temperature', 40))
        product = data.get('product', 'other')
        
        time_pred = backend.get_drying_time(user_datetime, thickness, user_temp, product)
        
        if time_pred is None:
            return jsonify({'success': False, 'error': 'Invalid thickness.'}), 400
            
        return jsonify({'success': True, 'drying_time': time_pred})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# Dynamic routes linking to ML models and CSV History
@app.route('/api/live', methods=['GET'])
def live():
    try:
        return jsonify(backend.get_live_data())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def dashboard_api():
    try:
        interval = request.args.get('interval', 60)
        return jsonify(backend.get_historical_data(interval))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast', methods=['GET'])
def forecast_api():
    try:
        date_val = request.args.get('date')
        time_val = request.args.get('time')
        am_pm = request.args.get('am_pm', '')
        horizon = request.args.get('horizon', 1)
        
        # Combine date, time and am_pm
        if am_pm:
            user_datetime_str = f"{date_val} {time_val} {am_pm}"
            dt = datetime.strptime(user_datetime_str, "%Y-%m-%d %I:%M %p")
        else:
            # If front-end already converted to 24 hr
            user_datetime_str = f"{date_val} {time_val}:00"
            dt = datetime.strptime(user_datetime_str, "%Y-%m-%d %H:%M:%S")
            
        return jsonify(backend.get_raw_prediction(dt.strftime("%Y-%m-%d %H:%M:%S"), horizon))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    return jsonify({"response": "This is a mock chatbot interface. The backend model is prioritizing predictions right now."})

if __name__ == '__main__':
    backend.train_models_if_needed() # Ensures model is loaded when server boots
    app.run(host='0.0.0.0', port=5000, debug=True)
