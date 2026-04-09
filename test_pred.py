import xgboost_newfinal_code as backend

backend.train_models_if_needed()

for time in ["2025-09-16 10:00:00", "2025-09-16 11:00:00", "2025-09-16 12:00:00"]:
    res = backend.get_raw_prediction(time, horizon=1)
    print(time, res)
