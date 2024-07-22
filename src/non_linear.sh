echo "Running experiment 1, linear regression";
python main.py --config ../configs/reg_squared_noise.yaml

echo "Running experiment 2, xgboost regression";
python main.py --config ../configs/xgb_squared_noise.yaml

echo "Running experiment 3, random forest regression";
python main.py --config ../configs/rf_squared_noise.yaml