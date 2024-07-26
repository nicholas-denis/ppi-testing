echo "Running experiment 3, xgboost, gold 128";
python main.py --config ../configs/Experiment_1/xgb_sq_128.yaml

echo "Running experiment 4, xgboost, gold 1024";
python main.py --config ../configs/Experiment_1/xgb_sq_1024.yaml

# chmod +x linear_xgb_only.sh
# ./linear_xgb_only.sh