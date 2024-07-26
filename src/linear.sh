 echo "Running experiment 1, xgboost, gold 8";
python main.py --config ../configs/Experiment_1/xgb_lin_8.yaml

echo "Running experiment 1, xgboost, gold 16";
python main.py --config ../configs/Experiment_1/xgb_lin_16.yaml

echo "Running experiment 1, xgboost, gold 128";
python main.py --config ../configs/Experiment_1/xgb_lin_128.yaml

echo "Running experiment 1, xgboost, gold 1024";
python main.py --config ../configs/Experiment_1/xgb_lin_1024.yaml

# chmod +x linear.sh
# ./linear.sh