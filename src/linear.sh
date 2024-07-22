echo "Running experiment 1, linear regression, gold 8";
python main.py --config ../configs/Experiment_1/reg_lin_8.yaml

echo "Running experiment 1, linear regression, gold 16";
python main.py --config ../configs/Experiment_1/reg_lin_16.yaml

echo "Running experiment 1, linear regression, gold 128";
python main.py --config ../configs/Experiment_1/reg_lin_128.yaml

echo "Running experiment 1, linear regression, gold 1024";
python main.py --config ../configs/Experiment_1/reg_lin_1024.yaml

echo "Running experiment 1, xgboost, gold 8";
python main.py --config ../configs/Experiment_1/xgboost_8.yaml

echo "Running experiment 1, xgboost, gold 16";
python main.py --config ../configs/Experiment_1/xgboost_16.yaml

echo "Running experiment 1, xgboost, gold 128";
python main.py --config ../configs/Experiment_1/xgboost_128.yaml

echo "Running experiment 1, xgboost, gold 1024";
python main.py --config ../configs/Experiment_1/xgboost_1024.yaml

# chmod +x linear.sh
# ./linear.sh