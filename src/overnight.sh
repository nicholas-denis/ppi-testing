echo "Running experiment 5, regression, gold 8", squared;
python main.py --config ../configs/Experiment_1/reg_sq_8.yaml

echo "Running experiment 6, regression, gold 16", squared;
python main.py --config ../configs/Experiment_1/reg_sq_16.yaml

echo "Running experiment 7, regression, gold 128", squared;
python main.py --config ../configs/Experiment_1/reg_sq_128.yaml

echo "Running experiment 8, regression, gold 1024", squared;
python main.py --config ../configs/Experiment_1/reg_sq_1024.yaml
# chmod +x overnight.sh
# ./overnight.sh