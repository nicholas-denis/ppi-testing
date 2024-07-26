echo "Running experiment 1, linear regression, gold 8", linear transformation;
python main.py --config ../configs/Experiment_1/reg_lin_8.yaml

echo "Running experiment 2, linear regression, gold 16", linear transformation;
python main.py --config ../configs/Experiment_1/reg_lin_16.yaml

echo "Running experiment 3, linear regression, gold 128", linear transformation;
python main.py --config ../configs/Experiment_1/reg_lin_128.yaml

echo "Running experiment 4, linear regression, gold 1024", linear transformation;
python main.py --config ../configs/Experiment_1/reg_lin_1024.yaml

echo "Running experiment 5, linear regression, gold 8", squared transformation;
python main.py --config ../configs/Experiment_1/reg_sq_8.yaml

echo "Running experiment 6, linear regression, gold 16", squared transformation;
python main.py --config ../configs/Experiment_1/reg_sq_16.yaml

echo "Running experiment 7, linear regression, gold 128", squared transformation;
python main.py --config ../configs/Experiment_1/reg_sq_128.yaml

echo "Running experiment 8, linear regression, gold 1024", squared transformation;
python main.py --config ../configs/Experiment_1/reg_sq_1024.yaml