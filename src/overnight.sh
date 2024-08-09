echo "Running experiment 1 1kt_c_lin_reg";
python main.py --config ../configs/Experiment_2/1kt_c_lin_reg.yaml

echo "Running experiment 2 1kt_nc_lin_reg";
python main.py --config ../configs/Experiment_2/1kt_nc_lin_reg.yaml

echo "Running experiment 3 50kt_c_lin_reg";
python main.py --config ../configs/Experiment_2/50kt_c_lin_reg.yaml

echo "Running experiment 4 50kt_nc_lin_reg";
python main.py --config ../configs/Experiment_2/50kt_nc_lin_reg.yaml

echo "Running experiment 5 1mt_c_lin_reg";
python main.py --config ../configs/Experiment_2/1mt_c_lin_reg.yaml

echo "Running experiment 5 1mt_nc_lin_reg";
python main.py --config ../configs/Experiment_2/1mt_nc_lin_reg.yaml

# shut off windows machine after running the script

# chmod +x overnight.sh
# ./overnight.sh