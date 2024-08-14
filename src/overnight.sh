echo "Running experiment 1 1kt non lin no clip xgb";
python main.py --config ../configs/Experiment_2_new/1kt_nc_nonlin_xgb.yaml

echo "Running experiment 2 10kt non lin no clip xgb";
python main.py --config ../configs/Experiment_2_new/10kt_nc_nonlin_xgb.yaml

echo "Running experiment 3 100kt non lin no clip xgb";
python main.py --config ../configs/Experiment_2_new/100kt_nc_nonlin_xgb.yaml

echo "Running experiment 1 1kt non lin clip xgb";
python main.py --config ../configs/Experiment_2_new/1kt_c_nonlin_xgb.yaml

echo "Running experiment 1 10kt non lin clip xgb";
python main.py --config ../configs/Experiment_2_new/10kt_c_nonlin_xgb.yaml

echo "Running experiment 1 1kt non lin clip xgb";
python main.py --config ../configs/Experiment_2_new/100kt_c_nonlin_xgb.yaml

# shut off windows machine after running the script
shutdown -s

# chmod +x overnight.sh
# ./overnight.sh