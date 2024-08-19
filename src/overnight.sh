echo "Running experiment 1";
python main.py --config ../configs/Experiment_2_new/10kt_c_nonlin_xgb.yaml  # random one that needs to be rerun

echo "Running experiment 2";
python main.py --config ../configs/Experiment_2_new/50kt_c_nonlin_xgb.yaml

echo "Running experiment 3";
python main.py --config ../configs/Experiment_2_new/50kt_nc_nonlin_xgb.yaml

echo "Running experiment 4";
python main.py --config ../configs/Experiment_2_new/100kt_c_nonlin_xgb.yaml


# shut off windows machine after running the script
# shutdown -s

# chmod +x overnight.sh
# ./overnight.sh