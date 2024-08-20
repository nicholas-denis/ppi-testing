echo "Running experiment 2";
python main.py --config ../configs/Experiment_3/50kt_nonlin_xgb.yaml

echo "Running experiment 3";
python main.py --config ../configs/Experiment_3/100kt_nonlin_xgb.yaml


# shut off windows machine after running the script
# shutdown -s

# chmod +x overnight.sh
# ./overnight.sh