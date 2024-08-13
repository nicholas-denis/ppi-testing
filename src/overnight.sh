echo "Running experiment 1 16g_nl_xgb";
python main.py --config ../configs/Experiment_1_new/16g_nl_xgb.yaml

echo "Running experiment 2 128g_nl_xgb";
python main.py --config ../configs/Experiment_1_new/128g_nl_xgb.yaml

# shut off windows machine after running the script
shutdown -s

# chmod +x overnight.sh
# ./overnight.sh