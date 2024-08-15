echo "Running experiment 1";
python main.py --config ../configs/Experiment_1_new/16g_nl_reg.yaml  # random one that needs to be rerun

echo "Running experiment 2";
python main.py --config ../configs/Experiment_1_new/512g_l_reg.yaml

echo "Running experiment 3";
python main.py --config ../configs/Experiment_1_new/512g_nl_reg.yaml

echo "Running experiment 4";
python main.py --config ../configs/Experiment_1_new/512g_nl_xgb.yaml

echo "Running experiment 5";
python main.py --config ../configs/Experiment_1_new/1024g_l_reg.yaml

echo "Running experiment 6";
python main.py --config ../configs/Experiment_1_new/1024g_nl_reg.yaml

echo "Running experiment 7";
python main.py --config ../configs/Experiment_1_new/1024g_nl_xgb.yaml

# shut off windows machine after running the script
# shutdown -s

# chmod +x overnight.sh
# ./overnight.sh