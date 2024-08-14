echo "Running experiment 3 16g_l_reg";
python main.py --config ../configs/Experiment_1_new/16g_l_reg.yaml

echo "Running experiment 4 128g_l_reg";
python main.py --config ../configs/Experiment_1_new/128g_l_reg.yaml

echo "Running experiment 5 16g_nl_reg";
python main.py --config ../configs/Experiment_1_new/16g_nl_reg.yaml

echo "Running experiment 6 128g_nl_reg";
python main.py --config ../configs/Experiment_1_new/128g_nl_reg.yaml

# shut off windows machine after running the script
#shutdown -s

# chmod +x overnight.sh
# ./overnight.sh