# run every config in experiment_2_new folder

for file in ../configs/Experiment_2_new_unrun/*.yaml
do
    echo "Running $file";
    python main.py --config $file
done


# shut off windows machine after running the script

# chmod +x overnight_2.sh
# ./overnight_2.sh