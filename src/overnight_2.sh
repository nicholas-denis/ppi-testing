# run every config in experiment_2_new folder

for file in ../experiments/*;
do
    echo "Replotting $file";
    python plot_only.py --folder_path $file
done


# shut off windows machine after running the script

# chmod +x overnight_2.sh
# ./overnight_2.sh