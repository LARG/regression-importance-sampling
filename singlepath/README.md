
Assuming you are in the singlepath subdirectory.

1. protoc --python_out=. *.proto
2. python run_singlepath.py /path/to/results/directory/ --num_trials=200
3. python plot_singlepath.py /path/to/results/directory/

