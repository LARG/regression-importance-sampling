
Assuming you are in the lds subdirectory.

1. protoc --python_out=. *.proto

2. python run_lds.py /path/to/results/directory/ --num_trials=200

3. python plot_lds_experiment.py /path/to/results/directory/


Plotting:

Plotting can be configured by uncommenting the appropriate lines at the top of plot_gridworld.py. In particular, plottype should be equal to 'gw-baselines' for Figures 2a or 2c. It should be 'gw-alternatives' for Figures 2b or 2d. And it should be 'gw-all-baselines' for the figures in the appendx.

The flag 'offpolicy' configures the methods to plot for on- or off-policy experiments.

