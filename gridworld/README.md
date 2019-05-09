
Assuming you are in the gridworld subdirectory.

1. Google protobuf should be installed. Assuming it is, compile the necessary protobuf files. Do this by:
  cd protos; bash compile.sh

2. Compile the code. Run make. If the compile fails, run make again.

3. python run_gridworld.py /path/to/results/directory/ --num_trials=100

4. python plot_gridworld.py /path/to/results/directory/
   See below for instructions to configure the plots.

To run on-policy, use:

python run_gridworld.py /path/to/results/directory/ --num_trials=100 --on-policy



Plotting:

Plotting can be configured by uncommenting the appropriate lines at the top of plot_gridworld.py. In particular, plottype should be equal to 'gw-baselines' for Figures 2a or 2c. It should be 'gw-alternatives' for Figures 2b or 2d. And it should be 'gw-all-baselines' for the figures in the appendx.

The flag 'offpolicy' configures the methods to plot for on- or off-policy experiments.

