#include <vector>

#include "Trajectory.h"
#include "Environment.hpp"

void LoadTrajIWs(vector<Trajectory> & trajs, const Policy &eval_pi);

MatrixXd getIWs(const vector<Trajectory> trajs, const bool & weighted, const int & L);

void MLE_LoadTrajIWs(vector<Trajectory> &estimate_data, vector<Trajectory> &pib_data, const Policy &eval_pi,
                     const Environment &env, int smoothing);

void RISN_LoadTrajIWs(vector<Trajectory> &data, const Policy &eval_pi, int n,
                      bool reset);

void initCounts(MatrixXd &state_action_counts, VectorXd &state_counts, int smoothing);

void updateCounts(vector<Trajectory> &new_data, MatrixXd &state_action_counts, VectorXd &state_counts);

void loadMLEWeightsFromCounts(vector<Trajectory> &estimate_data, const Policy &eval_pi, MatrixXd &state_action_counts, VectorXd &state_counts);
