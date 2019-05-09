#include "Trajectory.h"
#include "Model.hpp"
#include "Environment.hpp"

namespace batch {

float importance_sampling_estimate(const std::vector<Trajectory> &trajs,
                                   bool weighted, bool per_decision, int L);


float model_based_estimate(std::vector<Trajectory> &trajs,
                           const Policy &pi, const Environment *env);



float doubly_robust_estimate(std::vector<Trajectory> &trajs, 
                             const Policy &pi, const Environment *env,
                             bool weighted);


void smooth_IWs(std::vector<Trajectory> &trajs, float p, int num_actions);

};