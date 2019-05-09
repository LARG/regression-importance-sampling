
#include "ope_methods.h"
#include "utils.h"

float batch::importance_sampling_estimate(const std::vector<Trajectory> &trajs,
                                          bool weighted, bool per_decision, int L) {
  double estimate = 0;

  // getIWs takes care of normalization for weighted or unweighted.
  MatrixXd rhot = getIWs(trajs, weighted, L);

  for (unsigned int i=0; i < trajs.size(); i++) {
    if (per_decision){
      for (int t=0; t < trajs[i].len; t++) {
        estimate += trajs[i].rewards[t] * rhot(i, t);
      }
    } else {
      estimate += trajs[i].R * rhot(i, trajs[i].len - 1);
    }
  }
  return estimate;
}


float batch::model_based_estimate(std::vector<Trajectory> &trajs,
                                  const Policy &pi, const Environment *env) {
  float estimate;
  std::vector<Trajectory*> model_data;
  for (auto & traj : trajs){
    model_data.push_back(&traj);
  }
  Model m(model_data, env->getNumStates(), env->getNumActions(),
          env->getMaxTrajLen(), false);
  m.loadEvalPolicy(pi, env->getMaxTrajLen());
  estimate = m.evalPolicyValue;
  return estimate;
}

void batch::smooth_IWs(std::vector<Trajectory> &trajs, float p, int num_actions) {

  float prob;
  float uniform_prob = 1.0 / num_actions;
  for (auto & traj : trajs) {
    for (int t=0; t < traj.len; t++) {
      // prob = traj.actionProbabilities[t] / traj.IWs[t];
      // std::cout << prob - traj.actionProbabilities[t] << std::endl;
      prob = traj.actionProbabilities[t];
      prob = p * uniform_prob + (1 - p) * prob;
      traj.IWs[t] = traj.evalActionProbabilities[t] / prob;
      traj.cumIWs[t] = traj.IWs[t];
      if (t != 0)
        traj.cumIWs[t] *= traj.cumIWs[t-1];
    }
  }

}


float batch::doubly_robust_estimate(std::vector<Trajectory> &trajs, const Policy &pi,
                                    const Environment *env, bool weighted) {

  float estimate = 0;
  bool half_data = true;
  bool v2 = false;
  int limit = trajs.size();
  int L = env->getMaxTrajLen();
  if (half_data)
    limit = limit / 2;
  std::vector<Trajectory*> model_data;
  std::vector<Trajectory> dr_data;
  for (unsigned int i=0; i < limit; i++)
    model_data.push_back(&trajs[i]);
  if (not half_data)
    limit = 0;
  for (unsigned int i=limit; i < trajs.size(); i++)
    dr_data.push_back(trajs[i]);

  Model m(model_data, env->getNumStates(), env->getNumActions(), L, false);
  m.loadEvalPolicy(pi, L);

  double PDISEstimate = importance_sampling_estimate(dr_data, weighted, true, L);

  double baseline = 0;

  for (int t = 0; t < env->getMaxTrajLen(); t++) {
    double term_t = 0, IWSum = 0;
    for (unsigned int i = 0; i < dr_data.size(); i++) {
      double rHat = 0;
      if (t < dr_data[i].len) { // Else Rhat = 0
        // Compute Q differently depending on whether or not we are using v2
        double Q;
        int s = dr_data[i].states[t], a = dr_data[i].actions[t];
        if (v2) {
          // The lines below should be equivalent to: 
          // Q = m.Rsa[t](s,a);, but without using m.Rsa (which takes a long time to load)
          Q = 0;
          //if (t != L-1) {
          int numStates = m.P.size();
          for (int sPrime = 0; sPrime <= numStates; sPrime++) {
            Q += m.P[s][a][sPrime] * m.R[s][a][sPrime];
          }
          //}

          if (t < dr_data[i].len-1)
            Q += m.V[t+1](dr_data[i].states[t+1]);
        }
        else {
          Q = m.Q[t](s,a);
        }
        rHat = Q;
        if (t < dr_data[i].len - 1)
          rHat -= m.V[t + 1](dr_data[i].states[t + 1]);
        term_t += dr_data[i].cumIWs[t] * rHat;
        IWSum += dr_data[i].cumIWs[t];
      } else {
        IWSum += dr_data[i].cumIWs[dr_data[i].len-1];
      }
    }
    if (weighted)
      term_t /= IWSum;
    else
      term_t /= (double)dr_data.size();
    baseline += term_t;
  }
  double lastTerm = 0;
  for (unsigned int i = 0; i < dr_data.size(); i++)
    lastTerm += m.V[0](dr_data[i].states[0]);
  lastTerm /= static_cast<double>(dr_data.size());       // No importance weights - always one.
  baseline -= lastTerm;
  std::cout << baseline << std::endl;
  return PDISEstimate - baseline;
  // return lastTerm;
}
