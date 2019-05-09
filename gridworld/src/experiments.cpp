#include "experiments.h"
#include "utils.h"
#include "results.pb.h"
#include "ope_methods.h"


// Experiments to add
// WIS, PDIS, DR, WDR, IS x weighted and unweighted
// Evaluated smoothing 0.0, 0.1,....,1.0 for OIS
// VF learning on gridworld w/ multi-step returns
// WIS on real domains


void OffPolicyBatch(Environment *env, int target_policy_number,
                    int behavior_policy_number, int numIter, 
                    int trajs_per_iter, int seed, string outfile,
                    int print_freq, bool estimate_pib, bool per_decision,
                    bool weighted, bool use_control_variate, float p, bool use_hold_out_set,
                    bool use_all_data) {
  OffPolicyEvaluate(env, target_policy_number, behavior_policy_number, numIter,
                    trajs_per_iter, seed, outfile, print_freq,
                    estimate_pib, per_decision, weighted, use_control_variate, p, true,
                    use_hold_out_set, use_all_data);
}


void OffPolicyEvaluate(Environment *env, int target_policy_number,
                       int behavior_policy_number, int numIter, 
                       int trajs_per_iter, int seed, string outfile,
                       int print_freq, bool estimate_pib, bool per_decision,
                       bool weighted, bool use_control_variate, float p, bool batch_mode,
                       bool use_hold_out_set, bool use_all_data) {

  printf("Target Policy %d\n", target_policy_number);
  printf("Behavior Policy %d\n", behavior_policy_number);

  Policy pi = env->getPolicy(target_policy_number);
  Policy behavior_pi = env->getPolicy(behavior_policy_number);
  vector<Trajectory> data;
  vector<Trajectory> estimate_data;
  vector<Trajectory> hold_out_data;

  mt19937_64 generator(seed);

  int actual_trajs_per_iter = trajs_per_iter;
  double true_value = env->getTrueValue(pi);

  VectorXd state_counts = VectorXd::Zero(env->getNumStates());
  MatrixXd state_action_counts = MatrixXd::Zero(env->getNumStates(), env->getNumActions());

  if (use_hold_out_set)
    initCounts(state_action_counts, state_counts, 1);
  else
    initCounts(state_action_counts, state_counts, 0);

  policy_gradient::ImprovementResults result_proto;
  for (int i = 0; i < numIter; i++) {

    env->generateTrajectories(data, behavior_pi, actual_trajs_per_iter,
                              generator);
    LoadTrajIWs(data, pi);

    if (estimate_pib and not use_hold_out_set)
      updateCounts(data, state_action_counts, state_counts);


    for (auto & traj : data) {
      estimate_data.push_back(traj);
      if (use_all_data)
        hold_out_data.push_back(traj);
    }

    // We do this regardless of whether we use a hold out set or not.
    // This is to reduce variance between runs and make sure each method uses
    // the same trajectories.
    env->generateTrajectories(data, behavior_pi, actual_trajs_per_iter,
                              generator);

    for (auto & traj : data) {
      hold_out_data.push_back(traj);
    }

    if (estimate_pib and (use_hold_out_set or use_all_data))
      updateCounts(data, state_action_counts, state_counts);

    if (estimate_pib) {
      loadMLEWeightsFromCounts(estimate_data, pi, state_action_counts, state_counts);
    }
    if (p > 0) {
      batch::smooth_IWs(estimate_data, p, env->getNumActions());
    }

    float estimate = 0;
    float mse;

    if (use_control_variate) {
      estimate = batch::doubly_robust_estimate(estimate_data, pi, env, weighted);
    } else {
      estimate = batch::importance_sampling_estimate(estimate_data, weighted, per_decision, env->getMaxTrajLen());
    }

    // Record results
    mse = pow(true_value - estimate, 2.0);
    if (i == 0) printf("Itr.\tTrue Value\tEstimate\tMSE\n");
    if (i % print_freq == 0) printf("%d\t%f\t%f\t%f\n", i, true_value, estimate, mse);
    result_proto.add_dataset_sizes(actual_trajs_per_iter);
    result_proto.add_avg_return(true_value);
    result_proto.add_mse(mse);
    result_proto.add_estimated_avg_return(estimate);

  }

  fstream output(outfile, ios::out | ios::trunc | ios::binary);
  if (!result_proto.SerializeToOstream(&output)) {
    cerr << "Failed to write results." << endl;
  }
}
