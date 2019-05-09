#include "Environment.hpp"


void OffPolicyBatch(Environment *env, int target_policy_number,
                       int behavior_policy_number, int numIter, 
                       int trajs_per_iter, int seed, string outfile,
                       int print_freq, bool estimate_pib, bool per_decision,
                       bool weighted, bool use_control_variate, float p, bool use_hold_out_set,
                       bool use_all_data);

void OffPolicyEvaluate(Environment *env, int target_policy_number,
                      int behavior_policy_number, int numIter, 
                      int trajs_per_iter, int seed, string outfile,
                      int print_freq, bool estimate_pib,
                      bool per_decision, bool weighted, bool use_control_variate, float p,
                      bool batch_mode, bool use_hold_out_set, bool use_all_data);
