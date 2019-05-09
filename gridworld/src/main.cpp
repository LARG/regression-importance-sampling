
#include <math.h>
#include <iostream>
#include <fstream>

#include <Includes.hpp>

#include "utils.h"
#include "experiments.h"
#include "Gridworld.h"
#include "Trajectory.h"
#include "Model.hpp"
#include "results.pb.h"

using namespace std;


 void printHelp() {
  printf("--seed\t\tRandom seed\n");
  printf("--outfile\tResult file\n");
  printf("--weighted\tUse weighted importance sampling.\n");
  printf("--per-decision\tUse per-decision importance sampling.\n");
  printf("--iter\t\tNumber of iterations to run\n");
  printf("--iter-trajs\tTrajectories to collect per iteration\n");
  printf("--policy-number\tPolicy number to use for initial policy\n");
  printf("--behavior-number\tPolicy number to use for behavior policy\n");
  printf("--print_freq\t\tNumber of iterations to skip between evaluations\n");
  printf("--help\t\tPrint this message\n");
}

int main(int nargs, char* args[]) {

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  int seed = 0;
  int method = 0;
  int pg_method = 0;
  int numIter = 1000;
  int trajs_per_iter = 10;
  int policy_number = 2;
  int behavior_policy_number = 1;
  int print_freq = 10;
  float smooth_param = 0.0;
  bool use_control_variate = false;
  bool rare_event = false;
  bool per_decision = false;
  bool weighted = false;
  bool estimate_pib = false;
  bool use_hold_out_set = false;
  bool use_all_data = false;
  string input;
  string outfile_name("value.txt");

  for (int i=1; i < nargs; i++) {

    if (strcmp(args[i], "--seed") == 0 && i+1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> seed))
        cout << "Could not parse random seed " << endl;
    }
    if (strcmp(args[i], "--outfile") == 0 && i + 1 < nargs)
      outfile_name = args[i+1];
    if (strcmp(args[i],"--iter") == 0 && i + 1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> numIter))
        cout << "Could not parse number of iterations" << endl;
    }
    if (strcmp(args[i],"--print_freq") == 0 && i + 1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> print_freq))
        cout << "Could not parse number of iterations to skip" << endl;
    }
    if (strcmp(args[i],"--iter-trajs") == 0 && i + 1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> trajs_per_iter))
        cout << "Could not parse trajectories per iteration" << endl;
    }
    if (strcmp(args[i],"--policy-number") == 0 && i + 1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> policy_number))
        cout << "Could not parse policy number" << endl;
    }
    if (strcmp(args[i],"--behavior-number") == 0 && i + 1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> behavior_policy_number))
        cout << "Could not parse policy number" << endl;
    }
    if (strcmp(args[i],"--smooth_param") == 0 && i + 1 < nargs) {
      input = args[i + 1];
      istringstream iss(input);
      if (! (iss >> smooth_param))
        cout << "Could not parse p" << endl;
    }
    if (strcmp(args[i],"--help") == 0) {
      printHelp();
      return 0;
    }
    if (strcmp(args[i],"--per-decision") == 0) {
      per_decision = true;
    }
    if (strcmp(args[i],"--weighted") == 0) {
      weighted = true;
    }
    if (strcmp(args[i],"--estimate") == 0) {
      estimate_pib = true;
    }
    if (strcmp(args[i], "--hold-out") == 0) {
      use_hold_out_set = true;
    }
    if (strcmp(args[i], "--all-data") == 0) {
      use_all_data = true;
    }
    if (strcmp(args[i],"--use-cv") == 0) {
      printf("Learn control variate\n");
      use_control_variate = true;
    }


  }

  Environment *env = new Gridworld(false);
  int n_states = env->getNumStates();
  int n_actions = env->getNumActions();
  int horizon = env->getMaxTrajLen();

  printf("Running %d iterations\n", numIter);
  printf("Batch Size %d\n", trajs_per_iter);

  if (behavior_policy_number == -1)
    behavior_policy_number = policy_number;

  OffPolicyBatch(env, policy_number, behavior_policy_number, numIter, 
                 trajs_per_iter, seed, outfile_name, print_freq,
                 estimate_pib, per_decision, weighted, use_control_variate, smooth_param,
                 use_hold_out_set, use_all_data);

  delete env;

  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}

