#include <map>
#include "utils.h"
#include "Trajectory.h"
#include "Environment.hpp"
#include "Gridworld.h"


void LoadTrajIWs(vector<Trajectory> & trajs, const Policy &eval_pi) {
  for (int i = 0; i < static_cast<int>(trajs.size()); i++) {
    trajs[i].IWs.resize(trajs[i].len);
    trajs[i].cumIWs.resize(trajs[i].len);
    trajs[i].evalActionProbabilities.resize(trajs[i].len);
    for (int t = 0; t < trajs[i].len; t++) {
      trajs[i].evalActionProbabilities[t] = eval_pi.getActionProbability(
        trajs[i].states[t], trajs[i].actions[t]);
      trajs[i].IWs[t] = trajs[i].evalActionProbabilities[t] /
        trajs[i].actionProbabilities[t];
      trajs[i].cumIWs[t] = trajs[i].IWs[t];
      if (t != 0)
        trajs[i].cumIWs[t] *= trajs[i].cumIWs[t-1];
    }
  }
}


MatrixXd getIWs(const vector<Trajectory> trajs, const bool & weighted, const int & L) {
  int N = (int)trajs.size();
  MatrixXd rhot(N, L);
  for (int i = 0; i < N; i++) {
    for (int t = 0; t < L; t++) {
      if (t < trajs[i].len)
        rhot(i,t) = trajs[i].cumIWs[t];
      else
        rhot(i,t) = trajs[i].cumIWs[trajs[i].len - 1];
    }
  }
  if (weighted) {
    for (int t = 0; t < L; t++)
      rhot.col(t) = rhot.col(t) / rhot.col(t).sum();
  }
  else
    rhot = rhot / (double)N;
  return rhot;
}


void initCounts(MatrixXd &state_action_counts, VectorXd &state_counts, int smoothing) {

  int k = smoothing;
  for (int s=0; s < state_counts.size(); s++) {
    state_counts(s) += k * state_action_counts.cols();
    for (int a=0; a < state_action_counts.cols(); a++) {
      state_action_counts(s, a) += k;
    }
  }
}


void updateCounts(vector<Trajectory> &new_data, MatrixXd &state_action_counts, VectorXd &state_counts) {
  for (auto & traj : new_data) {
    for (int t=0; t < traj.len; t++) {
      state_action_counts(traj.states[t], traj.actions[t]) += 1;
      state_counts(traj.states[t]) += 1;
    }
  } 
}


void loadMLEWeightsFromCounts(vector<Trajectory> &estimate_data, const Policy &eval_pi, MatrixXd &state_action_counts, VectorXd &state_counts) {

  int numStates = state_action_counts.rows();
  int numActions = state_action_counts.cols();
  // std::cout << numStates << " " << numActions << std::endl;
  MatrixXd probs = MatrixXd::Zero(numStates, numActions);

  for (int s = 0; s < numStates; s++) {
    for (int a = 0; a < numActions; a++) {
      if (state_counts(s) > 0)
        probs(s, a) = state_action_counts(s,a) / state_counts(s);
    }
  }
  double action_prob;
  int state, action;
  for (auto & traj : estimate_data) {
    traj.IWs.resize(traj.len);
    traj.cumIWs.resize(traj.len);
    traj.evalActionProbabilities.resize(traj.len);
    for (int t = 0; t < traj.len; t++) {
      state = traj.states[t];
      action = traj.actions[t];
      action_prob = eval_pi.getActionProbability(state, action);
      traj.evalActionProbabilities[t] = action_prob;
      action_prob = probs(state, action);
      // action_prob = state_action_time_counts[t][state][action];
      traj.IWs[t] = traj.evalActionProbabilities[t] / action_prob;
      traj.cumIWs[t] = traj.IWs[t];
      if (t != 0)
        traj.cumIWs[t] *= traj.cumIWs[t-1];
    }
  }
}

void MLE_LoadTrajIWs(vector<Trajectory> &estimate_data, vector<Trajectory> &pib_data,
                     const Policy &eval_pi, const Environment &env,
                     int smoothing) {
  // Get counts for each (s,a) pair
  int k = smoothing;  // Laplace smoothing parameter
  int L = env.getMaxTrajLen();
  int numStates = env.getNumStates();
  int numActions = env.getNumActions();
  VectorXd state_counts = VectorXd::Zero(env.getNumStates());
  MatrixXd probs = MatrixXd::Zero(env.getNumStates(), env.getNumActions());

  initCounts(probs, state_counts, k);


  updateCounts(pib_data, probs, state_counts);


  for (int s = 0; s < numStates; s++) {
    for (int a = 0; a < numActions; a++) {
      if (state_counts(s) > 0)
        probs(s, a) /= state_counts(s);
    }
  }

  double action_prob;
  int state, action;
  for (auto & traj : estimate_data) {
    traj.IWs.resize(traj.len);
    traj.cumIWs.resize(traj.len);
    traj.evalActionProbabilities.resize(traj.len);
    for (int t = 0; t < traj.len; t++) {
      state = traj.states[t];
      action = traj.actions[t];
      action_prob = eval_pi.getActionProbability(state, action);
      traj.evalActionProbabilities[t] = action_prob;
      action_prob = probs(state, action);
      // action_prob = state_action_time_counts[t][state][action];
      traj.IWs[t] = traj.evalActionProbabilities[t] / action_prob;
      traj.cumIWs[t] = traj.IWs[t];
      if (t != 0)
        traj.cumIWs[t] *= traj.cumIWs[t-1];
    }
  }
}

void RISN_LoadTrajIWs(std::vector<Trajectory> &data, const Policy &eval_pi, int n,
                      bool reset) {
  static std::map<std::vector<int>, int> counts;
  static std::map<std::vector<int>, int> state_counts;
  static int start = 0;
  if (reset) {
    counts.clear();
    state_counts.clear();
    start = 0;
  }
  std::map<std::vector<int>, double> probs;
  std::vector<int> state_seg;
  std::vector<int> action_seg;
  std::map<std::vector<int>, int>::iterator it;
  for (int i=start; i < data.size(); i++) {
    state_seg.clear();
    action_seg.clear();
    for (int t=0; t < data[i].len; t++) {
      state_seg.push_back(data[i].states[t]);
      action_seg.push_back(data[i].states[t]);
      action_seg.push_back(data[i].actions[t]);
      it = counts.find(action_seg);
      if (it == counts.end()) {
        state_counts.insert(std::pair<vector<int>, int>(state_seg, 0));
        counts.insert(std::pair<vector<int>, int>(action_seg, 0));
        probs.insert(std::pair<vector<int>, double>(action_seg, 0.0));
      }
      state_counts[state_seg] += 1;
      counts[action_seg] += 1;
      state_seg.push_back(data[i].actions[t]);

      if (action_seg.size() >= 2 * n) {
        state_seg.erase(state_seg.begin(), state_seg.begin() + 2);
        action_seg.erase(action_seg.begin(), action_seg.begin() + 2);
      }
    }
  }
  start = data.size();
  it = state_counts.begin();
  int len = 0;
  for (map<vector<int>, int>::iterator it = counts.begin();
       it != counts.end(); ++it) {
    std::vector<int> new_state_seg(it->first);
    new_state_seg.pop_back();
    probs[it -> first] = static_cast<double>(it -> second) /
      state_counts[new_state_seg];
  }

  double action_prob;
  int state, action;
  for (auto & traj : data) {
    state_seg.clear();
    action_seg.clear();
    traj.IWs.resize(traj.len);
    traj.cumIWs.resize(traj.len);
    traj.evalActionProbabilities.resize(traj.len);
    for (int t = 0; t < traj.len; t++) {
      state_seg.push_back(traj.states[t]);
      action_seg.push_back(traj.states[t]);
      action_seg.push_back(traj.actions[t]);
      state = traj.states[t];
      action = traj.actions[t];
      action_prob = eval_pi.getActionProbability(state, action);
      traj.evalActionProbabilities[t] = action_prob;
      action_prob = probs[action_seg];
      traj.IWs[t] = traj.evalActionProbabilities[t] / action_prob;
      traj.cumIWs[t] = traj.IWs[t];
      if (t != 0)
        traj.cumIWs[t] *= traj.cumIWs[t-1];
      state_seg.push_back(traj.actions[t]);
      if (action_seg.size() >= 2*n) {
        state_seg.erase(state_seg.begin(), state_seg.begin() + 2);
        action_seg.erase(action_seg.begin(), action_seg.begin() + 2);
      }
    }
  }
}


