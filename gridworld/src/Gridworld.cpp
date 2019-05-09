
#include "Gridworld.h"

const int g_GRIDWORLD_SIZE = 4;
const int g_GRIDWORLD_MAX_TRAJLEN = 100;
const int BIG_PENALTY = -10;
const int BIG_REWARD = 100;

Gridworld::Gridworld(bool trueHorizon) {
  this->trueHorizon = trueHorizon;
  this->rareEvent = false;
}

void Gridworld::setRareEvent(bool rare) {rareEvent = rare;}

int Gridworld::getNumActions() const {
  return 4;
}

int Gridworld::getNumStates() const {
  return g_GRIDWORLD_SIZE * g_GRIDWORLD_SIZE - 1;
}

int Gridworld::getMaxTrajLen() const {
  if (trueHorizon) {
    return g_GRIDWORLD_MAX_TRAJLEN;
  } else {
    // Just lie by one, and the model will get partial observability.
    return g_GRIDWORLD_MAX_TRAJLEN + 1;
  }
}

double Gridworld::getMinReturn() {return -550;}
double Gridworld::getMaxReturn() {return 101;}

int Gridworld::getNumEvalTrajectories() {
  return 10000;
}

void Gridworld::generateTrajectories(vector<Trajectory> & buff,
 const Policy & pi, int numTraj, mt19937_64 & generator) {
  buff.resize(numTraj);
  for (int trajCount = 0; trajCount < numTraj; trajCount++) {
    buff[trajCount].len = 0;
    buff[trajCount].actionProbabilities.resize(0);
    buff[trajCount].actions.resize(0);
    buff[trajCount].rewards.resize(0);
    buff[trajCount].states.resize(1);
    buff[trajCount].R = 0;
    int x = 0, y = 0;
    buff[trajCount].states[0] = x + y*g_GRIDWORLD_SIZE;
    for (int t = 0; t < g_GRIDWORLD_MAX_TRAJLEN; t++) {
      int s = x + y*g_GRIDWORLD_SIZE;
      buff[trajCount].len++;  // We have one more transition!
      // Get action
      int action = pi.getAction(buff[trajCount].states[t], generator);
      buff[trajCount].actions.push_back(action);
      double actionProbability = pi.getActionProbability(
        buff[trajCount].states[t], buff[trajCount].actions[t]);
      buff[trajCount].actionProbabilities.push_back(actionProbability);

      // Get next state and reward
      if ((action == 2) && x == 0 && y == 0 && rareEvent) {
        x = g_GRIDWORLD_SIZE - 1; y = g_GRIDWORLD_SIZE - 1;
      } else if ((action == 0) && (x < g_GRIDWORLD_SIZE - 1)) {
        x++;
      } else if ((action == 1) && (x > 0)) {
        x--;
      } else if ((action == 2) && (y < g_GRIDWORLD_SIZE - 1)) {
        y++;
      } else if ((action == 3) && (y > 0)) {
        y--;
      }

      // Update the reward
      double reward;
      if ((x == 1) && (y == 1)) {
        reward = BIG_PENALTY;
      } else if ((x == 1) && (y == g_GRIDWORLD_SIZE - 1)) {
        reward = 1;
      } else if ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1)) {
          reward = BIG_REWARD;
        if (rareEvent && s == 0) reward = 2 * BIG_REWARD;
      } else {
        reward = -1;
      }
      buff[trajCount].rewards.push_back(reward);
      buff[trajCount].R += reward;

      if ((t == g_GRIDWORLD_MAX_TRAJLEN - 1) ||
        ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1))) {
        // Entered a terminal state. Last transition
        break;
      }

      // Add the state and features for the next element
      buff[trajCount].states.push_back(x + y*g_GRIDWORLD_SIZE);
    }
  }
}

double Gridworld::evaluatePolicy(const Policy & pi, mt19937_64 & generator) {
  int numSamples = 10000;

  double result = 0;
  for (int trajCount = 0; trajCount < numSamples; trajCount++) {
    int x = 0, y = 0;
    for (int t = 0; t < g_GRIDWORLD_MAX_TRAJLEN; t++) {
      int action = pi.getAction(x + y*g_GRIDWORLD_SIZE, generator);
      int s = x + y*g_GRIDWORLD_SIZE;
      if ((action == 2) && x == 0 && y == 0 && rareEvent) {
        x = g_GRIDWORLD_SIZE - 1; y = g_GRIDWORLD_SIZE - 1;
      } else if ((action == 0) && (x < g_GRIDWORLD_SIZE - 1)) {
        x++;
      } else if ((action == 1) && (x > 0)) {
        x--;
      } else if ((action == 2) && (y < g_GRIDWORLD_SIZE - 1)) {
        y++;
      } else if ((action == 3) && (y > 0)) {
        y--;
      }
      // Update reward
      if ((x == 1) && (y == 1)) {
        result += BIG_PENALTY;
      } else if ((x == 1) && (y == g_GRIDWORLD_SIZE - 1)) {
        result += 1;
      } else if ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1)) {
        if (rareEvent && s == 0)
          result += 2 * BIG_REWARD;
        else
          result += BIG_REWARD;
      } else {
        result += -1;
      }

      if ((t == g_GRIDWORLD_MAX_TRAJLEN - 1) ||
        ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1))) {
        // Entered a terminal state. Last transition
        break;
      }
    }
  }
  return result / static_cast<double>(numSamples);
}

Policy Gridworld::getPolicy(int index) {
  if (index == 1)
    return Policy("policies/gridworld/p1.txt", getNumActions(), getNumStates());
  if (index == 2)
    return Policy("policies/gridworld/p2.txt", getNumActions(), getNumStates());

  errorExit("Error091834");
  return Policy("error", 0, 0);
}

Model Gridworld::getTrueModel() {
  int numStates = g_GRIDWORLD_SIZE * g_GRIDWORLD_SIZE - 1;
  int numActions = 4;
  int L = g_GRIDWORLD_MAX_TRAJLEN;
  int dim = g_GRIDWORLD_SIZE;
  vector<Trajectory*> trajs;
  Model model(trajs, numStates, numActions, L, false);
  for (int i=0; i < numStates; i++) model.d0[i] = 0;
  model.d0[0] = 1.0;
  for (int s=0; s < numStates; s++) {
    for (int a=0; a < numActions; a++) {
      int x = s % dim; int y = s / dim; double rew = -1;
      if ((a == 2) && x == 0 && y == 0 && rareEvent) {
        x = g_GRIDWORLD_SIZE - 1; y = g_GRIDWORLD_SIZE - 1;
      } else if ((a == 0) && (x < g_GRIDWORLD_SIZE - 1)) {
        x++;
      } else if ((a == 1) && (x > 0)) {
              x--;
      } else if ((a == 2) && (y < g_GRIDWORLD_SIZE - 1)) {
              y++;
      } else if ((a == 3) && (y > 0)) {
              y--;
      }

      int sPrime = x + y * dim;
      if ((x == 1) && (y == 1)) {
        model.R[s][a][sPrime] = BIG_PENALTY;
      } else if ((x == 1) && (y == g_GRIDWORLD_SIZE - 1)) {
        model.R[s][a][sPrime] = 1;
      } else if ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1)) {
        model.R[s][a][sPrime] = BIG_REWARD;
        if (rareEvent && s == 0) model.R[s][a][sPrime] = 2 * BIG_REWARD;
      } else {
        model.R[s][a][sPrime] = -1;
      }
      // Transitions
      for (int j=0; j < numStates; j++)
        model.P[s][a][j] = 0.0;
      model.P[s][a][sPrime] = 1.0;
    }
  }
  return model;
}

double Gridworld::getTrueValue(int policy_number) {
  Model m = getTrueModel();
  Policy policy = getPolicy(policy_number);
  m.loadEvalPolicy(policy, g_GRIDWORLD_MAX_TRAJLEN);
  return m.evalPolicyValue;
}

double Gridworld::getTrueValue(const Policy & pi) {
  Model m = getTrueModel();
  m.loadEvalPolicy(pi, g_GRIDWORLD_MAX_TRAJLEN);
    return m.evalPolicyValue;
}
