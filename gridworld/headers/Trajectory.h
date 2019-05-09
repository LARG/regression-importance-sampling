#ifndef _TRAJECTORY_H_
#define _TRAJECTORY_H_

#include <Includes.hpp>

struct Trajectory {
	// These variables should not change after Trajectory generation
	int len;
	vector<int> states;
	vector<int> actions;
	vector<double> rewards;
	vector<double> actionProbabilities; 
	double R;
	// Variables after here can change for different off-policy evaluation methods

	// These are loaded for a specific evaluation policy
	VectorXd IWs;	// Will be loaded with per-time-step importance weights. [0] = pi_e(first action) / pi_b(first action)
	VectorXd cumIWs;	// Cumulative importance weights - loaded with product of importance weights up to current time. [0] = pi_e(first action) / pi_b(first action)
	vector<double> evalActionProbabilities;
	vector<double> behaviorActionProbabilities;
	Trajectory(){}
	Trajectory( const Trajectory &other);
	/*Trajectory (const Trajectory &other) {
		R = other.R;
		len = other.len;
		states.resize(len);
		actions.resize(len);
		rewards.resize(len);
		actionProbabilities.resize(len);
		evalActionProbabilities.resize(evalActionProbabilities.size());
		behaviorActionProbabilities.resize(behaviorActionProbabilities.size());
		IWs.resize(other.IWs.size());
		cumIWs.resize(other.cumIWs.size());
		for (int t=0; t < len; t++) {
			states[t] = other.states[t];
			actions[t] = other.actions[t];
			rewards[t] = other.rewards[t];
			actionProbabilities[t] = other.actionProbabilities[t];
		}
		for (int t=0; t < other.evalActionProbabilities.size(); t++)
			evalActionProbabilities[t] = other.evalActionProbabilities[t];
		for (int t=0; t < other.behaviorActionProbabilities.size(); t++)
			behaviorActionProbabilities[t] = other.behaviorActionProbabilities[t];
		for (int t=0; t < other.IWs.size(); t++)
			IWs[t] = other.IWs[t];
		for (int t=0; t < other.cumIWs.size(); t++)
			cumIWs[t] = other.cumIWs[t];
		
	}*/
};

#endif
