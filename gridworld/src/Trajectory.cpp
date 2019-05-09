#include "Trajectory.h"

Trajectory::Trajectory (const Trajectory &other) {
	R = other.R;
	len = other.len;
	states.resize(len);
	actions.resize(len);
	rewards.resize(len);
	actionProbabilities.resize(len);
	evalActionProbabilities.resize(other.evalActionProbabilities.size());
	behaviorActionProbabilities.resize(other.behaviorActionProbabilities.size());
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
	
}
