#include "Policy.h"
#include <fstream>

using namespace std;

Policy::Policy(){}

Policy::Policy(const char * fileName, int numActions, int numStates) {
	this->numActions = numActions;
	this->numStates = numStates;
	theta.resize(numActions * numStates);
	for (int i=0; i < numActions * numStates; i++)
		theta[i] = 0;
	ifstream in(fileName);
	for (int i = 0; i < (int)theta.size(); i++)
		in >> theta[i];
}
	
Policy::Policy( const Policy & policy ) {
	this->numActions = policy.getNumActions();
	this->numStates = policy.getNumStates();
	theta.resize(numActions*numStates);
	VectorXd other_theta = policy.getParameters();
	for (int i=0; i <theta.size(); i++)
		theta[i] = other_theta[i];
}

int Policy::getAction(const int & state, mt19937_64 & generator) const {
	return wrand(generator, getActionProbabilities(state));
}

double Policy::getActionProbability(const int & state, const int & action) const {
	return getActionProbabilities(state)(action);
}

VectorXd Policy::getActionProbabilities(const int & state) const {
	VectorXd actionProbabilities(numActions);
	for (int a = 0; a < numActions; a++)
		actionProbabilities[a] = theta[a*numStates + state];
	actionProbabilities.array() = actionProbabilities.array().exp();
	return actionProbabilities / actionProbabilities.sum();
}


VectorXd Policy::getPolicyDerivative() {

	VectorXd derivatives(numActions * numStates);
	
	for (int s=0; s<numStates; s++) {
		VectorXd actionDerivatives(numActions);
		for (int a=0; a < numActions; a++)
			actionDerivatives[a] = theta[a*numStates + s];
		actionDerivatives.array() = actionDerivatives.array().exp();
		double sum = actionDerivatives.sum();
		actionDerivatives.array() = 1 - actionDerivatives.array() / sum;
		for (int a=0; a<numActions; a++)
			derivatives[a*numStates + s] = actionDerivatives[a];
	}
	return derivatives;	

}

VectorXd Policy::getLogDerivative(int state, int action) {

	VectorXd log_derivatives = VectorXd::Zero(numActions * numStates);
	VectorXd actionProbs = getActionProbabilities(state);
	for (int a=0; a<numActions; a++)
		log_derivatives(a*numStates + state) = -1 * actionProbs(a);
	log_derivatives(action*numStates + state) += 1;
	return log_derivatives;

}

VectorXd Policy::getDerivative(int state, int action) {

	VectorXd derivatives = VectorXd::Zero(numActions * numStates);
	VectorXd actionProbs = getActionProbabilities(state);

	for (int a=0; a<numActions; a++) {
		if (a == action)
			derivatives(a*numStates + state) = actionProbs(a) * (1 - actionProbs(a));
		else 
			derivatives(a*numStates + state) = -1 * actionProbs(action) * actionProbs(a);
	}

	return derivatives;

}
	
void Policy::setRareProbability(double prob) {
	if (numActions < 4) return;
	if (prob < 0 or prob > 1) return;
        if (prob == 1.0) prob = 0.999999;
	int state = 0;
	int action = 3;
	double sum = 0.0;
	VectorXd actionProbs = getActionProbabilities(state); // softmax probs
	actionProbs *= actionProbs.sum(); // unnormalized softmax output
	for (int a = 0; a < numActions; a++)
		if (a != action) sum += actionProbs(a);
	double param = log( (prob / (1 - prob)) * sum);
	theta[action*numStates + state] = param;

}
/*VectorXd Policy::getPolicyDerivative(){

	VectorXd derivatives(numActions*numStates);
	for (int s=0; s<numStates; s++) {
		VectorXd actionDerivatives(numActions);
		for (int a=0; a < numActions; a++)
			actionDerivatives[a] = theta[a*numStates + s];
		actionDerivatives.array() = actionDerivatives.array().exp();
		double sum = actionDerivatives.sum();
		actionDerivatives.array() = (actionDerivatives.array() * sum + actionDerivatives.array() * actionDerivatives.array()) / (sum * sum);
		for (int a=0; a<numActions; a++)
			derivatives[a*numStates + s] = actionDerivatives[a];
	}
	return derivatives;	
}*/

VectorXd Policy::getParameters() const {return theta;}
void Policy::setParameters(VectorXd parameters) {theta = parameters;}

void Policy::savePolicy(const char * filename) {
	ofstream out(filename);
	for (int i = 0; i < (int)theta.size(); i++)
		out << theta[i] << endl;
}
