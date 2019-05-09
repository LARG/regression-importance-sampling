#ifndef _POLICY_H_
#define _POLICY_H_

#include <Includes.hpp>

class Policy {
public:
	Policy();
	Policy(const char * fileName, int numActions, int numStates);
	Policy( const Policy & policy );
	int getAction(const int & state, mt19937_64 & generator) const;
	double getActionProbability(const int & state, const int & action) const;
	VectorXd getActionProbabilities(const int & state) const;
	VectorXd getPolicyDerivative();
	VectorXd getLogDerivative(int state, int action);
	VectorXd getDerivative(int state, int action);
	VectorXd getParameters() const;
	int getNumStates() const {return numStates;}
	int getNumActions() const {return numActions;}
	void setParameters(VectorXd parameters);
	void savePolicy(const char * filename);
	void setRareProbability(double prob);

private:
	int numActions;
	int numStates;
	VectorXd theta;
};

#endif
