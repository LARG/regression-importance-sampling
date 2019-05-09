#ifndef _GRIDWORLD_H_
#define _GRIDWORLD_H_

#include <Includes.hpp>
#include "Policy.h"
#include "Trajectory.h"
#include "Environment.hpp"

class Gridworld : public Environment {
public:
	Gridworld(bool trueHorizon);
	int getNumActions() const override;
	int getNumStates() const override;
	void generateTrajectories(vector<Trajectory> & buff, const Policy & pi, int numTraj, mt19937_64 & generator) override;
	double evaluatePolicy(const Policy & pi, mt19937_64 & generator) override;	// Estimate expected return of this policy.
	int getMaxTrajLen() const override;
	Policy getPolicy(int index) override;
	int getNumEvalTrajectories() override;
	double getMinReturn() override;
	double getMaxReturn() override;
    Model getTrueModel() override;
    double getTrueValue(int policy_number) override;
    double getTrueValue(const Policy & pi) override;
	void setRareEvent(bool rare);

private:
	bool trueHorizon;
	bool rareEvent;
};

#endif
