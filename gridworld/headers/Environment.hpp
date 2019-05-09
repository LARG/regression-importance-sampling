#ifndef _ENVIRONMENT_HPP_
#define _ENVIRONMENT_HPP_

#include <Includes.hpp>
#include "Policy.h"
#include "Trajectory.h"
#include "Model.hpp"

class Environment {
public:
	virtual int getNumActions() const = 0;
	virtual int getNumStates() const = 0;
	virtual int getMaxTrajLen() const = 0;
	virtual void generateTrajectories(vector<Trajectory> & buff, const Policy & pi, int numTraj, mt19937_64 & generator) = 0;
	virtual double evaluatePolicy(const Policy & pi, mt19937_64 & generator) = 0;
	virtual Policy getPolicy(int index) = 0;
	virtual int getNumEvalTrajectories() = 0;
	virtual double getMinReturn() = 0;
	virtual double getMaxReturn() = 0;
    virtual Model getTrueModel() = 0;
    virtual double getTrueValue(int policy_number) = 0;
    virtual double getTrueValue(const Policy & pi) = 0;
};

#endif
