#ifndef _ZAFFRE_MATHUTILS_H_
#define _ZAFFRE_MATHUTILS_H_

#include <Includes.hpp>

bool isReal(const double & x);

double dist(const double & x1, const double & y1, const double & x2, const double & y2);

double distSquared(const double & x1, const double & y1, const double & x2, const double & y2);

int ipow(const int & a, const int & b);

/* Sort an Eigen vector of doubles */
void sort(VectorXd  & v);

/* Sort an Eigen vector  of integers */
void sort(VectorXi  & v);

/* Get the correlation between two vectors (Eigen vectors). */
double corr(const VectorXd & a, const VectorXd & b);

/* Get the covariance between two vectors (Eigen vectors)*/
double cov(const VectorXd & a, const VectorXd & b);

/* Sample variance of vector */
double var(const VectorXd & a);

/* Standard Deviation of a vector */
double stddev(const VectorXd & a);

/* Standard Error of a vector */
double stderror(const VectorXd & a);

/*
Get the sample covariance matrix.
rowWise = true means that m(i,j) = the j'th sample of the i'th random variable.
rowWise = false means that m(i,j) = the i'th sample of the j'th random variable.
*/
MatrixXd cov(const MatrixXd & m, bool rowWise);

/* Get the angle between two vectors in radians*/
double angleBetween(const VectorXd & a, const VectorXd & b);

/* Floating-point modulo:
The result (the remainder) has same sign as the divisor.
Similar to matlab's mod(); Not similar to fmod() -   Mod(-3,4)= 1   fmod(-3,4)= -3
*/
double Mod(const double & x, const double & y);

/* wrap [rad] angle to [-PI..PI) */
double WrapPosNegPI(const double & theta);

/* wrap [rad] angle to [0..TWO_PI) */
double WrapTwoPI(const double & theta);

/* wrap [deg] angle to [-180..180) */
double WrapPosNeg180(const double & theta);

/* wrap [deg] angle to [0..360) */
double Wrap360(const double & theta);

/* Treat counter as a vector containing digits (actually integers). The number represented by the digits is incremented (zeroth digit is incremented first)*/
void incrementCounter(VectorXi & counter, int maxDigit);

/* Return a random number in the range [min, max]. */
double rand(mt19937_64 & generator, const double & min, const double & max);

/* Returns an integer from 0 to num-1, each with the probability specified in probabilities (vector) */
int wrand(mt19937_64 & generator, const VectorXd & probabilities);

/* Returns an integer from 0 to num-1, each with the probability specified in probabilities (vector) */
int wrand(mt19937_64 & generator, const vector<double> & probabilities);

/* Returns a random integer between min and max */
int intRand(mt19937_64 & generator, const int & min, const int & max);

/* Get a random number, such that it's log is uniformly distributed */
double logRand(mt19937_64 & generator, const double & min, const double & max);

/* Thread-safe - Returns a number drawn from the standard normal distribution (multiply by sigma and add mu to get any normal distribution). */
double normRand(mt19937_64 & generator, const double & mu = 0.0, const double & sigma = 1.0);

/* Get a seed for an RNG using an RNG (for seeding new RNGs for each thread) */
unsigned long int getSeed(mt19937_64 & generator);

/* Thread-safe */
bool bernoulli(mt19937_64 & generator, const double & p = 0.5);

/* Sigmoid (logistic) function */
double Sigmoid(const double & x);

/* Sigmoid (logistic function) of a vector */
VectorXf Sigmoid(const VectorXf & v);

/* The CDF of the normal distribution, usually written as \Phi */
double normcdf(double value);

/* Inverse of normalcdf */
double norminv(double value);

/* erf^{-1} = inverse of the erf error function */
double erfinv(double x);

//////////////////////////////////////////////////
///// Templated Functions
//////////////////////////////////////////////////
// Concatenate two STL vectors. Places b at the end of a
template <typename T>
void append(vector<T> & a, const vector<T> & b)
{
	a.insert(a.end(), b.begin(), b.end());
}

/* Bound x between min and max */
template <typename _itemType>
_itemType bound(const _itemType & x, const _itemType & minValue, const _itemType & maxValue) {
	return min(maxValue, max(minValue, x));
}

/* Returns an integer that is +1, 0, or -1 */
template <typename _itemType>
int sign(_itemType x) {
	return (x>0) - (x<0);
}

#endif
