#include <Includes.hpp>

bool isReal(const double & x)
{
	return ((!std::isnan(x)) && (!std::isinf(x)));
}

double dist(const double & x1, const double & y1, const double & x2, const double & y2)
{
	return sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

double distSquared(const double & x1, const double & y1, const double & x2, const double & y2)
{
	return (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1);
}

int wrand(mt19937_64 & generator, const VectorXd & probabilities)
{
	double sum = 0;
	double r = rand(generator, 0, 1);	// A random real between 0 and 1
	for (int i = 0; i < (int)probabilities.size(); i++) {
		sum += (double)probabilities(i);
		if (sum >= r) return i;
	}
	return intRand(generator, 0, (int)probabilities.size() - 1); // If we get here, there was a rounding error... doh. Pick a random action
}

int wrand(mt19937_64 & generator, const vector<double> & probabilities)
{
	double sum = 0;
	double r = rand(generator, 0, 1);	// A random real between 0 and 1
	for (int i = 0; i < (int)probabilities.size(); i++) {
		sum += (double)probabilities[i];
		if (sum >= r) return i;
	}
	return intRand(generator, 0, (int)probabilities.size() - 1); // If we get here, there was a rounding error... doh. Pick a random action
}

/* Raise a to the power b, where b is an integer */
int ipow(const int & a, const int & b)
{
	if (b == 0) return 1;
	if (b == 1) return a;

	int tmp = ipow(a, b / 2);
	if (b % 2 == 0) return tmp * tmp;
	else return a * tmp * tmp;
}

/* Sort an Eigen vector (not for matrices) */
void sort(VectorXd  & v)
{
	std::sort(v.derived().data(), v.derived().data() + v.derived().size());
}

/* Sort an Eigen vector (not for matrices) */
void sort(VectorXi  & v)
{
	std::sort(v.derived().data(), v.derived().data() + v.derived().size());
}

/* Get the correlation between two vectors (Eigen vectors). */
double corr(const VectorXd & a, const VectorXd & b)
{
	assert(a.size() == b.size());
	double num = 0, den1 = 0, den2 = 0;
	double aMean = a.mean(), bMean = b.mean();
	for (int i = 0; i < a.size(); i++)
	{
		num += (a[i] - aMean)*(double)(b[i] - bMean);
		den1 += (a[i] - aMean)*(double)(a[i] - aMean);
		den2 += (b[i] - bMean)*(double)(b[i] - bMean);
	}
	if (den1*den2 == 0)
		return 1;	// The correlation is actually not defined in this case!
	return num / (double)sqrt(den1*den2);
}

/* Get the covariance between two vectors (Eigen vectors)*/
double cov(const VectorXd & a, const VectorXd & b)
{
	assert(a.size() == b.size());
	if ((int)a.size() <= 1)
		return 0;
	double muA = a.mean(), muB = b.mean();
	double result = 0;
	for (int i = 0; i < a.size(); i++)
		result += (a[i] - muA)*(b[i] - muB);
	result /= (double)(a.size() - 1);
	return result;
}

/* Sample variance of vector */
double var(const VectorXd & a)
{
	double mu = a.mean();
	double result = 0;
	for (int i = 0; i < (int)a.size(); i++)
		result += (a[i] - mu)*(a[i] - mu);
	return result / (double)(a.size() - 1); // We use Bessel's correction to get an unbiased estimate of the sample variance
}

/* Standard Deviation of a vector */
double stddev(const VectorXd & a)
{
	return sqrt(var(a));
}

/* Standard Error of a vector */
double stderror(const VectorXd & a)
{
	return sqrt(var(a)) / sqrt(a.size());
}

/*
Get the sample covariance matrix.
rowWise = true means that m(i,j) = the j'th sample of the i'th random variable.
rowWise = false means that m(i,j) = the i'th sample of the j'th random variable.
*/
MatrixXd cov(const MatrixXd & m, bool rowWise)
{
	if (rowWise) {
		MatrixXd result(m.rows(), m.rows());
		// Compute half of the matrix (it's symmetric, so we can deduce the other half)
		for (int i = 0; i < m.rows(); i++)
		{
			for (int j = 0; j < i; j++)
				result(i, j) = cov(m.row(i), m.row(j));
		}
		// Compute the diagonal
		for (int i = 0; i < m.rows(); i++)
			result(i, i) = var(m.row(i));
		// Deduce the other half
		for (int i = 0; i < m.rows(); i++)
		{
			for (int j = i + 1; j < m.rows(); j++)
				result(i, j) = result(j, i);
		}
		return result;
	}
	else {
		MatrixXd result(m.cols(), m.cols());
		// Compute half of the matrix (it's symmetric, so we can deduce the other half)
		for (int i = 0; i < m.cols(); i++)
		{
			for (int j = 0; j <= i; j++) {
				result(i, j) = cov(m.col(i), m.col(j));
			}
		}
		// Deduce the other half
		for (int i = 0; i < m.cols(); i++)
		{
			for (int j = i + 1; j < m.cols(); j++)
				result(i, j) = result(j, i);
		}
		return result;
	}
}

/* Get the angle between two vectors.*/
double angleBetween(const VectorXd & a, const VectorXd & b)
{
	assert(a.size() == b.size());
	return (double)acos(a.dot(b) / (a.norm()*b.norm()));
}

/* Floating-point modulo:
The result (the remainder) has same sign as the divisor.
Similar to matlab's mod(); Not similar to fmod() -   Mod(-3,4)= 1   fmod(-3,4)= -3
*/
double Mod(const double & x, const double & y)
{
	if (0. == y) return x;
	double m = x - y * floor(x / y);
	// handle boundary cases resulted from floating-point cut off:
	if (y > 0) {				// modulo range: [0..y)
		if (m >= y)		// Mod(-1e-16 , 360.): m=360.
			return 0;
		if (m < 0){
			if (y + m == y) return 0;			// just in case...
			else return (y + m);	// Mod(106.81415022205296 , _TWO_PI ): m= -1.421e-14
		}
	}
	else									// modulo range: (y..0]
	{
		if (m <= y) return 0;				// Mod(1e-16, -360.): m=-360.
		if (m > 0) {
			if (y + m == y) return 0;		// Just in case...
			else return (y + m);	// Mod(-106.81415022205296, -_TWO_PI): m= 1.421e-14
		}
	}
	return m;
}

/* wrap [rad] angle to [-PI..PI) */
double WrapPosNegPI(const double & theta)
{
	return Mod(theta + M_PI, 2.0*M_PI) - M_PI;
}

/* wrap [rad] angle to [0..TWO_PI) */
double WrapTwoPI(const double & theta)
{
	return Mod(theta, 2.0*M_PI);
}

/* wrap [deg] angle to [-180..180) */
double WrapPosNeg180(const double & theta)
{
	return Mod(theta + 180.0, 360.0) - 180.0;
}

/* wrap [deg] angle to [0..360) */
double Wrap360(const double & theta)
{
	return Mod(theta, 360.0);
}

/* Treat counter as a vector containing digits (actually integers). The number represented by the digits is incremented (zeroth digit is incremented first)*/
void incrementCounter(VectorXi & counter, int maxDigit)
{
	for (int i = 0; i < counter.rows(); i++) {
		counter(i, 0)++;
		if (counter(i, 0) > maxDigit)
			counter(i, 0) = 0;
		else
			break;
	}
}


double rand(mt19937_64 & generator, const double & min, const double & max)
{
	uniform_real_distribution<double> distribution(min, max);
	return distribution(generator);
}

int intRand(mt19937_64 & generator, const int & min, const int & max)
{
	uniform_int_distribution<int> distribution(min, max);
	return distribution(generator);
}

double logRand(mt19937_64 & generator, const double & min, const double & max)
{
	double logMin = log(min);
	double logMax = log(max);
	double r = rand(generator, logMin, logMax);
	return exp(r);
}

double normRand(mt19937_64 & generator, const double & mu, const double & sigma)
{
	normal_distribution<double> distribution(mu, sigma);
	return distribution(generator);
}

unsigned long int getSeed(mt19937_64 & generator)
{
	uniform_int_distribution<unsigned long> distribution(std::numeric_limits<unsigned long>::lowest(), std::numeric_limits<unsigned long>::max());
	return distribution(generator);
}

bool bernoulli(mt19937_64 & generator, const double & p)
{
	bernoulli_distribution distribution(p);
	return distribution(generator);
}

/* Sigmoid (logistic) function */
double Sigmoid(const double & x)
{
	return 1.0f / (1.0f + (double)exp(-x));
}

// Sigmoid (logistic function) of a vector
VectorXf Sigmoid(const VectorXf & v)
{
	return 1.0f / (1.0f + (-v).array().exp());
}

/* The CDF of the normal distribution, usually written as \Phi */
double normcdf(double value)
{
	return 0.5 * erfc(-value * M_SQRT1_2);
}

double erfinv(double x)
{
	double x2, r, y;
	int  sign_x;
	if (x < -1 || x > 1)
		return NAN;
	if (x == 0)
		return 0;
	if (x > 0)
		sign_x = 1;
	else
	{
		sign_x = -1;
		x = -x;
	}
	if (x <= 0.7)
	{
		x2 = x * x;
		r =
			x * (((-0.140543331 * x2 + 0.914624893) * x2 + -1.645349621) * x2 + 0.886226899);
		r /= (((0.012229801 * x2 + -0.329097515) * x2 + 1.442710462) * x2 +
			-2.118377725) * x2 + 1.0;
	}
	else
	{
		y = sqrt(-log((1 - x) / 2));
		r = (((1.641345311 * y + 3.429567803) * y + -1.62490649) * y + -1.970840454);
		r /= ((1.637067800 * y + 3.543889200) * y + 1.0);
	}
	r = r * sign_x;
	x = x * sign_x;
	r -= (erf(r) - x) / (2 / sqrt(M_PI) * exp(-r * r));
	r -= (erf(r) - x) / (2 / sqrt(M_PI) * exp(-r * r));
	return r;
}

double norminv(double value)
{
	assert(value > 0);
	assert(value < 1);
	return M_SQRT2*erfinv(2.0*value - 1.0);
}
