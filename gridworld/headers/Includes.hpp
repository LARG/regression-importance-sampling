#ifndef _INCLUDES_HPP_
#define _INCLUDES_HPP_

#define _USE_MATH_DEFINES		// The get M_PI and similar constants from cmath

#include <iostream>			// For console I/O
#include <fstream>			// For file I/O
#include <stdio.h>			// For getchar
#include <math.h>			// Basic math functions
#include <iomanip>			// For setprecision
#include <vector>			// For stl vectors
#include <string>			// For stl strings
#include <time.h>			// For getting the time (e.g., and a RNG seed)
#include <sstream>			// Stringstream
#include <stdarg.h>			// For functions with different numbers of inputs
#include <functional>		// For passing functions
#include <random>			// For random numbers
#include <thread>			// For creating multiple threads
#include <unistd.h>			// To get the working directory
#include <iterator>			// Used by things like istream_iterator

// For getting the current directory
//#include <unistd.h>
//#define GetCurrentDir getcwd

#include <Eigen/Dense>			// For matrix math
#include <Eigen/SVD>			// For writing our Moore-Penrose pseudo-inverse implementation
#include <Eigen/Eigenvalues>		// For getting eigenvalues
#include <Eigen/Core>
#include <Eigen/Cholesky>

using namespace std;
using namespace Eigen;

#include <IOStringUtils.hpp>				// Basic file and string I/O functions
#include <MathUtils.hpp>					// Some useful math functions that aren't in other included libraries
#include <MarsagliaGenerator.hpp>


#endif
