#ifndef _ZAFFRE_MARSAGLIAGENERATOR_HPP_
#define _ZAFFRE_MARSAGLIAGENERATOR_HPP_

#include <Includes.hpp>

// Fast RNG that returns integers between 0 and max (inclusive).

// Returns number from 0-max
class MarsagliaGenerator {
public:
	MarsagliaGenerator(int RNGSeed, unsigned long max);
	unsigned long sample();
private:
	unsigned long x;
	unsigned long y;
	unsigned long z;

	unsigned long max;
};

#endif
