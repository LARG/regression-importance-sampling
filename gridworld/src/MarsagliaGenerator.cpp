#include <Includes.hpp>

MarsagliaGenerator::MarsagliaGenerator(int RNGSeed, unsigned long max) {
	this->max = max;
	mt19937_64 generator(RNGSeed);
	uniform_int_distribution<int> distribution(INT_MIN, INT_MAX);
	x = distribution(generator);
	y = distribution(generator);
	z = distribution(generator);
}

unsigned long MarsagliaGenerator::sample() {          //period 2^96-1
	unsigned long t;
	x ^= x << 16;
	x ^= x >> 5;
	x ^= x << 1;
	t = x;
	x = y;
	y = z;
	z = t ^ x ^ y;
	return z % (max + 1);
}
