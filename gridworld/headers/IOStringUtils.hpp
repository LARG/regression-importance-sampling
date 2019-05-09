#ifndef _ZAFFRE_STRINGUTILS_H_
#define _ZAFFRE_STRINGUTILS_H_

// This file contains I/O utils, and string utils.

#include <Includes.hpp>

//////////////////////////////////////////////////
///// Function prototypes
//////////////////////////////////////////////////

/* Get next line of .csv and split into tokens. */
vector<string> csv_getNextLineAndSplitIntoTokens(istream& str);

/* Eat leading whitespace on a string */
string eatWhite(const string & s, const string & whitespace = " \t");

/* Eat until the end of the line */
void eatLine(ifstream & in);

/* Remove everything from the cin buffer */
void cinFlush();

/* Makes the user actually press enter again, regardless of if they hit it before */
int forceGetchar();

/* Makes the user press enter, then the program exits with the provided code */
void forceGetcharExit(int code = 0);

/* Print the error message to stderr, then exit with error code 1 after the user hits enter */
void errorExit(const string & s);
void errorExit(const char * s);			// Just casts it to a string

/* Get the current working directory. */
string getWorkingDirectory();

//////////////////////////////////////////////////
///// Templated functions
//////////////////////////////////////////////////
/* Convert any printable object into a string. Try std::to_string first, and use this if that doesn't exist*/
template <typename _itemType>
string xtoa(_itemType x)
{
	stringstream out;
	out << x;
	return out.str();
}

#endif
