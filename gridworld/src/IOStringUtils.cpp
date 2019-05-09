// This file contains I/O utils, and string utils.

#include <Includes.hpp>

/* Get next line of .csv and split into tokens. */
vector<string> csv_getNextLineAndSplitIntoTokens(istream& str)
{
	vector<string> result;
	string line, cell;
	getline(str, line);
	stringstream lineStream(line);
	while (getline(lineStream, cell, ','))
		result.push_back(cell);
	return result;
}

/* Eat leading whitespace on a string */
string eatWhite(const string & s, const string & whitespace)
{
	int strBegin = (int)(s.find_first_not_of(whitespace));
	if (strBegin == (int)string::npos) return ""; // no content
	return s.substr(strBegin, s.length() - strBegin);
}

/* Eat until the end of the line */
void eatLine(ifstream & in)
{
	string s;
	std::getline(in, s);
}

/* Remove everything from the cin buffer */
void cinFlush()
{
	cin.ignore(cin.rdbuf()->in_avail());
}

/* Makes the user actually press enter again, regardless of if they hit it before */
int forceGetchar()
{
	cinFlush();
	return getchar();
}

/* Makes the user press enter, then the program exits with the provided code */
void forceGetcharExit(int code)
{
	cinFlush();
	getchar();
	exit(code);
}

/* Print the error message to stderr, then exit with error code 1 after the user hits enter */
void errorExit(const string & s)
{
	cerr << s << endl;
	forceGetcharExit(1);
}

/* Print the error message to stderr, then exit with error code 1 after the user hits enter */
void errorExit(const char * s)
{
	errorExit((string)s);
}


/* Get the current working directory. */
string getWorkingDirectory()
{
	char cCurrentPath[FILENAME_MAX];
	if (!getcwd(cCurrentPath, sizeof(cCurrentPath)))
		errorExit("Something failed in getWorkingDirectory.");
	return (string)cCurrentPath;
}
