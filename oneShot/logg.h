#ifndef _LOGG_H_
#define _LOGG_H_
#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<mutex>
#include<assert.h>
#include<time.h>
#include "iofile.h"
#include "errorCode.h"



static std::string LOGLEVELS[] = { "INFO", "WARNING", "ERROR", "FATAL", "STACKTRACE" };

const int GLOG_INFO = 0;
const int GLOG_WARNING = 1;
const int GLOG_ERROR = 2;
const int GLOG_FATAL = 3;
const int NUM_SEVERITIES = 4;

const int INFO = GLOG_INFO;
const int WARNING = GLOG_WARNING;
const int ERROR = GLOG_ERROR;
const int FATAL = GLOG_FATAL;

#define LOGDIRPATH "./UnreLogDir/"

class mkLogDir
{
public:
	mkLogDir() 
	{
		mkUnreLogDir();
	};
	~mkLogDir() {};
	int mkUnreLogDir()
	{
		if (!unre::FileOP::FolderExist(LOGDIRPATH))
		{
			unre::FileOP::MkDir(LOGDIRPATH);
		}
		return 0;
	}
};

static mkLogDir ins_mkdir;


namespace LOGG
{
	static int logLoopCnt = 0;
	class logger
	{
	public:
		logger(const char*fileName, int lineIdx, int logLevel);
		logger(const char*fileName, int lineIdx, int logLevel, int Nevery);
		~logger();
		int curLogLevel;
		int curNevery;
		std::stringstream&getStream()
		{
			return logStreamData;
		}
	private:
		std::stringstream logStreamData;
	};
}

#define LOG_FATAL_IF(condition) condition? LOGG::logger("-1", -1, -1).getStream():LOGG::logger(__FILE__, __LINE__, FATAL).getStream()
#define CHECK(condition)  LOG_FATAL_IF(condition)<< "Check failed: "#condition" "
#define CHECK_OP(val1,val2,op) LOG_FATAL_IF(val1 op val2) << "Check failed: " #val1 " " #op " " #val2 " ("  << val1 <<  " vs. "  << val2  << ") "
#define CHECK_EQ(val1,val2) CHECK_OP(val1,val2,==)
#define CHECK_NE(val1,val2) CHECK_OP(val1,val2,!=)
#define CHECK_GT(val1,val2) CHECK_OP(val1,val2,>)
#define CHECK_GE(val1,val2) CHECK_OP(val1,val2,>=)
#define CHECK_LT(val1,val2) CHECK_OP(val1,val2,<)
#define CHECK_LE(val1,val2) CHECK_OP(val1,val2,<=)


#define LOG(severity) LOGG::logger(__FILE__,__LINE__,severity).getStream()
#define DLOG(severity) LOG(severity)
#define LOG_IF(severity,condition) if(condition) LOGG::logger(__FILE__,__LINE__,severity).getStream()
#define LOG_EVERY_N(severity,n) LOGG::logger(__FILE__,__LINE__,severity,n).getStream()

#define CHECK_NOTNULL(ptr)  (ptr == nullptr)?(  LOG(ERROR), nullptr):(ptr);

#ifdef NDEBUG
#define DCHECK(x) \
  while (false) CHECK(x)
#define DCHECK_LT(x, y) \
  while (false) CHECK((x) < (y))
#define DCHECK_GT(x, y) \
  while (false) CHECK((x) > (y))
#define DCHECK_LE(x, y) \
  while (false) CHECK((x) <= (y))
#define DCHECK_GE(x, y) \
  while (false) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) \
  while (false) CHECK((x) == (y))
#define DCHECK_NE(x, y) \
  while (false) CHECK((x) != (y))
#else
#define DCHECK(x) CHECK(x)
#define DCHECK_LT(x, y) CHECK((x) < (y))
#define DCHECK_GT(x, y) CHECK((x) > (y))
#define DCHECK_LE(x, y) CHECK((x) <= (y))
#define DCHECK_GE(x, y) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) CHECK((x) == (y))
#define DCHECK_NE(x, y) CHECK((x) != (y))
#endif  // NDEBUG

#endif // !LOGG

