#include"logg.h"

namespace LOGG
{
	
	
		logger::logger(const char*fileName, int lineIdx, int logLevel)
		{
			curLogLevel = logLevel;
			curNevery = -1;
			const time_t t = time(NULL);
			logStreamData.str("");
			logStreamData << LOGLEVELS[logLevel] << ":" << t
				<< "[" << fileName << " : " << lineIdx << "]";
		}
		logger::logger(const char*fileName, int lineIdx, int logLevel, int Nevery)
		{
			curLogLevel = logLevel;
			curNevery = Nevery;
			logLoopCnt++;
			const time_t t = time(NULL);
			logStreamData.str("");
			logStreamData << LOGLEVELS[logLevel] << ":" << t
				<< "[" << fileName << " : " << lineIdx << "]";
		}
		logger::~logger()
		{
			std::lock_guard<std::mutex>lck(logg_mtx);
			const time_t t = time(NULL);
			std::string logFile = LOGDIRPATH + std::to_string(t / 50) + ".txt";
			std::ofstream fout(logFile, std::ios::app);
			if (curNevery<0 && curLogLevel >= 0)
			{
				fout << logStreamData.str() << std::endl;
				std::cout << logStreamData.str() << std::endl;
			}
			if (curNevery>0 && curLogLevel >= 0 && logLoopCnt % 10 == 0)
			{
				fout << logStreamData.str() << std::endl;
				std::cout << logStreamData.str() << std::endl;
			}
			if (curLogLevel >= 3)
			{
				assert(false);
				//system("pause");
			}
			fout.close();
		}

		
}