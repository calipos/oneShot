#ifdef COMPILE_TEST
#include <atomic>
#include <chrono>
#include <vector>
#include"logg.h"
#include"ringBuffer.h"
#include"threadPool.h"
#include"dataExplorer.h"
#include "stringOp.h"
#include"jsonExplorer.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#include"opencv2/opencv.hpp"
#include "opencvAssistant.h"


int TEST_oneDev()
{
	{
		unre::DataExplorer de(3, false);
		de.oneDevShow();
	}
	return 0;
}

#endif