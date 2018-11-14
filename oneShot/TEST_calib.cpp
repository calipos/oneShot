#ifdef COMPILE_TEST
#include <atomic>
#include <chrono>
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
int TEST_calib()
{
	{
		unre::DataExplorer de(3, true);
		de.calibAllStream();
		de.deleteDevice();
	}
	{
		unre::DataExplorer de(3, false);
		de.getBuffer_fortest3();
		de.deleteDevice();
	}

	return 0;
}
#endif