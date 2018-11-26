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

int TEST_tsdf()
{

	{
		unre::DataExplorer de(3, false);
		de.doTsdf();
		de.deleteDevice();
	}

	return 0;
}
#endif