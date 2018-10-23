#ifdef COMPILE_TEST
#include <atomic>
#include <chrono>
#include"logg.h"
#include"ringBuffer.h"
#include"threadPool.h"
#include"dataExplorer.h"
#include "stringOp.h"
#include"JsonExplorer.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#include"opencv2/opencv.hpp"


int TEST_jsonExplorer()
{
	JsonExplorer je("../../test2.json");
	system("pause");
	return 0;
}
#endif // COMPILE_TEST
