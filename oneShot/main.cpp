#include <atomic>
#include <chrono>
#include"logg.h"
#include"ringBuffer.h"
#include"threadPool.h"
#include"dataExplorer.h"
#include "stringOp.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#include"opencv2/opencv.hpp"



extern int TEST_thread_pauseContinue();
extern int TEST_dataExplorer();
extern int TEST_calib();
int main()
{
	TEST_calib();
	system("pause");
	return 0;
}