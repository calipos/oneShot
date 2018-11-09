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
extern int TEST_calib2();
extern int TEST_dataExplorer_destroy();
int main()
{
	TEST_dataExplorer_destroy();
	//TEST_dataExplorer();
	//TEST_calib2();
	system("pause");
	return 0;
}