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
extern int TEST_dataExplorer_destroy();
extern void TEST_ogjectDestroy();
extern int TEST_calib();
extern int TEST_tsdf();
extern int TEST_generate_testDepthMat();
extern int TEST_oneDev();
extern int TEST_inverse();
extern int TEST_tennisBall();
struct color
{
public:
	int r, g, b;
	color() {}
private:int a;
};
int main()
{
	



	//TEST_tennisBall();
	TEST_oneDev();
	//TEST_calib();
	//TEST_generate_testDepthMat();
	//TEST_tsdf();
	//TEST_dataExplorer_destroy();
	//TEST_dataExplorer();
	//TEST_calib();
	system("pause");
	return 0;
}