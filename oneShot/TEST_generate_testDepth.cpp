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


int TEST_generate_testDepthMat()
{
	int rows = 720;
	int cols = 1080;
	int rows_hlaf = 0.5*rows;
	int cols_hlaf = 0.5*cols;
	cv::Mat testDepthMat = cv::Mat::zeros(rows, cols, CV_16UC1);
	short maxThis = std::min(SHRT_MAX,1200);
	float scale = maxThis / std::sqrtf(rows_hlaf*rows_hlaf+ cols_hlaf*cols_hlaf);
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			float thisLength = std::sqrtf((i-rows_hlaf)*(i - rows_hlaf) + (j-cols_hlaf)*(j - cols_hlaf));
			short thisDepth = thisLength*scale;
			testDepthMat.ptr<short>(i)[j] = thisDepth;
		}
	}
	cv::FileStorage fs("testDepthMat.xml", cv::FileStorage::WRITE);	
	fs << "testDepthMat" << testDepthMat;
	fs.release();
	return 0;
}

#endif