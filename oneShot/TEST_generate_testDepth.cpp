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
#include"unreGpu.h"

#include"opencv2/opencv.hpp"
void generate_calibImg(int rows,int cols,int heihgt_cnt,int width_cnt,double unitPixel)
{
	//定标的时候生成的gt坐标不会以0开始，我会把这些gt坐标的重心移动到体素s的中心！！！
	CHECK(unitPixel*(heihgt_cnt + 1)<rows*0.5);
	CHECK(unitPixel*(width_cnt + 1)<cols*0.5);	
	cv::Mat img = cv::Mat::ones(rows, cols, CV_8UC1)*255;
	std::vector<cv::Point> leftUpPoints;
	int theLeftStart = (cols- unitPixel*(width_cnt + 1))*0.5+1;
	int theUpStart = (rows+ unitPixel*(heihgt_cnt - 1))*0.5 ;
	int theRight = 0;
	int theBottom = 0;
	for (size_t i = 0; i < heihgt_cnt+1; i++)
	{
		for (size_t j = 0; j < width_cnt + 1; j++)
		{
			if ((i+j)%2-1)
			{
				leftUpPoints.push_back(cv::Point(theLeftStart + j*unitPixel, theUpStart - i*unitPixel));
			}			
		}
	}
	cv::Mat blackBlock = cv::Mat::zeros(unitPixel, unitPixel,CV_8UC1);
	for (size_t i = 0; i < leftUpPoints.size(); i++)
	{
		blackBlock.copyTo(img(cv::Rect(leftUpPoints[i],cv::Size(unitPixel, unitPixel))));
	}
	cv::imwrite("captured.jpg", img);
}

double figureChessBorad()
{
	cv::Mat intr = cv::Mat::zeros(3, 3, CV_64FC1);
	intr.ptr<double>(0)[0] = 300.;
	intr.ptr<double>(1)[1] = 300.;
	intr.ptr<double>(0)[2] = 540.;
	intr.ptr<double>(1)[2] = 360.;
	intr.ptr<double>(2)[2] = 1.;
	cv::Mat gt_points = cv::Mat(3,2,CV_64FC1);
	gt_points.ptr<double>(0)[0] = 0.;
	gt_points.ptr<double>(1)[0] = 0.;
	gt_points.ptr<double>(2)[0] = VOLUME_SIZE_Z;
	gt_points.ptr<double>(0)[1] = 0.;
	gt_points.ptr<double>(1)[1] = 27.;
	gt_points.ptr<double>(2)[1] = VOLUME_SIZE_Z;

	cv::Mat result = intr*gt_points;
	result /= VOLUME_SIZE_Z;
	return result.ptr<double>(1)[1] - result.ptr<double>(1)[0];
}

int TEST_generate_testDepthMat()
{
	//generate_calibImg(720, 1080, 9, 6, 15);
	//int diff =  figureChessBorad();
	int rows = 720;
	int cols = 1080;
	int rows_hlaf = 0.5*rows;
	int cols_hlaf = 0.5*cols;
	cv::Mat testDepthMat = cv::Mat::zeros(rows, cols, CV_16UC1);
	short minThis = 570;
	short maxThis = std::min(SHRT_MAX,1200);
	float scale = (maxThis- minThis) / std::sqrtf(rows_hlaf*rows_hlaf+ cols_hlaf*cols_hlaf);
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			float thisLength = std::sqrtf((i-rows_hlaf)*(i - rows_hlaf) + (j-cols_hlaf)*(j - cols_hlaf));
			short thisDepth = minThis+thisLength*scale;
			testDepthMat.ptr<short>(i)[j] = thisDepth;
		}
	}
	cv::FileStorage fs("testDepthMat.xml", cv::FileStorage::WRITE);	
	fs << "testDepthMat" << testDepthMat;
	fs.release();
	return 0;
}

#endif