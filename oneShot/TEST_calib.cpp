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
int test_opencv_pnp()
{
	std::vector<cv::Point3f> gtPoints;
	cv::Size chessBoardSize(6, 9);
	cv::Size2f unitSize(27.,27.);
	for (size_t i = 0; i < chessBoardSize.height; i++)
	{
		for (size_t j = 0; j < chessBoardSize.width; j++)
		{
			gtPoints.emplace_back(cv::Point3f(unitSize.width*j, unitSize.height*i,0.));
		}
	}
	cv::Mat img = cv::imread("../../captured.jpg");
	cv::Mat imageGray;
	cv::cvtColor(img, imageGray,CV_BGR2GRAY);
	std::vector<cv::Point2f> srcCandidateCorners;
	cv::waitKey(1);
	bool patternfound = cv::findChessboardCorners(imageGray, cv::Size(6, 9), srcCandidateCorners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
	if (patternfound)
	{
		cv::cornerSubPix(imageGray, srcCandidateCorners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cv::Mat intr = cv::Mat::zeros(3,3,CV_64FC1);
		intr.ptr<double>(0)[0] = 900.0;
		intr.ptr<double>(1)[1] = 900.0;
		intr.ptr<double>(2)[2] = 1.0;
		intr.ptr<double>(0)[2] = 960.0;
		intr.ptr<double>(1)[2] = 540.0;

		cv::Rect bbox = cv::boundingRect(srcCandidateCorners);
		intr.ptr<double>(0)[2] = bbox.x+ bbox.width*0.5;
		intr.ptr<double>(1)[2] = bbox.y + bbox.height*0.5;

		cv::Mat Rvect;
		cv::Mat t;
		cv::solvePnP(gtPoints, srcCandidateCorners, intr, cv::Mat::zeros(1, 5, CV_64FC1), Rvect, t);
		cv::Mat Rmetrix;
		cv::Rodrigues(Rvect, Rmetrix);
		cv::drawChessboardCorners(img, cv::Size(6, 9), srcCandidateCorners, true);
		double err = 0;
		for (int i = 0; i < gtPoints.size(); i++)
		{
			cv::Mat testP = cv::Mat::zeros(3, 1, CV_64FC1);
			testP.ptr<double>(0)[0] = gtPoints[i].x;
			testP.ptr<double>(1)[0] = gtPoints[i].y;
			testP.ptr<double>(2)[0] = gtPoints[i].z;
			cv::Mat camP = (Rmetrix*testP + t);
			double scale1 = camP.ptr<double>(2)[0];
			cv::Mat y = intr*camP;
			y = y / scale1;
			cv::circle(img,cv::Point(y.ptr<double>(0)[0], y.ptr<double>(1)[0]),3,cv::Scalar(0,0,255),-1);
			err = err + (srcCandidateCorners[i].x - y.ptr<double>(0)[0])*(srcCandidateCorners[i].x - y.ptr<double>(0)[0])
				+ (srcCandidateCorners[i].y - y.ptr<double>(1)[0])*(srcCandidateCorners[i].y - y.ptr<double>(1)[0]);
		}
		LOG(INFO) << err / gtPoints.size();
		
	}
	return 0;
}

int TEST_calib()
{
	//test_opencv_pnp();
	//return 0;
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