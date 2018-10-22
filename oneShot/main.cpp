#include"logg.h"
#include"ringBuffer.h"
#include"dataExplorer.h"
#include "stringOp.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"


#include"opencv2/opencv.hpp"

int main()
{

	unre::FrameRingBuffer<uchar> buffer(400,300,3);
	for (size_t i = 0; i < 5; i++)
	{
		cv::Mat img = cv::Mat::ones(400, 300, CV_8UC3)*i;
		buffer.push(img.data);
	}
	cv::Mat poped = cv::Mat::ones(400, 300, CV_8UC3);
	poped.data = buffer.pop();
	cv::Mat img = cv::Mat::ones(400, 300, CV_8UC3)*6;
	buffer.push(img.data);
	while (!buffer.empty())
	{
		poped.data = buffer.pop();
	}
	
	
	rapidjson::Document docRoot;
	docRoot.Parse<0>(unre::StringOP::parseJsonFile2str("E:/repo/oneShot/test.json").c_str());
	if (!docRoot.HasParseError())
	{
		LOG(INFO) << "parse json start";
		rapidjson::Value &imageRoot = docRoot["test_int"];
		rapidjson::Value &annoRoot = docRoot["test_object"]["vec_key"];
		LOG(INFO) << imageRoot.GetInt();
		LOG(INFO) << annoRoot.Size();
		LOG(INFO) << "parse json end";
	}

	unre::DataExplorer<float> ed(6);
	unre::FrameRingBuffer<int>  x(10,10,3);
	for (size_t i = 0; i < 100000; i++)
	{
		LOG(INFO) << i;
	}
	system("pause");
	return 0;
}