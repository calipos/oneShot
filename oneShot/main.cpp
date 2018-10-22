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

unre::FrameRingBuffer<uchar> buffer(400, 300, 3);


std::atomic<bool> threadRunningFlag(true);

void push_data(unre::FrameRingBuffer<uchar>*buffer)
{
	while (threadRunningFlag.load())
	{
		int idx = 0;
		while (!buffer->full() && idx < 50)
		{
			cv::Mat img = cv::Mat::ones(400, 300, CV_8UC3)*idx;
			buffer->push(img.data);
			std::cout << idx++ << " pushed" << std::endl;
			idx %= 200;
		}
	}
}
void pop_data(unre::FrameRingBuffer<uchar>*buffer)
{
	while (threadRunningFlag.load())
	{
		while (!buffer->empty())
		{
			cv::Mat img = cv::Mat::ones(400, 300, CV_8UC3);
			uchar*poppedBufferData = buffer->pop();
			memcpy(img.data, poppedBufferData, 400 * 300 * 3 * sizeof(uchar));
			std::cout << (int)img.data[0] <<"popped" << std::endl;
		}
	}
}


int main()
{
	try {
		unre::threadPool executor{ 6 };
		std::future<void> fg = executor.commit(pop_data, &buffer);
		std::future<void> ff = executor.commit(push_data, &buffer);
		
	}
	catch (std::exception& e) {
		std::cout << "some unhappy happened...  " << std::this_thread::get_id() << e.what() << std::endl;
	}
	
	std::this_thread::sleep_for(std::chrono::microseconds(500));
	threadRunningFlag.store(false);

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
	for (size_t i = 0; i < 100; i++)
	{
		LOG(INFO) << i;
	}
	system("pause");
	return 0;
}