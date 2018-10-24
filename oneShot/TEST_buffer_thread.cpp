#ifdef COMPILE_TEST
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
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

std::mutex mtx_pauseThread;
std::condition_variable cv_pauseThread;
std::atomic<bool> pauseThreadFlag(false);

std::atomic<bool> runThreadFlag(true);


void push_data(unre::FrameRingBuffer<uchar>*buffer)
{
	while (runThreadFlag)
	{
		while (pauseThreadFlag)
		{
			//std::this_thread::yield();
			std::unique_lock<std::mutex>(mtx_pauseThread);
			cv_pauseThread.wait(mtx_pauseThread);
			break;
		}
		
		int idx = 0;
		while (!buffer->full())
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
	while (runThreadFlag)
	{
		
		while(pauseThreadFlag)
		{
			//std::this_thread::yield();
			std::unique_lock<std::mutex>(mtx_pauseThread);
			cv_pauseThread.wait(mtx_pauseThread);
			break;
		}
		while (!buffer->empty())
		{
			cv::Mat img = cv::Mat::ones(400, 300, CV_8UC3);
			uchar*poppedBufferData = buffer->pop();
			memcpy(img.data, poppedBufferData, 400 * 300 * 3 * sizeof(uchar));
			std::cout << (int)img.data[0] << "popped" << std::endl;
		}
	}
}


int TEST_buffer_thread()
{
	try {
		//unre::threadPool executor( 6 );
		//std::future<void> fg = executor.commit(pop_data, &buffer);
		//std::future<void> ff = executor.commit(push_data, &buffer);
		std::thread t1(pop_data, &buffer);
		std::thread t2(push_data, &buffer);
		std::this_thread::sleep_for(std::chrono::microseconds(100));
		std::this_thread::sleep_for(std::chrono::seconds(2));
		pauseThreadFlag = true;
		LOG(INFO) << "PAUSE";
		
		std::this_thread::sleep_for(std::chrono::seconds(2));
		LOG(INFO) << "CONTINUE";
		pauseThreadFlag = false;
		std::unique_lock<std::mutex>(mtx_pauseThread);
		cv_pauseThread.notify_all();
		std::this_thread::sleep_for(std::chrono::seconds(2));
		//cv_pauseThread.notify_all();
		//std::this_thread::sleep_for(std::chrono::seconds(5));
		runThreadFlag=(false);
		std::this_thread::sleep_for(std::chrono::seconds(2));
		runThreadFlag=(false);
	}
	catch (std::exception& e) {
		std::cout << "some unhappy happened...  " << std::this_thread::get_id() << e.what() << std::endl;
	}





	system("pause");
	return 0;
}
#endif // COMPILE_TEST
