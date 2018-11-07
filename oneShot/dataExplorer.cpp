#include<algorithm>
#include<chrono>
#include"stringOp.h"
#include"logg.h"
#include"iofile.h"
#include"dataExplorer.h"


#ifdef OPENCV_SHOW
#include "opencv2/opencv.hpp"
#endif

namespace unre
{
	DataExplorer::DataExplorer(int streamNum)
	{
		std::string theInitFile;
		if (FileOP::FileExist("config.json"))
		{
			je = JsonExplorer("config.json");
			theInitFile = "config.json";
		}
		else if (FileOP::FileExist("config_default.json"))
		{
			je = JsonExplorer("config_default.json");
			theInitFile = "config_default.json";
		}
		else
		{
			CHECK(false) << "no config.json, even config_default.jsonㄐㄐㄐ";
		}
		int tmp_exactStreamCnt = 0;
		auto &SensorsInfo = je.getSensorAssignmentInfo();
		std::vector<std::string> usedDeviceType;
		std::find_if(SensorsInfo.begin(), SensorsInfo.end(), [&usedDeviceType, &tmp_exactStreamCnt](auto&item)
		{
			const std::string &this_dev_type = std::get<0>(item);
			if (usedDeviceType.end() == std::find(usedDeviceType.begin(), usedDeviceType.end(), this_dev_type))
			{
				usedDeviceType.emplace_back(this_dev_type);
				LOG(INFO) << this_dev_type << " HAS BEEN SET.";
			}
			tmp_exactStreamCnt += std::get<1>(item).size();
			return false;
		});
		exactStreamCnt = tmp_exactStreamCnt;
		LOG(INFO) << SensorsInfo.size() << " devices are required.";


		dev_e = new DeviceExplorer(theInitFile,usedDeviceType, SensorsInfo, je.getExtraConfigFilPath());
		dev_e->init();
		dev_e->run();
		bufferVecP.resize(exactStreamCnt);//必须相等，因为后边和遍历check和pop
		constBuffer.resize(exactStreamCnt);
		for (auto&item : bufferVecP)  item = Buffer();
		//dev_e->initalConstBuffer(constBuffer);
		dev_e->pushStream(bufferVecP);//before push, the [bufferVecP] has be initalized in this function



	}

	int DataExplorer::getExactStreamCnt()
	{
		return exactStreamCnt;
	}

	int DataExplorer::getBuffer_fortest()
	{
		for (auto&item : bufferVecP)  CHECK(item.data) << "null data is disallowed!";
		int height1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->height;
		int width1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->width;
		int channels1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->channels;
		int height2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->height;
		int width2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->width;
		int channels2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->channels;
		int height3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->height;
		int width3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->width;
		int channels3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->channels;
		int height4 = ((FrameRingBuffer<unsigned char>*)bufferVecP[3].data)->height;
		int width4 = ((FrameRingBuffer<unsigned char>*)bufferVecP[3].data)->width;
		int channels4 = ((FrameRingBuffer<unsigned char>*)bufferVecP[3].data)->channels;
		int height5 = ((FrameRingBuffer<unsigned short>*)bufferVecP[4].data)->height;
		int width5 = ((FrameRingBuffer<unsigned short>*)bufferVecP[4].data)->width;
		int channels5 = ((FrameRingBuffer<unsigned short>*)bufferVecP[4].data)->channels;
		int height6 = ((FrameRingBuffer<unsigned char>*)bufferVecP[5].data)->height;
		int width6 = ((FrameRingBuffer<unsigned char>*)bufferVecP[5].data)->width;
		int channels6 = ((FrameRingBuffer<unsigned char>*)bufferVecP[5].data)->channels;
		int height7 = ((FrameRingBuffer<unsigned char>*)bufferVecP[6].data)->height;
		int width7 = ((FrameRingBuffer<unsigned char>*)bufferVecP[6].data)->width;
		int channels7 = ((FrameRingBuffer<unsigned char>*)bufferVecP[6].data)->channels;
		int height8 = ((FrameRingBuffer<unsigned short>*)bufferVecP[7].data)->height;
		int width8 = ((FrameRingBuffer<unsigned short>*)bufferVecP[7].data)->width;
		int channels8 = ((FrameRingBuffer<unsigned short>*)bufferVecP[7].data)->channels;
		int height9 = ((FrameRingBuffer<unsigned char>*)bufferVecP[8].data)->height;
		int width9 = ((FrameRingBuffer<unsigned char>*)bufferVecP[8].data)->width;
		int channels9 = ((FrameRingBuffer<unsigned char>*)bufferVecP[8].data)->channels;
#ifdef OPENCV_SHOW
		cv::Mat show1 = cv::Mat(height1, width1, channels1 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show2 = cv::Mat(height2, width2, channels2 == 1 ? CV_16UC1 : CV_16UC3);
		cv::Mat show3 = cv::Mat(height3, width3, channels3 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show4 = cv::Mat(height4, width4, channels4 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show5 = cv::Mat(height5, width5, channels5 == 1 ? CV_16UC1 : CV_16UC3);
		cv::Mat show6 = cv::Mat(height6, width6, channels6 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show7 = cv::Mat(height7, width7, channels7 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show8 = cv::Mat(height8, width8, channels8 == 1 ? CV_16UC1 : CV_16UC3);
		cv::Mat show9 = cv::Mat(height9, width9, channels9 == 1 ? CV_8UC1 : CV_8UC3);
#endif
		while (true)
		{
			auto xxx = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->pop(show1.data);
			auto yyy = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->pop(show2.data);
			auto zzz = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->pop(show3.data);
			auto xxx2 = ((FrameRingBuffer<unsigned char>*)bufferVecP[3].data)->pop(show4.data);
			auto yyy2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[4].data)->pop(show5.data);
			auto zzz2 = ((FrameRingBuffer<unsigned char>*)bufferVecP[5].data)->pop(show6.data);
			auto xxx3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[6].data)->pop(show7.data);
			auto yyy3 = ((FrameRingBuffer<unsigned short>*)bufferVecP[7].data)->pop(show8.data);
			auto zzz3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[8].data)->pop(show9.data);
#ifdef OPENCV_SHOW
			cv::imshow("1", show1);
			cv::imshow("2", show2);
			cv::imshow("3", show3);
			cv::imshow("4", show4);
			cv::imshow("5", show5);
			cv::imshow("6", show6);
			cv::imshow("7", show7);
			cv::imshow("8", show8);
			cv::imshow("9", show9);
			int key = cv::waitKey(12);
			if (key == 'a')
			{
				dev_e->pauseThread();
				for (size_t i = 0; i < 10; i++)
				{
					std::this_thread::sleep_for(std::chrono::seconds(1));
					LOG(INFO) << "WAIT 1s";

				}
				dev_e->continueThread();
			}
			else if (key == 'q')
			{
				dev_e->terminateThread();
				cv::destroyAllWindows();
				break;
			}
#endif		
		}
		return 0;
	}

	int DataExplorer::getBuffer_fortest3()
	{
		for (auto&item : bufferVecP)  CHECK(item.data) << "null data is disallowed!";
		int height1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->height;
		int width1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->width;
		int channels1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->channels;
		int height2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->height;
		int width2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->width;
		int channels2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->channels;
		int height3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->height;
		int width3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->width;
		int channels3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->channels;
#ifdef OPENCV_SHOW
		cv::Mat show1 = cv::Mat(height1, width1, channels1 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show2 = cv::Mat(height2, width2, channels2 == 1 ? CV_16UC1 : CV_16UC3);
		cv::Mat show3 = cv::Mat(height3, width3, channels3 == 1 ? CV_8UC1 : CV_8UC3);		
#endif
		while (true)
		{
			auto xxx = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->pop(show1.data);
			auto yyy = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->pop(show2.data);
			auto zzz = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->pop(show3.data);
			
#ifdef OPENCV_SHOW
			cv::imshow("1", show1);
			cv::imshow("2", show2);
			cv::imshow("3", show3);
			
			int key = cv::waitKey(12);
			if (key == 'a')
			{
				dev_e->pauseThread();
				for (size_t i = 0; i < 10; i++)
				{
					std::this_thread::sleep_for(std::chrono::seconds(1));
					LOG(INFO) << "WAIT 1s";

				}
				dev_e->continueThread();
			}
			else if (key == 'q')
			{
				dev_e->terminateThread();
				cv::destroyAllWindows();
				break;
			}
#endif		
		}
		return 0;
	}

	const std::vector<Buffer>&DataExplorer::getBufferVecP()
	{
		return bufferVecP;
	}

	const std::vector<std::tuple<std::string, oneDevMap> >&DataExplorer::getStreamInfo()
	{
		return je.getSensorAssignmentInfo();
	}
}