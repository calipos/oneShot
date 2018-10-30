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
		for (auto&item : bufferVecP)  item = Buffer();
		dev_e->pushStream(bufferVecP);



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
		int height4 = ((FrameRingBuffer<unsigned short>*)bufferVecP[3].data)->height;
		int width4 = ((FrameRingBuffer<unsigned short>*)bufferVecP[3].data)->width;
		int channels4 = ((FrameRingBuffer<unsigned short>*)bufferVecP[3].data)->channels;
		int height5 = ((FrameRingBuffer<unsigned char>*)bufferVecP[4].data)->height;
		int width5 = ((FrameRingBuffer<unsigned char>*)bufferVecP[4].data)->width;
		int channels5 = ((FrameRingBuffer<unsigned char>*)bufferVecP[4].data)->channels;
#ifdef OPENCV_SHOW
		cv::Mat show1 = cv::Mat(height1, width1, channels1 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show2 = cv::Mat(height2, width2, channels2 == 1 ? CV_16UC1 : CV_16UC3);
		cv::Mat show3 = cv::Mat(height3, width3, channels3 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show4 = cv::Mat(height4, width4, channels4 == 1 ? CV_16UC1 : CV_16UC3);
		cv::Mat show5 = cv::Mat(height5, width5, channels5 == 1 ? CV_8UC1 : CV_8UC3);
#endif
		while (true)
		{
			auto xxx = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->pop();
			auto yyy = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->pop();
			auto zzz = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->pop();
			auto uuu = ((FrameRingBuffer<unsigned short>*)bufferVecP[3].data)->pop();
			auto vvv = ((FrameRingBuffer<unsigned char>*)bufferVecP[4].data)->pop();
#ifdef OPENCV_SHOW
			memcpy(show1.data, xxx, height1*width1*channels1 * sizeof(unsigned char));
			memcpy(show2.data, yyy, height2*width2*channels2 * sizeof(unsigned short));
			memcpy(show3.data, zzz, height3*width3*channels3 * sizeof(unsigned char));
			memcpy(show4.data, uuu, height4*width4*channels4 * sizeof(unsigned short));
			memcpy(show5.data, vvv, height5*width5*channels5 * sizeof(unsigned char));
			cv::imshow("1", show1);
			cv::imshow("2", show2);
			cv::imshow("3", show3);
			cv::imshow("4", show4);
			cv::imshow("5", show5);
			int key = cv::waitKey(12);
			if (key =='a')
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
}