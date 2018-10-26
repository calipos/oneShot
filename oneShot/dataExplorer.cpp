#include<algorithm>
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
		if (FileOP::FileExist("config.json"))
		{
			je = JsonExplorer("config.json");
		}
		else if (FileOP::FileExist("config_default.json"))
		{
			je = JsonExplorer("config_default.json");
		}
		else
		{
			CHECK(false) << "no config.json, even config_default.json£¡£¡£¡";
		}
		int exactStreamCnt = 0;
		auto &SensorsInfo = je.getSensorAssignmentInfo();
		std::vector<std::string> usedDeviceType;
		std::find_if( SensorsInfo.begin(), SensorsInfo.end(), [&usedDeviceType,&exactStreamCnt](auto&item)
		{
			const std::string &this_dev_type = std::get<0>(item);
			if (usedDeviceType.end() == std::find(usedDeviceType.begin(), usedDeviceType.end(), this_dev_type))
			{
				usedDeviceType.emplace_back(this_dev_type);
				LOG(INFO) << this_dev_type << " HAS BEEN SET.";
			}
			exactStreamCnt += std::get<1>(item).size();
			return false;
		});
		LOG(INFO) << SensorsInfo.size() << " devices are required.";


		dev_e = new DeviceExplorer(usedDeviceType, SensorsInfo, je.getExtraConfigFilPath());
		dev_e->init();
		dev_e->run();
		bufferVecP.resize(exactStreamCnt);
		dev_e->pushStream(bufferVecP);	



	}
	int DataExplorer::getBuffer()
	{
		LOG(INFO) << bufferVecP.size();
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
			auto xxx = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->pop();
			auto yyy = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->pop();
			auto zzz = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->pop();
#ifdef OPENCV_SHOW
			memcpy(show1.data, xxx, height1*width1*channels1 * sizeof(unsigned char));
			memcpy(show2.data, yyy, height2*width2*channels2 * sizeof(unsigned short));
			memcpy(show3.data, zzz, height3*width3*channels3 * sizeof(unsigned char));
			cv::imshow("1", show1);
			cv::imshow("2", show2);
			cv::imshow("3", show3);
			cv::waitKey(12);
#endif		

		}


		return 0;
	}
}