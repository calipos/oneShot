#include<algorithm>
#include"stringOp.h"
#include"logg.h"
#include"iofile.h"
#include"dataExplorer.h"

#define OPENCV_SHOW
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
		for (auto&bufferP : bufferVecP)
		{
			if (bufferP.data)
			{
				if (bufferP.Dtype=="uchar")
				{
					FrameRingBuffer<unsigned char>*ringBuffer = (FrameRingBuffer<unsigned char>*)bufferP.data;
					unsigned char*frameData = ringBuffer->pop();
					int height = ringBuffer->height;
					int width = ringBuffer->width;
					int channels = ringBuffer->channels;
#ifdef OPENCV_SHOW
					cv::Mat show1 = cv::Mat(height, width, channels == 1 ? CV_8UC1 : CV_8UC3);
					memcpy(show1.data, frameData,height*width*channels*sizeof(unsigned char));
#endif
				}

			}
		}
		return 0;
	}
}