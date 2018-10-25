#include<algorithm>
#include"stringOp.h"
#include"logg.h"
#include"iofile.h"
#include"dataExplorer.h"

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
		for (auto it1 = SensorsInfo.begin(); it1 != SensorsInfo.end(); it1++)
			{
				auto&this_device_sn = std::get<0>(*it1);
				auto&this_map = std::get<1>(*it1);
				for (auto map_it = this_map.begin(); map_it != this_map.end(); map_it++)
				{
						std::string sensorType = map_it->first;
						int sensoridx = std::get<0>(map_it->second);
						int height = std::get<1>(map_it->second);
						int width = std::get<2>(map_it->second);
						int channels = std::get<3>(map_it->second);
						std::string dataType = std::get<4>(map_it->second);
						auto intr = std::get<5>(map_it->second);
				}
			}
		LOG(INFO) << " - " << exactStreamCnt;

	}
}