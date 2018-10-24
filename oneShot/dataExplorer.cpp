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
			CHECK(false) << "no config.json, even config_default.json������";
		}
		int exactStreamCnt = 0;
		auto &SensorsInfo = je.getSensorAssignmentInfo();
		std::vector<std::string> usedDeviceType;
		std::find_if( SensorsInfo.begin(), SensorsInfo.end(), [&usedDeviceType](auto&item)
		{
			std::vector<std::string> segs = StringOP::splitString(std::get<0>(item),",");
			CHECK(segs.size()==2)<<"the config err! must be seperated by a comma!";
			if (usedDeviceType.end() == std::find(usedDeviceType.begin(), usedDeviceType.end(),segs[0]))
			{
				usedDeviceType.emplace_back(segs[0]);
				LOG(INFO) << segs[0] << " HAS BEEN SET.";
			}
			return false;
		});
		LOG(INFO) << SensorsInfo.size() << " devices are required.";

		if (loadDevices2Stream())
		{

		}

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