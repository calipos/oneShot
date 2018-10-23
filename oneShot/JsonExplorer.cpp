#include<algorithm>
#include"JsonExplorer.h"
#include"stringOp.h"
#include"logg.h"


JsonExplorer::~JsonExplorer()
{
}

template<>
int JsonExplorer::getValue<int>(const rapidjson::Value&node, const char*key)
{
	if (node.HasMember(key))
	{
		return node[key].GetInt();
	}
	else
	{
		LOG(FATAL) << "the json node:"<<&node<<", has no member "<< key <<"!!!";
		return -1;
	}
}
template<>
std::string JsonExplorer::getValue<std::string>(const rapidjson::Value&node, const char*key)
{
	if (node.HasMember(key))
	{
		return node[key].GetString();
	}
	else
	{
		LOG(FATAL) << "the json node:" << &node << ", has no member " << key << "!!!";
		return "";
	}
}

JsonExplorer::JsonExplorer(const char*jsonFilePath)
{
	docRoot.Parse<0>(unre::StringOP::parseJsonFile2str(jsonFilePath).c_str());
	if (!docRoot.IsArray())
	{
		LOG(FATAL) << "the json file err, not a array!!!";
	}
	sensorCnt = docRoot.Size();
	if (sensorCnt<1)
	{
		LOG(FATAL) << "the json file err, just 1 sensor?!!!";
	}
	for (size_t configIdx = 0; configIdx < sensorCnt; configIdx++)
	{
		int height = getValue<int>(docRoot[configIdx], "height");
		int width = getValue<int>(docRoot[configIdx], "width");
		int channels = getValue<int>(docRoot[configIdx], "channels");
		std::string dtype = getValue<std::string>(docRoot[configIdx], "Dtype");
		std::string sn = getValue<std::string>(docRoot[configIdx], "serialNumber");
		std::string sensorType = getValue<std::string>(docRoot[configIdx], "sensorType");
		std::string extraFilePath = getValue<std::string>(docRoot[configIdx], "extraConfigPath");

		extraConfigFilPath.emplace_back(std::move(extraFilePath));
		auto it = std::find_if(sensorAssignmentInfo.begin(), sensorAssignmentInfo.end(), [&](auto&item)
		{
			std::string&this_sn = std::get<0>(item);
			if (sn.compare(this_sn)!=0)
			{
				return false;
			}
			else
			{
				auto&this_sensors_map = std::get<1>(item);
				if (this_sensors_map.find(sensorType) != this_sensors_map.end())
				{
					LOG(FATAL) << "The sensors sharing a identity sn, contains more than one "<< sensorType;
				}
				else
				{
					this_sensors_map[sensorType] = std::make_tuple(configIdx, height, width, channels, dtype);
				}
				return true;
			}
		});		
		if (it==sensorAssignmentInfo.end())
		{
			sensorAssignmentInfo.emplace_back(
				std::make_tuple( sn, 
					std::unordered_map<std::string, std::tuple<int, int, int, int, std::string>>
					{ { sensorType ,std::make_tuple(configIdx, height, width, channels,dtype) } } )
			);
		}		
	}
	CHECK(extraConfigFilPath.size() == sensorCnt) << "extraConfigFiles not match sensors :(" << extraConfigFilPath.size() << " vs. " << sensorCnt<<")";
}