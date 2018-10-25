#include<algorithm>
#include"jsonExplorer.h"
#include"stringOp.h"
#include"logg.h"

namespace unre
{
	JsonExplorer::JsonExplorer()
	{

	}
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
			LOG(FATAL) << "the json node:" << &node << ", has no member " << key << "!!!";
			return -1;
		}
	}
	template<>
	float JsonExplorer::getValue<float>(const rapidjson::Value&node, const char*key)
	{
		if (node.HasMember(key))
		{
			return node[key].GetFloat();
		}
		else
		{
			LOG(FATAL) << "the json node:" << &node << ", has no member " << key << "!!!";
			return -1;
		}
	}
	template<>
	double JsonExplorer::getValue<double>(const rapidjson::Value&node, const char*key)
	{
		if (node.HasMember(key))
		{
			return node[key].GetDouble();
		}
		else
		{
			LOG(FATAL) << "the json node:" << &node << ", has no member " << key << "!!!";
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
		if (sensorCnt < 1)
		{
			LOG(FATAL) << "the json file err, just 1 sensor?!!!";
		}
		for (int configIdx = 0; configIdx < sensorCnt; configIdx++)
		{
			int height = getValue<int>(docRoot[configIdx], "height");
			int width = getValue<int>(docRoot[configIdx], "width");
			int channels = getValue<int>(docRoot[configIdx], "channels");
			std::string dtype = getValue<std::string>(docRoot[configIdx], "Dtype");
			std::string friendName = getValue<std::string>(docRoot[configIdx], "friendName");
			std::string sn = getValue<std::string>(docRoot[configIdx], "serialNumber");
			std::string sensorType = getValue<std::string>(docRoot[configIdx], "sensorType");
			double cx = getValue<double>(docRoot[configIdx], "cx");
			double cy = getValue<double>(docRoot[configIdx], "cy");
			double fx = getValue<double>(docRoot[configIdx], "fx");
			double fy = getValue<double>(docRoot[configIdx], "fy");
			std::string extraFilePath = getValue<std::string>(docRoot[configIdx], "extraConfigPath");
			
			auto it = std::find_if(sensorAssignmentInfo.begin(), sensorAssignmentInfo.end(), [&](auto&item)
			{
				std::string&this_friendname_and_sn = std::get<0>(item);
				if (this_friendname_and_sn.compare(friendName + "," + sn) != 0)
				{
					return false;
				}
				else
				{
					auto&this_sensors_map = std::get<1>(item);
					if (this_sensors_map.find(sensorType) != this_sensors_map.end())
					{
						LOG(FATAL) << "The sensors sharing a identity sn, contains more than one " << sensorType;
					}
					else
					{
						this_sensors_map[sensorType] = std::make_tuple(configIdx, height, width, channels, dtype, std::unordered_map<std::string, double>{ {"cx", cx}, { "cy",cy }, { "fx",fx }, { "fy",fy } });
						CHECK(std::get<1>(extraConfigFilPath[extraConfigFilPath.size()-1]).compare(extraFilePath)==0)
							<<"she sensors sharing identity sn must be specified one configuration";
					}
					return true;
				}
			});
			if (it == sensorAssignmentInfo.end())
			{
				sensorAssignmentInfo.emplace_back
				(
					std::make_tuple
					(
						friendName + "," + sn,
						std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > >
						{ 
							{ 
								sensorType, 
								std::make_tuple
								(
									configIdx, 
									height, 
									width, 
									channels, 
									dtype, 
									std::unordered_map<std::string, double>{ {"cx",cx},{ "cy",cy },{ "fx",fx },{ "fy",fy } }
								) 
							} 
						}
					)
				);
				extraConfigFilPath.emplace_back(std::move(make_tuple(friendName + "," + sn,extraFilePath)));
			}
		}
		CHECK(extraConfigFilPath.size() == sensorAssignmentInfo.size()) << "extraConfigFiles not match sensors :(" << extraConfigFilPath.size() << " vs. " << sensorCnt << ")";
	}

	JsonExplorer::JsonExplorer(const JsonExplorer&other)
	{
		sensorCnt= other.sensorCnt;
		sensorAssignmentInfo = other.sensorAssignmentInfo;
		extraConfigFilPath = other.extraConfigFilPath;
	}


	JsonExplorer & JsonExplorer::operator=(const JsonExplorer& other)
	{
		sensorCnt = other.sensorCnt;
		sensorAssignmentInfo = other.sensorAssignmentInfo;
		extraConfigFilPath = other.extraConfigFilPath;
		return *this;
	}

	const std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > >&
		JsonExplorer::getSensorAssignmentInfo()
	{
		return sensorAssignmentInfo;
	}

	const std::vector<std::tuple<std::string, std::string>> & JsonExplorer::getExtraConfigFilPath()
	{
		return extraConfigFilPath;
	}
}