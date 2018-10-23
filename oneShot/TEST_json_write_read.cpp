#ifdef COMPILE_TEST
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


int TEST_json_write_read()
{
	/*rapidjson::Document document;
	rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
	rapidjson::Value root(rapidjson::kObjectType);
	root.AddMember("name", "¸çÂ×²¼°¡", allocator);
	root.AddMember("gold", 1234, allocator);
	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	root.Accept(writer);
	std::string reststring = buffer.GetString();
	std::cout << reststring << std::endl;*/
	


	rapidjson::Document docRoot;
	rapidjson::Document::AllocatorType& allocator = docRoot.GetAllocator();
	docRoot.Parse<0>(unre::StringOP::parseJsonFile2str("test2.json").c_str());
	if (!docRoot.HasParseError())
	{
		LOG(INFO) << "parse json start";
		int config_number = docRoot.Size();
		LOG(INFO) << "sensors config number = " << docRoot.Size();
		LOG(INFO) << docRoot.IsArray();
		for (size_t i = 0; i < config_number; i++)
		{
			auto &thisSensorConfig = docRoot[i];
			LOG(INFO) << "-----------------------------------";
			LOG(INFO) << "friendName = " << thisSensorConfig["friendName"].GetString();
			LOG(INFO) << "sensorType = " << thisSensorConfig["sensorType"].GetString();
			LOG(INFO) << "height = " << thisSensorConfig["height"].GetInt();
			LOG(INFO) << "width = " << thisSensorConfig["width"].GetInt();
			thisSensorConfig["width"].SetInt(1);
			if (thisSensorConfig.HasMember("gold"))
			{
				thisSensorConfig["gold"].SetInt(12345);
			}
			else
			{
				thisSensorConfig.AddMember("gold", 1234, allocator);
			}
			LOG(INFO) << "extraConfigPath = " << thisSensorConfig["extraConfigPath"].GetString();
			LOG(INFO) << "-----------------------------------"; 
		}

		LOG(INFO) << "parse json end";


		rapidjson::StringBuffer buffer;
		rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
		docRoot.Accept(writer);
		std::string reststring = buffer.GetString();
		std::fstream fout("test2.json",std::ios::out);
		fout << reststring << std::endl;
		fout.close();
	}


	system("pause");
	return 0;
}
#endif // COMPILE_TEST
