#ifndef _JSON_EXPLORER_H_
#define _JSON_EXPLORER_H_
#include <vector>
#include <tuple>
#include <unordered_map>
#include <string>

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

namespace unre
{
	//streamIdx, h, w, c, dtype, intrMap
	using oneSensorInfo = std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> >;
	using oneDevMap = std::unordered_map<std::string, oneSensorInfo>;
	class JsonExplorer
	{
	public:
		JsonExplorer();
		explicit JsonExplorer(const char*jsonFilePath);
		explicit JsonExplorer(const JsonExplorer&other);
		JsonExplorer & operator=(const JsonExplorer& other);
		~JsonExplorer();
		template<typename Dtype>
		inline static Dtype getValue(const rapidjson::Value &node, const char*key);
		////|    ["rgb" : <0,h,w,c,dtype,intrMap>] |
		////|sn ["dep" : <1,h,w,c,dtype,intrMap>] |
		////|    ["inf"   : <2,h,w,c,dtype,intrMap>] |
		////|vvvvvvvvvvvvvvvvvvvvv|
		////|sn ["rgb" : <3,h,w,c,dtype,intrMap>] |
		////|vvvvvvvvvvvvvvvvvvvvv|
		////|sn ["dep" : <4,h,w,c,dtype,intrMap>] |
		////|     ["inf" : <5,h,w,c,dtype,intrMap>]  |
		////|           ...               |
		const std::vector<std::tuple<std::string, oneDevMap> >& getSensorAssignmentInfo();
		const std::vector<std::tuple<std::string,std::string>> & getExtraConfigFilPath();
	private:
		rapidjson::Document docRoot;
		int sensorCnt;

		////|    ["rgb" : <0,h,w,c,dtype,intrMap>] |
		////|sn ["dep" : <1,h,w,c,dtype,intrMap>] |
		////|    ["inf"   : <2,h,w,c,dtype,intrMap>] |
		////|vvvvvvvvvvvvvvvvvvvvv|
		////|sn ["rgb" : <3,h,w,c,dtype,intrMap>] |
		////|vvvvvvvvvvvvvvvvvvvvv|
		////|sn ["dep" : <4,h,w,c,dtype,intrMap>] |
		////|     ["inf" : <5,h,w,c,dtype,intrMap>]  |
		////|           ...               |
		std::vector<std::tuple<std::string, oneDevMap> > sensorAssignmentInfo;
		std::vector<std::tuple<std::string, std::string>> extraConfigFilPath;
	};
}
#endif // !_JSON_EXPLORER_H_
