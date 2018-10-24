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

	class JsonExplorer
	{
	public:
		JsonExplorer();
		explicit JsonExplorer(const char*jsonFilePath);
		explicit JsonExplorer(const JsonExplorer&other);
		JsonExplorer & operator=(const JsonExplorer& other);
		~JsonExplorer();
		template<typename Dtype>
		inline Dtype getValue(const rapidjson::Value &node, const char*key);
		const std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > >& getSensorAssignmentInfo();
		const std::vector<std::string> & getExtraConfigFilPath();
	private:
		rapidjson::Document docRoot;
		int sensorCnt;

		////|    ["rgb" : <0,h,w,c,dtype,intrMap>] |
		////|sn ["dep" : <1,h,w,c,dtype,intrMap>] |
		////|    ["inf"   : <2,h,w,c,dtype,intrMap>] |
		////|=====================|
		////|sn ["rgb" : <3,h,w,c,dtype,intrMap>] |
		////|=====================|
		////|sn ["dep" : <4,h,w,c,dtype,intrMap>] |
		////|     ["inf" : <5,h,w,c,dtype,intrMap>]  |
		////|           ...               |
		std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > > sensorAssignmentInfo;
		std::vector<std::string> extraConfigFilPath;
	};
}
#endif // !_JSON_EXPLORER_H_
