#ifndef _JSON_EXPLORER_H_
#define _JSON_EXPLORER_H_
#include <vector>
#include <tuple>
#include <unordered_map>
#include <string>

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

class JsonExplorer
{
public:
	JsonExplorer()=delete;
	explicit JsonExplorer(const char*jsonFilePath);
	~JsonExplorer();
	template<typename Dtype>
	Dtype getValue(const rapidjson::Value &node,const char*key);

private:
	rapidjson::Document docRoot;
	int sensorCnt;
	
	////|    ["rgb" : <0,h,w,c,dtype>] |
	////|sn ["dep" : <1,h,w,c,dtype>] |
	////|    ["inf"   : <2,h,w,c,dtype>] |
	////|=====================|
	////|sn ["rgb" : <3,h,w,c>,dtype] |
	////|=====================|
	////|sn ["dep" : <4,h,w,c,dtype>] |
	////|     ["inf" : <5,h,w,c,dtype>]  |
	////|           ...               |
	std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int,int,int,int,std::string>> > > sensorAssignmentInfo;
	std::vector<std::string> extraConfigFilPath;
};




#endif // !_JSON_EXPLORER_H_
