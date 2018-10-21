#include"logg.h"
#include"ringBuffer.h"
#include"dataExplorer.h"
#include "stringOp.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
int main()
{

	rapidjson::Document docRoot;
	docRoot.Parse<0>(unre::StringOP::parseJsonFile2str("E:/repo/oneShot/test.json").c_str());
	if (!docRoot.HasParseError())
	{
		LOG(INFO) << "parse json start";
		rapidjson::Value &imageRoot = docRoot["test_int"];
		rapidjson::Value &annoRoot = docRoot["test_object"]["vec_key"];
		LOG(INFO) << imageRoot.GetInt();
		LOG(INFO) << annoRoot.Size();
		LOG(INFO) << "parse json end";
	}

	unre::DataExplorer<float> ed(6);
	unre::FrameRingBuffer<int>  x(10,10,3);
	for (size_t i = 0; i < 100000; i++)
	{
		LOG(INFO) << i;
	}
	system("pause");
	return 0;
}