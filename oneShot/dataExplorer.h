#ifndef _DATAEXPLORER_H_
#define _DATAEXPLORER_H_

#include<vector>
#include"ringBuffer.h"
#include"jsonExplorer.h"
#include"deviceExplorer.h"
namespace unre
{
	class DataExplorer
	{
	public:
		DataExplorer() = delete;
		DataExplorer(DataExplorer&de) = delete;
		DataExplorer(DataExplorer&&de) = delete;
		explicit DataExplorer(int streamNum = 9, bool doCalib=false);
		int getExactStreamCnt();
		~DataExplorer() { /*delete dev_e;*/ };
		int deleteDevice() { delete dev_e; return 0; };
		int getBuffer_fortest();
		int getBuffer_fortest3();
		const std::vector<Buffer>&getBufferVecP();
		const std::vector<std::tuple<std::string, oneDevMap> >& getStreamInfo();
		int calibAllStream();
		std::unordered_map<int, cv::Mat*> stream2Intr;
		std::unordered_map<int, std::tuple<cv::Mat*, cv::Mat*>> stream2Extr;
	private:
		std::vector<Buffer> bufferVecP;//this member wrap the ringbuffer,and the ring buffer never std::move
		std::vector<void*> constBuffer = {0};//the ringbuffer will copy to there
		JsonExplorer je;
		DeviceExplorer*dev_e;
		int exactStreamCnt = 0;
		int initMatVect(std::vector<cv::Mat*>&imgs);
		int pop2Mats(std::vector<cv::Mat*>&imgs);//initMatVect must be called before
		bool doCalib_{false};
	};
}

#endif // !_DATAEXPLORER_H_
