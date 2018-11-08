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
		explicit DataExplorer(int streamNum = 9);
		int getExactStreamCnt();
		~DataExplorer() {};
		int getBuffer_fortest();
		int getBuffer_fortest3();
		const std::vector<Buffer>&getBufferVecP();
		const std::vector<std::tuple<std::string, oneDevMap> >& getStreamInfo();
		int calibAllStream();
		
	private:
		std::vector<Buffer> bufferVecP;//this member wrap the ringbuffer,and the ring buffer never std::move
		std::vector<void*> constBuffer = {0};//the ringbuffer will copy to there
		JsonExplorer je;
		DeviceExplorer* dev_e;
		int exactStreamCnt = 0;
		int initMatVect(std::vector<cv::Mat*>&imgs);
		int pop2Mats(std::vector<cv::Mat*>&imgs);//initMatVect must be called before
	};
}

#endif // !_DATAEXPLORER_H_
