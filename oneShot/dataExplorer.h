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
		int doTsdf();
		int readExtrParams();
		const std::vector<Buffer>&getBufferVecP();
		const std::vector<std::tuple<std::string, oneDevMap> >& getStreamInfo();
		int calibAllStream_noLaser();
		int calibAllStream_Laser();
		int calibData();
		int oneDevShow();

		std::unordered_map<int, cv::Mat*> stream2Intr;
		std::unordered_map<int, std::tuple<cv::Mat, cv::Mat>> stream2Extr;
	private:
		std::vector<Buffer> bufferVecP;//this member wrap the ringbuffer,and the ring buffer never std::move
		std::vector<void*> constBuffer = {0};//the ringbuffer will copy to there
		JsonExplorer je;
		DeviceExplorer*dev_e;
		int exactStreamCnt = 0;
		////分配初始空间，然后才能pop, 在之后在调用 readExtrParams();
		int initMatVect(std::vector<cv::Mat*>&imgs);
		
		int pop2Mats(std::vector<cv::Mat*>&imgs);//initMatVect must be called before
		int pop2Mats_noInfred(std::vector<cv::Mat*>&imgs);
		bool doCalib_{false};
	};
}

#endif // !_DATAEXPLORER_H_
