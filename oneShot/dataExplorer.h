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
		int getBuffer();
	private:
		std::vector<Buffer> bufferVecP;
		JsonExplorer je;
		DeviceExplorer* dev_e;
		int exactStreamCnt = 0;
	};
}

#endif // !_DATAEXPLORER_H_
