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
		~DataExplorer() {};
		int loadDevices2Stream();
	private:
		std::vector<void*> bufferVecP;
		JsonExplorer je;
		DeviceExplorer* dev_e;
	};
}

#endif // !_DATAEXPLORER_H_
