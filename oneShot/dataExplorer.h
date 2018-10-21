#ifndef _DATAEXPLORER_H_
#define _DATAEXPLORER_H_

#include<vector>
#include"ringBuffer.h"

namespace unre
{
	template<typename Dtype>
	class DataExplorer
	{
	public:
		DataExplorer() = delete;
		DataExplorer(DataExplorer&de) = delete;
		DataExplorer(int streamNum = 6)
		{
			bufferVec_ = new std::vector<FrameRingBuffer<Dtype>*>();
			for (size_t i = 0; i < streamNum; i++)
			{
				bufferVec_->emplace_back();
			}

		}
		~DataExplorer() {};

	private:
		std::vector<FrameRingBuffer<Dtype>*> *bufferVec_;

	};
}

#endif // !_DATAEXPLORER_H_
