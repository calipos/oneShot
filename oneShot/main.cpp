#include"logg.h"
#include"ringBuffer.h"
int main()
{
	FrameRingBuffer<int>  x(10,10,3);
	for (size_t i = 0; i < 100000; i++)
	{
		LOG(INFO) << i;
	}
	system("pause");
	return 0;
}