#ifndef _RINGBUFFER_H_
#define _RINGBUFFER_H_
#include < atomic >
#include < mutex >



class spinlock
{
	std::atomic_flag flag = ATOMIC_FLAG_INIT;
public:
	void lock() noexcept
	{
		while (flag.test_and_set(std::memory_order_acquire))
			;
	}
	void unlock() noexcept 
	{ 
		flag.clear(std::memory_order_release); 
	}
	bool try_lock() noexcept 
	{
		return !flag.test_and_set(std::memory_order_acquire);
	}
};

#define MAX_QUEUE_SIZE (20)

namespace unre
{
	
	


	template < typename T >
	class FrameRingBuffer
	{
	public:
		typedef T Value_type;
		int height;
		int width;
		int channels;
		int frameEleCnt;
		std::string sensorType;
		std::string dType;
		double cx{ 0. }, cy{ 0. }, fx{ 0. }, fy{ 0. };
		
		//FrameRingBuffer() = delete;;
		explicit FrameRingBuffer(const int height_, const int width_, const int channels_);

		FrameRingBuffer(const FrameRingBuffer &other);
		FrameRingBuffer& operator=(const FrameRingBuffer &) = delete;

		~FrameRingBuffer();

		// pushes an item to FrameRingBuffer tail
		void push(const T*itemPtr);

		// pops an item from FrameRingBuffer head
		int pop(void*mem = NULL);//if mem=NULL, it means pop nothing but the ringBuffer wheels

		// try to push an item to FrameRingBuffer tail
		bool try_and_push(const T* itemPtr);

		// try to pop and item from FrameRingBuffer head
		int try_and_pop(void*mem = NULL);//if mem=NULL, it means pop nothing but the ringBuffer wheels

		bool full();
		bool empty();
		unsigned capacity() { return CAPACITY; }
		unsigned count();

	protected:
		spinlock lock;
		const unsigned CAPACITY;  // FrameRingBuffer capacity
		T *data;                  // array to store the items
		unsigned cnt;             // FrameRingBuffer count
		unsigned head;            // also the readIndex
		unsigned tail;            // also the writeIndex
	};

	
	template < typename T >
	FrameRingBuffer< T >::FrameRingBuffer(const int height_, const int width_, const int channels_) :
		CAPACITY(MAX_QUEUE_SIZE), height(height_), width(width_), channels(channels_), cnt(0), head(0), tail(0)
	{
		frameEleCnt = height_*width_*channels_;
		data = new T[frameEleCnt*CAPACITY];
	}

	template < typename T >
	FrameRingBuffer< T >::FrameRingBuffer(const FrameRingBuffer &other)
	{
		std::lock_guard< spinlock > lg(lock);
		CAPACITY = other.CAPACITY;
		cnt = other.cnt;
		head = other.head;
		tail = other.tail;
		height = other.height;
		width = other.width;
		channels = other.channels;
		frameEleCnt = other.frameEleCnt;
		data = new T[frameEleCnt*CAPACITY];
		memcpy(data, other.data, height_*width_*channels_*CAPACITY * sizeof(T));
	}

	template < typename T >
	FrameRingBuffer< T >::~FrameRingBuffer()
	{
		//LOG(INFO) << "FrameRingBuffer DESTROYED";
		delete[] data;
	}

	template < typename T >
	void FrameRingBuffer< T >::push(const T*itemPtr)
	{
		while (!try_and_push(itemPtr))
			;
	}

	template < typename T >
	int FrameRingBuffer< T >::pop(void*mem)
	{
		thread_local int ret=-1;
		while (ret = try_and_pop(mem))
		{
			if (ret==0)
			{
				break;
			}
		}			
		return 0;
	}

	template < typename T >
	bool FrameRingBuffer< T >::try_and_push(const T *itemPtr)
	{
		std::lock_guard< spinlock > lg(lock);
		if (cnt == CAPACITY)
			return false;    // full
		++cnt;
		memcpy((void*)(data + frameEleCnt * tail), (void*)itemPtr, frameEleCnt * sizeof(T));
		tail++;
		if (tail == CAPACITY)
			tail -= CAPACITY;
		return true;
	}

	template < typename T >
	int FrameRingBuffer< T >::try_and_pop(void*mem)
	{
		std::lock_guard< spinlock > lg(lock);
		if (cnt == 0)
			return -1;    // empty
		--cnt;
		unsigned idx = head;
		++head;
		if (head == CAPACITY)
			head -= CAPACITY;
		if (mem!=NULL)
		{
			memcpy(mem, (void*)(data + frameEleCnt * idx), frameEleCnt * sizeof(T));
		}
		return 0;
	}

	template < typename T >
	bool FrameRingBuffer< T >::full()
	{
		std::lock_guard< spinlock > lg(lock);
		return cnt == CAPACITY;
	}

	template < typename T >
	bool FrameRingBuffer< T >::empty()
	{
		std::lock_guard< spinlock > lg(lock);
		return cnt == 0;
	}

	template < typename T >
	unsigned FrameRingBuffer< T >::count()
	{
		std::lock_guard< spinlock > lg(lock);
		return cnt;
	}


	struct Buffer
	{
		void*data;
		std::string Dtype;
		Buffer()
		{
			data = NULL;
			Dtype = "";
		}
		template<typename T>
		Buffer(FrameRingBuffer<T>*buffer)
		{
			data = (void*)buffer;
			if (std::string(typeid(T(0)).name()).compare("unsigned char") == 0)
			{
				Dtype = "unsigned char";
			}
			else if (std::string(typeid(T(0)).name()).compare("unsigned short") == 0)
			{
				Dtype = "unsigned short";
			}
			else
			{
				LOG(FATAL) << "not support type : " << typeid(T(0)).name();
			}
		}
	};
}
#endif