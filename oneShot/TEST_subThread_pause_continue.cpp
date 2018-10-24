#ifdef COMPILE_TEST
#include "logg.h"
#include <thread>               
#include <mutex>                
#include <condition_variable>   
#include <time.h>
#include <chrono>
#include <atomic>
using namespace std;

std::atomic<bool> doPause = false; // 是否准备好
std::mutex cv_pause_mtx; // doPause的锁
std::condition_variable cv_pause; // doPause的条件变量.

std::mutex termin_mtx; //g_bThreadRun的锁

std::atomic<bool> doTerminate = false;

void do_print_id(int id)
{
	while (1)
	{
		{
			std::unique_lock <std::mutex> lck(cv_pause_mtx);
			while (doPause)
			{
				cv_pause.wait(lck);
				doPause = false;
			}
		}
		{
			std::lock_guard <std::mutex> lck(termin_mtx);
			if (doTerminate)
			{
				cout << "Exit thread" << endl;
				break;
			}
		}
		std::cout << "sunThread running ... " << clock() << endl;
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}
}


int TEST_thread_pauseContinue()
{
	doPause = false;

	std::thread thread;
	thread = std::thread(do_print_id, 1);
	for (size_t i = 0; i < 50; i++)
	{
		std::cout << "main thread setup... " << clock() << endl;
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}

	std::cout << "main thread sleep... " << clock() << endl;
	std::this_thread::sleep_for(std::chrono::seconds(1));
	std::cout << "main thread wakeup. and pause sub thread" << clock() << endl;
	{
		std::unique_lock <std::mutex> lck(cv_pause_mtx);
		doPause = true; //sub thread will hanged up until the signal
	}
	for (size_t i = 0; i < 50; i++)
	{
		std::cout << "subThread hanged  and main thread prepare the signal... " << clock() << endl;
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}
	std::cout << "main thread give the condition variable... " << clock() << endl;
	{
		std::unique_lock <std::mutex> lck(cv_pause_mtx);
		cv_pause.notify_all();
	}

	std::this_thread::sleep_for(std::chrono::seconds(3));

	std::cout << "main thread terminate the subThread " << clock() << endl;
	{
		std::unique_lock <std::mutex> lck(termin_mtx);
		doTerminate = true;
	}
	std::this_thread::sleep_for(std::chrono::seconds(100));
	return 0;
}
#endif