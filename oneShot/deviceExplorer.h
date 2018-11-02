#ifndef _DEVICE_EXPLORER_H_
#define _DEVICE_EXPLORER_H_

#include<tuple>
#include<unordered_map>
#include<string>
#include<mutex>
#include<condition_variable>

#include "logg.h"
#include "ringBuffer.h"

#ifdef USE_REALSENSE
#include "librealsense2/rs.h"
#include <librealsense2/rs_advanced_mode.hpp>
//#pragma comment(lib,"realsense2.lib")   = = 不起作用
#endif // USE_REALSENSE

#ifdef USE_VIRTUALCAMERA
#include"opencv2/opencv.hpp"
#endif

namespace unre
{


	class DeviceExplorer
	{
	public:
		DeviceExplorer();
		DeviceExplorer(const std::string&jsonFile,const std::vector<std::string>& serial_numbers,
			const std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > >& sensorInfo,
			const std::vector<std::tuple<std::string, std::string>> & getExtraConfigFilPath);
		void init();
		void run();
		int pushStream(std::vector<Buffer> &bufferVecP);
		const std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, double>>>getRunTime_intr();
		void pauseThread();
		void continueThread() ;
		void terminateThread() ;
		int initalConstBuffer(std::vector<void*>&constBuffer);
#ifdef USE_REALSENSE
	public:
		void remove_rs_devices(const rs2::event_information& info);
		void init_rs_devices(const std::vector<std::string>& serial_numbers,
			const std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > >& sensorInfo,
			const std::vector<std::tuple<std::string, std::string>> & getExtraConfigFilPath);
		void initRS();
		void runRS();
		int pushRsStream(std::vector<Buffer> &bufferVecP);
		int checkRsIntriParamAndWriteBack(const std::string&jsonFile, const std::string&sn, const std::string&sensorType,const std::string&whichIntr,const double&param);//this action doing after running
		
		//下面的函数会开一个线程一直push，其中会对几个流都push，所以一旦有个流push卡住了，会导致整个线程空转，而其余流的数据也会很快取空//所以外部都要pop
		template<typename T1, typename T2>
		void rs_pushStream_2(rs2::pipeline&p, FrameRingBuffer<T1>*buffer1, FrameRingBuffer<T2>*buffer2, DeviceExplorer*current, int channel1, int channel2)
		{
			CHECK(channel1 == 1 && channel2 == 2)<<"For rs, when pushing 2 stream, they must be depth and infred!";
			while (1)
			{
				while (!(buffer1->full() || buffer2->full()))
				{
					{
						std::unique_lock <std::mutex> lck(current->cv_pause_mtx);
						while (current->doPause)
						{
							current->cv_pause.wait(lck);
							current->doPause = false;
						}
					}
					{
						std::lock_guard <std::mutex> lck(current->termin_mtx);
						if (current->doTerminate)
						{
							LOG(INFO) << "Exit thread";
							break;
						}
					}
					rs2::frameset depth_and_color_frameset = p.wait_for_frames();
					//auto cf_pt = (unsigned char*)depth_and_color_frameset.first(RS2_STREAM_COLOR).get_data();
					auto df_pt = (unsigned short*)depth_and_color_frameset.get_depth_frame().get_data();
					auto if_pt = (unsigned char*)depth_and_color_frameset.get_infrared_frame().get_data();
					buffer1->push(df_pt);
					buffer2->push(if_pt);
				}
			}
		}
		//下面的函数会开一个线程一直push，其中会对几个流都push，所以一旦有个流push卡住了，会导致整个线程空转，而其余流的数据也会很快取空
		template<typename T1, typename T2, typename T3>
		void rs_pushStream_3(rs2::pipeline&p, FrameRingBuffer<T1>*buffer0, FrameRingBuffer<T2>*buffer1, FrameRingBuffer<T3>*buffer2, DeviceExplorer*current)
		{
			while (1)
			{
				while (!(buffer0->full() || buffer1->full() || buffer2->full()))
				{

					{
						std::unique_lock <std::mutex> lck(current->cv_pause_mtx);
						while (current->doPause)
						{
							current->cv_pause.wait(lck);
							current->doPause = false;
						}
					}
					{
						std::lock_guard <std::mutex> lck(current->termin_mtx);
						if (current->doTerminate)
						{
							LOG(INFO) << "Exit thread";
							break;
						}
					}
					rs2::frameset depth_and_color_frameset = p.wait_for_frames();
					auto cf_pt = (unsigned char*)depth_and_color_frameset.first(RS2_STREAM_COLOR).get_data();
					auto df_pt = (unsigned short*)depth_and_color_frameset.get_depth_frame().get_data();
					auto if_pt = (unsigned char*)depth_and_color_frameset.get_infrared_frame().get_data();

					buffer0->push(cf_pt);
					buffer1->push(df_pt);
					buffer2->push(if_pt);
				}
			}
		}
	private:		
		rs2::context rs_ctx;
		std::unordered_map<std::string, std::tuple<rs2::pipeline, rs2::config, rs2::pipeline_profile, std::unordered_map<std::string, int>>> rsMap;
#endif // USE_REALSENSE
#ifdef USE_VIRTUALCAMERA
	public:
		void initVirtualCamera();
		void runVirtualCamera();
		int pushVirtualCameraStream(std::vector<Buffer> &bufferVecP);
		//下面的函数会开一个线程一直push，其中会对几个流都push，所以一旦有个流push卡住了，会导致整个线程空转，而其余流的数据也会很快取空//所以外部都要pop
		template<typename T1, typename T2>
		void virtualCamera_pushStream_2(FrameRingBuffer<T1>*buffer1, FrameRingBuffer<T2>*buffer2, DeviceExplorer*current, int channel1, int channel2)
		{
			CHECK(channel1 == 1 && channel2 == 2) << "For rs, when pushing 2 stream, they must be depth and infred!";
			unsigned int showFrameIdx = 0;
			while (1)
			{
				while (!(buffer1->full() || buffer2->full()))
				{
					{
						std::unique_lock <std::mutex> lck(current->cv_pause_mtx);
						while (current->doPause)
						{
							current->cv_pause.wait(lck);
							current->doPause = false;
						}
					}
					{
						std::lock_guard <std::mutex> lck(current->termin_mtx);
						if (current->doTerminate)
						{
							LOG(INFO) << "Exit thread";
							break;
						}
					}

					cv::Mat dep_temp = cv::Mat::ones(buffer1->height, buffer1->width, CV_16UC1)*(showFrameIdx * 100 % 65000);
					cv::Mat inf_temp = cv::Mat::ones(buffer2->height, buffer2->width, CV_8UC1)*(showFrameIdx % 250);
					showFrameIdx++;
					buffer1->push((unsigned short*)dep_temp.data);
					buffer2->push(inf_temp.data);
				}
			}
		}
		//下面的函数会开一个线程一直push，其中会对几个流都push，所以一旦有个流push卡住了，会导致整个线程空转，而其余流的数据也会很快取空
		template<typename T1, typename T2, typename T3>
		void virtualCamera_pushStream_3(FrameRingBuffer<T1>*buffer0, FrameRingBuffer<T2>*buffer1, FrameRingBuffer<T3>*buffer2, DeviceExplorer*current)
		{
			unsigned int showFrameIdx = 0;
			while (1)
			{
				while (!(buffer0->full() || buffer1->full() || buffer2->full()))
				{

					{
						std::unique_lock <std::mutex> lck(current->cv_pause_mtx);
						while (current->doPause)
						{
							LOG(INFO) << "PAUSED!";
							current->cv_pause.wait(lck);
							LOG(INFO) << "CONTINUE!";
							current->doPause = false;
						}
					}
					{
						std::lock_guard <std::mutex> lck(current->termin_mtx);
						if (current->doTerminate)
						{
							LOG(INFO) << "Exit thread";
							break;
						}
					}
					cv::Mat color_temp = cv::Mat::ones(buffer0->height, buffer0->width, CV_8UC3)*(showFrameIdx % 250);
					cv::Mat dep_temp = cv::Mat::ones(buffer1->height, buffer1->width, CV_16UC1)*(showFrameIdx * 100 % 65000);
					cv::Mat inf_temp = cv::Mat::ones(buffer2->height, buffer2->width, CV_8UC1)*(showFrameIdx % 250);
					showFrameIdx++;

					buffer0->push(color_temp.data);
					buffer1->push((unsigned short*)dep_temp.data);
					buffer2->push(inf_temp.data);
				}
			}
		}
	private:
		std::unordered_map<std::string, std::unordered_map<std::string, int>> virtualCameraMap;
#endif // USE_VIRTUALCAMERA
	private:
		std::string jsonFile_;//this file is the init file, that is useful for modifying the intriParam
		std::vector<std::string> serial_numbers_;
		std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > > sensorInfo_;
		std::vector<std::tuple<std::string, std::string>>  extraConfigFilPath_;
		std::mutex _mutex;
		std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, double>>>runTime_intr;
		bool isDevicesInit{ false };
		bool isDevicesRunning{ false };

		std::atomic<bool> doPause = false; // 是否准备好
		std::mutex cv_pause_mtx; // doPause的锁
		std::condition_variable cv_pause; // doPause的条件变量.
		std::mutex termin_mtx; //g_bThreadRun的锁
		std::atomic<bool> doTerminate = false;

		std::vector<std::thread> threadSet;
		bool existRS = false;//是否需要用到RS
		bool existVirtualCamera = false;//是否需要用到虚拟camera
	};

}



#endif