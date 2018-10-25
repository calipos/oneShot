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

namespace unre
{


	class DeviceExplorer
	{
	public:
		DeviceExplorer();
		DeviceExplorer(const std::vector<std::string>& serial_numbers,
			const std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > >& sensorInfo,
			const std::vector<std::tuple<std::string, std::string>> & getExtraConfigFilPath);
		void init();
		void run();
		int pushStream(std::vector<Buffer> &bufferVecP);
		const std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, double>>>getRunTime_intr();
#ifdef USE_REALSENSE
	public:
		void remove_rs_devices(const rs2::event_information& info);
		void init_rs_devices(const std::vector<std::string>& serial_numbers,
			const std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > >& sensorInfo,
			const std::vector<std::tuple<std::string, std::string>> & getExtraConfigFilPath);
		void initRS();
		void runRS();
		int pushRsStream(std::vector<Buffer> &bufferVecP);

		template<typename T1, typename T2>
		void rs_pushStream_2(rs2::pipeline&p, FrameRingBuffer<T1>*buffer1, FrameRingBuffer<T2>*buffer2, DeviceExplorer*current, int channel1, int channel2)
		{
			CHECK(channel1 == 1 && channel2 == 2)<<"For rs, when pushing 2 stream, they must be depth and infred!";
			while (1)
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
		template<typename T1, typename T2, typename T3>
		void rs_pushStream_3(rs2::pipeline&p, FrameRingBuffer<T1>*buffer0, FrameRingBuffer<T2>*buffer1, FrameRingBuffer<T3>*buffer2, DeviceExplorer*current)
		{
			while (1)
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


	private:		
		rs2::context rs_ctx;
		std::unordered_map<std::string, std::tuple<rs2::pipeline, rs2::config, rs2::pipeline_profile, std::unordered_map<std::string, int>>> rsMap;
#endif // USE_REALSENSE
	private:
		std::vector<std::string> serial_numbers_;
		std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > > sensorInfo_;
		std::vector<std::tuple<std::string, std::string>>  getExtraConfigFilPath_;
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
	};

}



#endif