#include<set>
#include<thread>
#include"stringOp.h"
#include"deviceExplorer.h"
#include"dataExplorer.h"
#include"ringBuffer.h"
#ifdef OPENCV_SHOW
#include "opencv2/opencv.hpp"
#endif
#ifdef USE_REALSENSE
#include "librealsense2/rs.h"
#include <librealsense2/rs_advanced_mode.hpp>
//#pragma comment(lib,"realsense2.lib")    = = 不起作用
#endif // USE_REALSENSE
namespace unre
{
	
	DeviceExplorer::DeviceExplorer()
	{
	}

	DeviceExplorer::DeviceExplorer(const std::vector<std::string>& serial_numbers,
		const std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > >& sensorInfo,
		const std::vector<std::tuple<std::string, std::string>> & getExtraConfigFilPath)
	{
		serial_numbers_ = serial_numbers;
		sensorInfo_ = sensorInfo;
		getExtraConfigFilPath_ = getExtraConfigFilPath;
	}
	const std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, double>>>DeviceExplorer::getRunTime_intr()
	{
		return runTime_intr;
	}
	
	void DeviceExplorer::pauseThread()
	{
		std::unique_lock <std::mutex> lck(cv_pause_mtx);
		doPause = true; 
	}
	void DeviceExplorer::continueThread()
	{
		while (doPause)
		{
			std::unique_lock <std::mutex> lck(cv_pause_mtx);
			LOG(INFO) << "NOTICE_ALL";
			cv_pause.notify_all();
			doPause = false;
		}
	}
	void DeviceExplorer::terminateThread()
	{
		std::unique_lock <std::mutex> lck(termin_mtx);
		doTerminate = true;
	}
	
	void DeviceExplorer::init()
	{
#ifdef USE_REALSENSE		
		for(auto&dev:serial_numbers_)
		{
			if (StringOP::splitString(dev, ",")[0].compare("realsenseD415")==0)
			{
				existRS = true;
				break;
			}
		}
		if (existRS)
		{
			initRS();
		}
#endif
#ifdef USE_VIRTUALCAMERA		
		for (auto&dev : serial_numbers_)
		{
			if (StringOP::splitString(dev, ",")[0].compare("virtualCamera") == 0)
			{
				existVirtualCamera = true;
				break;
			}
		}
		if (existVirtualCamera)
		{
			initVirtualCamera();
		}		
#endif
		isDevicesInit=true;
	}
	void DeviceExplorer::run()
	{
#ifdef USE_REALSENSE
		if (existRS)
		{
			runRS();
		}		
#endif
#ifdef USE_VIRTUALCAMERA
		if (existVirtualCamera)
		{
			runVirtualCamera();
		}
#endif
		isDevicesRunning = true;
	}

	int DeviceExplorer::pushStream(std::vector<Buffer> &bufferVecP)
	{
		int ret = 0;
#ifdef USE_REALSENSE
		int ret0 = pushRsStream(bufferVecP);
#endif
#ifdef USE_VIRTUALCAMERA
		int ret1= pushVirtualCameraStream(bufferVecP);
#endif
		return ret;
	}

#ifdef USE_REALSENSE
	void DeviceExplorer::remove_rs_devices(const rs2::event_information& info)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		auto itr = rsMap.begin();
		while (itr != rsMap.end())
		{
			if (info.was_removed(std::get<2>(itr->second).get_device()))
			{
				itr = rsMap.erase(itr);
			}
			else
			{
				++itr;
			}
		}
	}
	void DeviceExplorer::init_rs_devices(const std::vector<std::string>& serial_numbers,
		const std::vector<std::tuple<std::string, std::unordered_map<std::string, std::tuple<int, int, int, int, std::string, std::unordered_map<std::string, double> > > > >& sensorInfo,
		const std::vector<std::tuple<std::string, std::string>> & getExtraConfigFilPath)
	{
		CHECK(serial_numbers.size()>0)<<"the rs sn is NULL";
		std::set<std::string> serial_number_set;
		for (auto&d : serial_numbers) serial_number_set.insert(d);
		CHECK(serial_number_set.size()== serial_numbers.size())<<"the serial_numbers duplicated";
		bool dev_not_find = false;
		for (auto&& dev : rs_ctx.query_devices())
		{
			std::string serial_number(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
			std::lock_guard<std::mutex> lock(_mutex);
			if (serial_numbers.end() == std::find(serial_numbers.begin(), serial_numbers.end(), "realsenseD415,"+serial_number))
			{
				continue;
			}
			else
			{
				if ("Intel RealSense D415" != std::string(dev.get_info(RS2_CAMERA_INFO_NAME)))
				{
					std::vector<rs2::sensor> sensors = dev.query_sensors();
					auto color_sensor = sensors[0];
					if (color_sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE))
						color_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
				}
				if (dev.is<rs400::advanced_mode>())
				{
					auto advanced_mode_dev = dev.as<rs400::advanced_mode>();
					if (!advanced_mode_dev.is_enabled())
					{
						advanced_mode_dev.toggle_advanced_mode(true);
					}
					std::string str;
					std::fstream _file;
					_file.open("rs415_setting.json", std::ios::in);
					if (!_file)
					{
						str = "{\"aux-param-autoexposure-setpoint\": \"400\",\"aux-param-colorcorrection1\": \"0.461914\",\"aux-param-colorcorrection10\": \"-0.553711\",\"aux-param-colorcorrection11\": \"-0.553711\",\"aux-param-colorcorrection12\": \"0.0458984\",\"aux-param-colorcorrection2\": \"0.540039\",\"aux-param-colorcorrection3\": \"0.540039\",\"aux-param-colorcorrection4\": \"0.208008\",\"aux-param-colorcorrection5\": \"-0.332031\",\"aux-param-colorcorrection6\": \"-0.212891\",\"aux-param-colorcorrection7\": \"-0.212891\",\"aux-param-colorcorrection8\": \"0.68457\",\"aux-param-colorcorrection9\": \"0.930664\",\"aux-param-depthclampmax\": \"1000\",\"aux-param-depthclampmin\": \"0\",\"aux-param-disparityshift\": \"0\",\"controls-autoexposure-auto\": \"True\",\"controls-autoexposure-manual\": \"33000\",\"controls-color-autoexposure-auto\": \"True\",\"controls-color-autoexposure-manual\": \"156\",\"controls-color-backlight-compensation\": \"0\",\"controls-color-brightness\": \"0\",\"controls-color-contrast\": \"50\",\"controls-color-gain\": \"64\",\"controls-color-gamma\": \"300\",\"controls-color-hue\": \"0\",\"controls-color-power-line-frequency\": \"3\",\"controls-color-saturation\": \"64\",\"controls-color-sharpness\": \"50\",\"controls-color-white-balance-auto\": \"True\",\"controls-color-white-balance-manual\": \"4600\",\"controls-depth-gain\": \"16\",\"controls-depth-white-balance-auto\": \"False\",\"controls-laserpower\": \"360\",\"controls-laserstate\": \"on\",\"ignoreSAD\": \"0\",\"param-autoexposure-setpoint\": \"400\",\"param-censusenablereg-udiameter\": \"9\",\"param-censusenablereg-vdiameter\": \"9\",\"param-censususize\": \"9\",\"param-censusvsize\": \"9\",\"param-depthclampmax\": \"1000\",\"param-depthclampmin\": \"0\",\"param-depthunits\": \"1000\",\"param-disableraucolor\": \"0\",\"param-disablesadcolor\": \"0\",\"param-disablesadnormalize\": \"0\",\"param-disablesloleftcolor\": \"0\",\"param-disableslorightcolor\": \"0\",\"param-disparitymode\": \"0\",\"param-disparityshift\": \"0\",\"param-lambdaad\": \"618\",\"param-lambdacensus\": \"15\",\"param-leftrightthreshold\": \"18\",\"param-maxscorethreshb\": \"1443\",\"param-medianthreshold\": \"789\",\"param-minscorethresha\": \"96\",\"param-neighborthresh\": \"12\",\"param-raumine\": \"2\",\"param-rauminn\": \"1\",\"param-rauminnssum\": \"6\",\"param-raumins\": \"3\",\"param-rauminw\": \"3\",\"param-rauminwesum\": \"7\",\"param-regioncolorthresholdb\": \"0.109589\",\"param-regioncolorthresholdg\": \"0.572407\",\"param-regioncolorthresholdr\": \"0.0176125\",\"param-regionshrinku\": \"4\",\"param-regionshrinkv\": \"0\",\"param-robbinsmonrodecrement\": \"6\",\"param-robbinsmonroincrement\": \"21\",\"param-rsmdiffthreshold\": \"1.21875\",\"param-rsmrauslodiffthreshold\": \"0.28125\",\"param-rsmremovethreshold\": \"0.488095\",\"param-scanlineedgetaub\": \"8\",\"param-scanlineedgetaug\": \"200\",\"param-scanlineedgetaur\": \"279\",\"param-scanlinep1\": \"55\",\"param-scanlinep1onediscon\": \"326\",\"param-scanlinep1twodiscon\": \"134\",\"param-scanlinep2\": \"235\",\"param-scanlinep2onediscon\": \"506\",\"param-scanlinep2twodiscon\": \"206\",\"param-secondpeakdelta\": \"222\",\"param-texturecountthresh\": \"0\",\"param-texturedifferencethresh\": \"2466\",\"param-usersm\": \"1\",\"param-zunits\": \"1000\",\"stream-depth-format\": \"Z16\",\"stream-fps\": \"30\",\"stream-height\": \"720\",\"stream-ir-format\": \"Y8\",\"stream-width\": \"1280\"}";
					}
					else
					{
						_file.close();
						std::fstream fs("rs415_setting.json");
						std::stringstream ss;
						ss << fs.rdbuf();
						str = ss.str(); 
					}
					advanced_mode_dev.load_json(str);
				}
				else
				{
					CHECK(false)<< "Current device doesn't support advanced-mode!\n";
				}
				std::vector<rs2::sensor> sensors = dev.query_sensors();
				auto color_sensor = sensors[1]; //1 color sensor 0 depth 
				if (color_sensor.supports(RS2_OPTION_EXPOSURE))
				{
					color_sensor.set_option(RS2_OPTION_EXPOSURE, 1250);
				}
				if (color_sensor.supports(RS2_OPTION_WHITE_BALANCE))
				{
					color_sensor.set_option(RS2_OPTION_WHITE_BALANCE, 4600);
				}
				auto depth_sensor = sensors[0];
				if (depth_sensor.supports(RS2_OPTION_LASER_POWER))
				{
					auto range = depth_sensor.get_option_range(RS2_OPTION_LASER_POWER);
				}
				//auto depth_sensor = sensors[0];
				if (depth_sensor.supports(RS2_OPTION_VISUAL_PRESET))
				{
					depth_sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY); // Set max power
				}
				if (depth_sensor.supports(RS2_OPTION_FRAMES_QUEUE_SIZE))
				{
					depth_sensor.set_option(RS2_OPTION_FRAMES_QUEUE_SIZE, 0);
				}

				std::string key_ = "realsenseD415," + serial_number;

				int rgb_h = -1, rgb_w = -1 ;
				int dep_h = -1, dep_w = -1 ;
				int inf_h = -1, inf_w = -1 ;
				std::unordered_map<std::string, int> streamTable;
				if (sensorInfo.end() == std::find_if(sensorInfo.begin(), sensorInfo.end(), [&](auto&item) 
				{
					if (key_.compare(std::get<0>(item))!=0) return false;
					auto &this_rs_dev = std::get<1>(item);
					for (auto&map_item : this_rs_dev)
					{
						if (map_item.first.compare("rgb")==0)
						{
							int streamIdx = std::get<0>(map_item.second);
							streamTable["rgb"] = streamIdx;
							rgb_h = std::get<1>(map_item.second);
							rgb_w = std::get<2>(map_item.second);
						}
						if (map_item.first.compare("depth") == 0)
						{
							int streamIdx = std::get<0>(map_item.second);
							streamTable["depth"] = streamIdx;
							dep_h = std::get<1>(map_item.second);
							dep_w = std::get<2>(map_item.second);
						}
						if (map_item.first.compare("infred") == 0)
						{
							int streamIdx = std::get<0>(map_item.second);
							streamTable["infred"] = streamIdx;
							inf_h = std::get<1>(map_item.second);
							inf_w = std::get<2>(map_item.second);
						}
					}
					return true;
				
				}))
				{
					CHECK(false)<<"SN not match the jsonExplorer Info!";
				}
				CHECK(dep_h > 0 && dep_w > 0 && dep_h == inf_h && dep_w == inf_w)<<"RS's depth and infred must be set the same!";
				if (rgb_h<1)
				{
					rgb_h = dep_h;
					rgb_w = dep_w;
				}
				dev_not_find = true;
				rs2::pipeline p;
				rs2::config c;
				rs2::pipeline_profile profile;
				c.enable_device(serial_number);
				c.enable_stream(RS2_STREAM_COLOR, rgb_w, rgb_h, RS2_FORMAT_RGB8, 30);
				//c.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 30);
				c.enable_stream(RS2_STREAM_DEPTH, dep_w, dep_h, RS2_FORMAT_Z16, 30);
				c.enable_stream(RS2_STREAM_INFRARED, 1, inf_w, inf_h, RS2_FORMAT_Y8, 30);

				//profile = p.start(c);

				rsMap[key_] = std::make_tuple(p,c, profile, streamTable);
			}
		}
	
		if (!dev_not_find)
		{
			CHECK(false)<<"the serial number specified in config not match the ctx!";
		}
	}
	void DeviceExplorer::initRS()
	{
		rs_ctx.set_devices_changed_callback([&](rs2::event_information& info)
		{
			remove_rs_devices(info);
			init_rs_devices(serial_numbers_, sensorInfo_, getExtraConfigFilPath_);
		});
		init_rs_devices(serial_numbers_, sensorInfo_, getExtraConfigFilPath_);
	}
	void DeviceExplorer::runRS()
	{
		runTime_intr.clear();
		//一下得到运行时的内参
		for (auto& dev_info : rsMap)
		{
			const std::string&name_sn = dev_info.first;
			rs2::pipeline&p = std::get<0>(dev_info.second);
			rs2::config&c = std::get<1>(dev_info.second);
			rs2::pipeline_profile&profile = std::get<2>(dev_info.second);
			profile = p.start(c);
			auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
			auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
			auto infrared_stream = profile.get_stream(RS2_STREAM_INFRARED).as<rs2::video_stream_profile>();
			auto c_i = color_stream.get_intrinsics();
			auto d_i = depth_stream.get_intrinsics();
			auto f_i = infrared_stream.get_intrinsics();

			auto &streamMap = std::get<3>(dev_info.second);
			for (auto&stream : streamMap)
			{
				const std::string&sensorType = stream.first;
				if (sensorType.compare("rgb")==0)
				{
					runTime_intr[name_sn]["rgb"]["cx"] = c_i.ppx;
					runTime_intr[name_sn]["rgb"]["cy"] = c_i.ppy;
					runTime_intr[name_sn]["rgb"]["fx"] = c_i.fx;
					runTime_intr[name_sn]["rgb"]["fy"] = c_i.fy;
				}
				else if (sensorType.compare("depth") == 0)
				{
					runTime_intr[name_sn]["depth"]["cx"] = d_i.ppx;
					runTime_intr[name_sn]["depth"]["cy"] = d_i.ppy;
					runTime_intr[name_sn]["depth"]["fx"] = d_i.fx;
					runTime_intr[name_sn]["depth"]["fy"] = d_i.fy;
				}
				else if (sensorType.compare("infred") == 0)
				{
					runTime_intr[name_sn]["infred"]["cx"] = f_i.ppx;
					runTime_intr[name_sn]["infred"]["cy"] = f_i.ppy;
					runTime_intr[name_sn]["infred"]["fx"] = f_i.fx;
					runTime_intr[name_sn]["infred"]["fy"] = f_i.fy;
				}
				else
				{
					CHECK(false)<<"only support [egb][depth][infred], not "<< sensorType;
				}
			}	
		}
	}
	int DeviceExplorer::pushRsStream(std::vector<Buffer> &bufferVecP)
	{
		CHECK(bufferVecP.size()>0)<<"bufferVecP must be resize before!";
		int rs_stream_cnt = 0;
		for (auto&rs : rsMap)
		{
			for(auto&stream_ : std::get<3>(rs.second))
			{
				rs_stream_cnt++;
			}
		}
		CHECK(rs_stream_cnt<=bufferVecP.size())<<"rs stream cnt big than total stream cnt: ("<< rs_stream_cnt<<" : "<< bufferVecP.size()<<")";
		for (auto&rs : rsMap)
		{
			rs2::pipeline&p = std::get<0>(rs.second);
			auto&this_dev_info = std::get<3>(rs.second);
			std::vector<int> bufferIdx(10,-1);//一个相机支持最多输出10个流，0位代表rgb，1位代表dep，2位代表inf
			for (auto&sensorInfo : this_dev_info)
			{
				const std::string&sensorType = sensorInfo.first;
				const int&sensorIdx = sensorInfo.second;
				int height_ = -1;
				int width_ = -1;;
				int channels_ = -1;
				std::string sensorType_ = "";
				std::string dataType_ = "";
				if (sensorInfo_.end() == std::find_if(sensorInfo_.begin(), sensorInfo_.end(), [&](auto&item) 
				{
					auto&dev_map = std::get<1>(item);
					for (auto&this_sensor : dev_map)
					{
						if (sensorIdx==std::get<0>(this_sensor.second))
						{
							sensorType_ = this_sensor.first;
							height_ = std::get<1>(this_sensor.second);
							width_ = std::get<2>(this_sensor.second);
							channels_ = std::get<3>(this_sensor.second);
							dataType_ = std::get<4>(this_sensor.second);
							return true;
						}
					}
					return false;				
				}))
				{
					CHECK(false)<<"MATCH ERR";
				}
			
				CHECK(bufferVecP[sensorIdx].data == NULL)<<"there should be null";
				if (dataType_.compare("uchar")==0)
				{
					bufferVecP[sensorIdx].data = new FrameRingBuffer<unsigned char>(height_, width_, channels_);
					bufferVecP[sensorIdx].Dtype = "uchar";
				}
				else if (dataType_.compare("ushort")==0)
				{
					bufferVecP[sensorIdx].data = new FrameRingBuffer<unsigned short>(height_, width_, channels_);
					bufferVecP[sensorIdx].Dtype = "ushort";
				}
				else
				{
					CHECK(false) << "NOT SOPPORT TYPE";
				}
				if (sensorType_.compare("rgb")==0)
				{
					bufferIdx[0] = sensorIdx;
				}
				else if (sensorType_.compare("depth") == 0)
				{
					bufferIdx[1] = sensorIdx;
				}
				else if (sensorType_.compare("infred") == 0)
				{
					bufferIdx[2] = sensorIdx;
				}
				else
				{
					CHECK(false) << "NOT SOPPORT TYPE";
				}
			}
			
			//TODO:not elegent
			if (bufferIdx[0] >= 0&& bufferIdx[1] >= 0&&bufferIdx[2] >= 0)
			{				
				threadSet.emplace_back(std::thread(&DeviceExplorer::rs_pushStream_3<unsigned char, unsigned short, unsigned char>, this,p, (FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[0]].data, (FrameRingBuffer<unsigned short>*)bufferVecP[bufferIdx[1]].data, (FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[2]].data, this));
				
				//rs_pushStream_3(p, (FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[0]], (FrameRingBuffer<unsigned short>*)bufferVecP[bufferIdx[1]], (FrameRingBuffer<unsigned short>*)bufferVecP[bufferIdx[2]], this);
			}
			else if (bufferIdx[0] < 0 && bufferIdx[1] >= 0 && bufferIdx[2] >= 0)
			{
				threadSet.emplace_back(std::thread(&DeviceExplorer::rs_pushStream_2<unsigned short, unsigned char>, this, p, (FrameRingBuffer<unsigned short>*)bufferVecP[bufferIdx[1]].data, (FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[2]].data, this,1,2));
				//rs_pushStream_2(p, (FrameRingBuffer<unsigned short>*)bufferVecP[bufferIdx[1]], (FrameRingBuffer<unsigned short>*)bufferVecP[bufferIdx[1]], this, 1, 2);
			}
			else
			{
				CHECK(false) << "NOT SOPPORT TYPE";
			}

			//auto xxx = ((FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[0]].data)->pop();
			//int height = ((FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[0]].data)->height;
//			int width = ((FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[0]].data)->width;
//			int channels = ((FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[0]].data)->channels;
//#ifdef OPENCV_SHOW
//			cv::Mat show1 = cv::Mat(height, width, channels == 1 ? CV_8UC1 : CV_8UC3);
//			memcpy(show1.data, xxx, height*width*channels * sizeof(unsigned char));
//#endif

		}
		
		return 0;
	}
#endif
#ifdef USE_VIRTUALCAMERA
	void DeviceExplorer::initVirtualCamera()
	{
		CHECK(serial_numbers_.size()>0) << "the virtualCamera sn is NULL";
		std::set<std::string> serial_number_set;
		for (auto&d : serial_numbers_) serial_number_set.insert(d);
		CHECK(serial_number_set.size() == serial_numbers_.size()) << "the serial_numbers duplicated";
		bool dev_not_find = false;
		for (auto&& dev : std::vector<std::string>{ {"0"},{"1"},{"2"} })
		{
			std::string serial_number(dev);
			std::lock_guard<std::mutex> lock(_mutex);
			if (serial_numbers_.end() == std::find(serial_numbers_.begin(), serial_numbers_.end(), "virtualCamera," + serial_number))
			{
				continue;
			}
			else
			{
				std::string key_ = "virtualCamera," + serial_number;
				int rgb_h = -1, rgb_w = -1;
				int dep_h = -1, dep_w = -1;
				int inf_h = -1, inf_w = -1;
				std::unordered_map<std::string, int> streamTable;
				if (sensorInfo_.end() == std::find_if(sensorInfo_.begin(), sensorInfo_.end(), [&](auto&item)
				{
					if (key_.compare(std::get<0>(item)) != 0) return false;
					auto &this_rs_dev = std::get<1>(item);
					for (auto&map_item : this_rs_dev)
					{
						if (map_item.first.compare("rgb") == 0)
						{
							int streamIdx = std::get<0>(map_item.second);
							streamTable["rgb"] = streamIdx;
							rgb_h = std::get<1>(map_item.second);
							rgb_w = std::get<2>(map_item.second);
						}
						if (map_item.first.compare("depth") == 0)
						{
							int streamIdx = std::get<0>(map_item.second);
							streamTable["depth"] = streamIdx;
							dep_h = std::get<1>(map_item.second);
							dep_w = std::get<2>(map_item.second);
						}
						if (map_item.first.compare("infred") == 0)
						{
							int streamIdx = std::get<0>(map_item.second);
							streamTable["infred"] = streamIdx;
							inf_h = std::get<1>(map_item.second);
							inf_w = std::get<2>(map_item.second);
						}
					}
					return true;

				}))
				{
					CHECK(false) << "SN not match the jsonExplorer Info!";
				}
				CHECK(dep_h > 0 && dep_w > 0 && dep_h == inf_h && dep_w == inf_w) << "RS's depth and infred must be set the same!";
				if (rgb_h<1)
				{
					rgb_h = dep_h;
					rgb_w = dep_w;
				}
				dev_not_find = true;
				virtualCameraMap[key_] = streamTable;
			}
		}

		if (!dev_not_find)
		{
			CHECK(false) << "the serial number specified in config not match the ctx!";
		}
	}
	void DeviceExplorer::runVirtualCamera()
	{
	}
	int DeviceExplorer::pushVirtualCameraStream(std::vector<Buffer> &bufferVecP)
	{
		CHECK(bufferVecP.size()>0) << "bufferVecP must be resize before!";
		int virtualCamera_stream_cnt = 0;
		for (auto&vc : virtualCameraMap)
		{
			for (auto&stream_ : vc.second)
			{
				virtualCamera_stream_cnt++;
			}
		}
		CHECK(virtualCamera_stream_cnt <= bufferVecP.size()) << "virtualCamera stream cnt big than total stream cnt: (" << virtualCamera_stream_cnt << " : " << bufferVecP.size() << ")";
		for (auto&vc : virtualCameraMap)
		{
			auto&this_dev_info = vc.second;
			std::vector<int> bufferIdx(10, -1);//一个相机支持最多输出10个流，0位代表rgb，1位代表dep，2位代表inf
			for (auto&sensorInfo : this_dev_info)
			{
				const std::string&sensorType = sensorInfo.first;
				const int&sensorIdx = sensorInfo.second;
				int height_ = -1;
				int width_ = -1;;
				int channels_ = -1;
				std::string sensorType_ = "";
				std::string dataType_ = "";
				if (sensorInfo_.end() == std::find_if(sensorInfo_.begin(), sensorInfo_.end(), [&](auto&item)
				{
					auto&dev_map = std::get<1>(item);
					for (auto&this_sensor : dev_map)
					{
						if (sensorIdx == std::get<0>(this_sensor.second))
						{
							sensorType_ = this_sensor.first;
							height_ = std::get<1>(this_sensor.second);
							width_ = std::get<2>(this_sensor.second);
							channels_ = std::get<3>(this_sensor.second);
							dataType_ = std::get<4>(this_sensor.second);
							return true;
						}
					}
					return false;
				}))
				{
					CHECK(false) << "MATCH ERR";
				}

				CHECK(bufferVecP[sensorIdx].data == NULL) << "there should be null";
				if (dataType_.compare("uchar") == 0)
				{
					bufferVecP[sensorIdx].data = new FrameRingBuffer<unsigned char>(height_, width_, channels_);
					bufferVecP[sensorIdx].Dtype = "uchar";
				}
				else if (dataType_.compare("ushort") == 0)
				{
					bufferVecP[sensorIdx].data = new FrameRingBuffer<unsigned short>(height_, width_, channels_);
					bufferVecP[sensorIdx].Dtype = "ushort";
				}
				else
				{
					CHECK(false) << "NOT SOPPORT TYPE";
				}
				if (sensorType_.compare("rgb") == 0)
				{
					bufferIdx[0] = sensorIdx;
				}
				else if (sensorType_.compare("depth") == 0)
				{
					bufferIdx[1] = sensorIdx;
				}
				else if (sensorType_.compare("infred") == 0)
				{
					bufferIdx[2] = sensorIdx;
				}
				else
				{
					CHECK(false) << "NOT SOPPORT TYPE";
				}
			}

			//TODO:not elegent
			if (bufferIdx[0] >= 0 && bufferIdx[1] >= 0 && bufferIdx[2] >= 0)
			{
				threadSet.emplace_back(std::thread(&DeviceExplorer::virtualCamera_pushStream_3<unsigned char, unsigned short, unsigned char>, this, (FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[0]].data, (FrameRingBuffer<unsigned short>*)bufferVecP[bufferIdx[1]].data, (FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[2]].data, this));
			}
			else if (bufferIdx[0] < 0 && bufferIdx[1] >= 0 && bufferIdx[2] >= 0)
			{
				threadSet.emplace_back(std::thread(&DeviceExplorer::virtualCamera_pushStream_2<unsigned short, unsigned char>, this, (FrameRingBuffer<unsigned short>*)bufferVecP[bufferIdx[1]].data, (FrameRingBuffer<unsigned char>*)bufferVecP[bufferIdx[2]].data, this, 1, 2));
			}
			else
			{
				CHECK(false) << "NOT SOPPORT TYPE";
			}

		}

		return 0;
	}
#endif
}
