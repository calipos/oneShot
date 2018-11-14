#include<algorithm>
#include<chrono>
#include"stringOp.h"
#include"logg.h"
#include"iofile.h"
#include"dataExplorer.h"


#ifdef OPENCV_SHOW
#include "opencv2/opencv.hpp"
#endif

namespace unre
{
	DataExplorer::DataExplorer(int streamNum,bool doCalib)
	{
		doCalib_ = doCalib;
		std::string theInitFile;
		if (FileOP::FileExist("config.json"))
		{
			je = JsonExplorer("config.json");
			theInitFile = "config.json";
		}
		else if (FileOP::FileExist("config_default.json"))
		{
			je = JsonExplorer("config_default.json");
			theInitFile = "config_default.json";
		}
		else
		{
			CHECK(false) << "no config.json, even config_default.jsonㄐㄐㄐ";
		}
		int tmp_exactStreamCnt = 0;
		auto &SensorsInfo = je.getSensorAssignmentInfo();
		std::vector<std::string> usedDeviceType;
		std::find_if(SensorsInfo.begin(), SensorsInfo.end(), [&usedDeviceType, &tmp_exactStreamCnt](auto&item)
		{
			const std::string &this_dev_type = std::get<0>(item);
			if (usedDeviceType.end() == std::find(usedDeviceType.begin(), usedDeviceType.end(), this_dev_type))
			{
				usedDeviceType.emplace_back(this_dev_type);
				LOG(INFO) << this_dev_type << " HAS BEEN SET.";
			}
			tmp_exactStreamCnt += std::get<1>(item).size();
			return false;
		});
		exactStreamCnt = tmp_exactStreamCnt;
		LOG(INFO) << SensorsInfo.size() << " devices are required.";
		
		dev_e = new DeviceExplorer(theInitFile, usedDeviceType, SensorsInfo, je.getExtraConfigFilPath(), doCalib_);
		dev_e->init();
		dev_e->run();
		je = JsonExplorer(theInitFile.c_str());//reload json,因为上一行dev_e->run()会把json文件更新到实际inrt
		//std::unordered_map<int, cv::Mat*> stream2Intr;
		//std::unordered_map<int, std::tuple<cv::Mat*, cv::Mat*>> stream2Extr;
		for (auto&dev : je.getSensorAssignmentInfo())
		{
			for (auto&sensor : std::get<1>(dev))
			{
				const int& streamIdx = std::get<0>(sensor.second);
				std::unordered_map<std::string, double> intrMap = std::get<5>(sensor.second);
				stream2Intr[streamIdx] = new cv::Mat(3, 3, CV_64FC1);
				stream2Intr[streamIdx]->ptr<double>(0)[0] = intrMap["fx"];
				stream2Intr[streamIdx]->ptr<double>(1)[1] = intrMap["fy"];
				stream2Intr[streamIdx]->ptr<double>(0)[2] = intrMap["cx"];
				stream2Intr[streamIdx]->ptr<double>(1)[2] = intrMap["cy"];
				stream2Intr[streamIdx]->ptr<double>(2)[2] = 1.;
			}
		}
		bufferVecP.resize(exactStreamCnt);//必须相等，因为后边和遍历check和pop
		constBuffer.resize(exactStreamCnt);
		for (auto&item : bufferVecP)  item = Buffer();
		//dev_e->initalConstBuffer(constBuffer);
		dev_e->pushStream(bufferVecP);//before push, the [bufferVecP] has be initalized in this function
		//calibAllStream();


	}

	int DataExplorer::getExactStreamCnt()
	{
		return exactStreamCnt;
	}

	int DataExplorer::getBuffer_fortest()
	{
		for (auto&item : bufferVecP)  CHECK(item.data) << "null data is disallowed!";
		int height1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->height;
		int width1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->width;
		int channels1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->channels;
		int height2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->height;
		int width2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->width;
		int channels2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->channels;
		int height3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->height;
		int width3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->width;
		int channels3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->channels;
		int height4 = ((FrameRingBuffer<unsigned char>*)bufferVecP[3].data)->height;
		int width4 = ((FrameRingBuffer<unsigned char>*)bufferVecP[3].data)->width;
		int channels4 = ((FrameRingBuffer<unsigned char>*)bufferVecP[3].data)->channels;
		int height5 = ((FrameRingBuffer<unsigned short>*)bufferVecP[4].data)->height;
		int width5 = ((FrameRingBuffer<unsigned short>*)bufferVecP[4].data)->width;
		int channels5 = ((FrameRingBuffer<unsigned short>*)bufferVecP[4].data)->channels;
		int height6 = ((FrameRingBuffer<unsigned char>*)bufferVecP[5].data)->height;
		int width6 = ((FrameRingBuffer<unsigned char>*)bufferVecP[5].data)->width;
		int channels6 = ((FrameRingBuffer<unsigned char>*)bufferVecP[5].data)->channels;
		int height7 = ((FrameRingBuffer<unsigned char>*)bufferVecP[6].data)->height;
		int width7 = ((FrameRingBuffer<unsigned char>*)bufferVecP[6].data)->width;
		int channels7 = ((FrameRingBuffer<unsigned char>*)bufferVecP[6].data)->channels;
		int height8 = ((FrameRingBuffer<unsigned short>*)bufferVecP[7].data)->height;
		int width8 = ((FrameRingBuffer<unsigned short>*)bufferVecP[7].data)->width;
		int channels8 = ((FrameRingBuffer<unsigned short>*)bufferVecP[7].data)->channels;
		int height9 = ((FrameRingBuffer<unsigned char>*)bufferVecP[8].data)->height;
		int width9 = ((FrameRingBuffer<unsigned char>*)bufferVecP[8].data)->width;
		int channels9 = ((FrameRingBuffer<unsigned char>*)bufferVecP[8].data)->channels;
#ifdef OPENCV_SHOW
		cv::Mat show1 = cv::Mat(height1, width1, channels1 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show2 = cv::Mat(height2, width2, channels2 == 1 ? CV_16UC1 : CV_16UC3);
		cv::Mat show3 = cv::Mat(height3, width3, channels3 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show4 = cv::Mat(height4, width4, channels4 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show5 = cv::Mat(height5, width5, channels5 == 1 ? CV_16UC1 : CV_16UC3);
		cv::Mat show6 = cv::Mat(height6, width6, channels6 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show7 = cv::Mat(height7, width7, channels7 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show8 = cv::Mat(height8, width8, channels8 == 1 ? CV_16UC1 : CV_16UC3);
		cv::Mat show9 = cv::Mat(height9, width9, channels9 == 1 ? CV_8UC1 : CV_8UC3);
#endif
		while (true)
		{
			auto xxx = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->pop(show1.data);
			auto yyy = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->pop(show2.data);
			auto zzz = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->pop(show3.data);
			auto xxx2 = ((FrameRingBuffer<unsigned char>*)bufferVecP[3].data)->pop(show4.data);
			auto yyy2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[4].data)->pop(show5.data);
			auto zzz2 = ((FrameRingBuffer<unsigned char>*)bufferVecP[5].data)->pop(show6.data);
			auto xxx3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[6].data)->pop(show7.data);
			auto yyy3 = ((FrameRingBuffer<unsigned short>*)bufferVecP[7].data)->pop(show8.data);
			auto zzz3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[8].data)->pop(show9.data);
#ifdef OPENCV_SHOW
			cv::imshow("1", show1);
			cv::imshow("2", show2);
			cv::imshow("3", show3);
			cv::imshow("4", show4);
			cv::imshow("5", show5);
			cv::imshow("6", show6);
			cv::imshow("7", show7);
			cv::imshow("8", show8);
			cv::imshow("9", show9);
			int key = cv::waitKey(12);
			if (key == 'a')
			{
				dev_e->pauseThread();
				for (size_t i = 0; i < 10; i++)
				{
					std::this_thread::sleep_for(std::chrono::seconds(1));
					LOG(INFO) << "WAIT 1s";

				}
				dev_e->continueThread();
			}
			else if (key == 'q')
			{
				dev_e->terminateThread();
				cv::destroyAllWindows();
				break;
			}
#endif		
		}
		return 0;
	}

	int DataExplorer::getBuffer_fortest3()
	{
		for (auto&item : bufferVecP)  CHECK(item.data) << "null data is disallowed!";
		int height1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->height;
		int width1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->width;
		int channels1 = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->channels;
		int height2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->height;
		int width2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->width;
		int channels2 = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->channels;
		int height3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->height;
		int width3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->width;
		int channels3 = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->channels;
#ifdef OPENCV_SHOW
		cv::Mat show1 = cv::Mat(height1, width1, channels1 == 1 ? CV_8UC1 : CV_8UC3);
		cv::Mat show2 = cv::Mat(height2, width2, channels2 == 1 ? CV_16UC1 : CV_16UC3);
		cv::Mat show3 = cv::Mat(height3, width3, channels3 == 1 ? CV_8UC1 : CV_8UC3);
#endif
		int time__ = 50;
		while (time__--)
		{
			auto xxx = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->pop(show1.data);
			auto yyy = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->pop(show2.data);
			auto zzz = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->pop(show3.data);

#ifdef OPENCV_SHOW
			cv::imshow("1", show1);
			cv::imshow("2", show2);
			cv::imshow("3", show3);

			int key = cv::waitKey(12);
			if (key == 'a')
			{
				dev_e->pauseThread();
				for (size_t i = 0; i < 10; i++)
				{
					std::this_thread::sleep_for(std::chrono::seconds(1));
					LOG(INFO) << "WAIT 1s";

				}
				dev_e->continueThread();
			}
			else if (key == 'q')
			{
				dev_e->terminateThread();
				cv::destroyAllWindows();
				
					for (auto&thre : dev_e->threadSet)
					{
						if (thre.joinable())
						{
							thre.join();
						}
					}
				break;
			}
#endif		
		}
		return 0;
	}

	const std::vector<Buffer>&DataExplorer::getBufferVecP()
	{
		return bufferVecP;
	}

	const std::vector<std::tuple<std::string, oneDevMap> >&DataExplorer::getStreamInfo()
	{
		return je.getSensorAssignmentInfo();
	}

	int DataExplorer::initMatVect(std::vector<cv::Mat*>&imgs)
	{
		imgs.clear();
		imgs.resize(exactStreamCnt);
		for (auto&dev : je.getSensorAssignmentInfo())
		{
			const std::string&sn = std::get<0>(dev);
			const unre::oneDevMap&sensors = std::get<1>(dev);
			for (auto&sensor : sensors)
			{
				const std::string&sensorType = std::get<0>(sensor);
				const int&streamIdx = std::get<0>(std::get<1>(sensor));
				const int&height = std::get<1>(std::get<1>(sensor));
				const int&width = std::get<2>(std::get<1>(sensor));
				const int&channels = std::get<3>(std::get<1>(sensor));
				const std::string&dtype = std::get<4>(std::get<1>(sensor));
				auto&intrMap = std::get<5>(std::get<1>(sensor));

				if (sensorType.compare("rgb") == 0)
				{
					if (dtype.compare("uchar") == 0 && channels == 3)
					{
						if (imgs[streamIdx] != 0)
						{
							imgs[streamIdx]->release();
						}
						imgs[streamIdx] = new cv::Mat(height, width, CV_8UC3);
					}
					else
					{
						LOG(FATAL) << "NOT SUPPORT TYPE";
					}
				}
				else if (sensorType.compare("depth") == 0)
				{
					if (dtype.compare("ushort") == 0 && channels == 1)
					{
						if (imgs[streamIdx] != 0)
						{
							imgs[streamIdx]->release();
						}
						imgs[streamIdx] = new cv::Mat(height, width, CV_16UC1);
					}
					else
					{
						LOG(FATAL) << "NOT SUPPORT TYPE";
					}
				}
				else if (sensorType.compare("infred") == 0)
				{
					if (dtype.compare("uchar") == 0 && channels == 1)
					{
						if (imgs[streamIdx] != 0)
						{
							imgs[streamIdx]->release();
						}
						imgs[streamIdx] = new cv::Mat(height, width, CV_8UC1);
					}
					else
					{
						LOG(FATAL) << "NOT SUPPORT TYPE";
					}
				}

			}
		}

		return 0;
	}

	int DataExplorer::pop2Mats(std::vector<cv::Mat*>&imgs)
	{
		CHECK(imgs.size() != 0);

		for (auto&dev : je.getSensorAssignmentInfo())
		{
			const std::string&sn = std::get<0>(dev);
			const unre::oneDevMap&sensors = std::get<1>(dev);
			for (auto&sensor : sensors)
			{
				const std::string&sensorType = std::get<0>(sensor);
				const int&streamIdx = std::get<0>(std::get<1>(sensor));
				const int&height = std::get<1>(std::get<1>(sensor));
				const int&width = std::get<2>(std::get<1>(sensor));
				const int&channels = std::get<3>(std::get<1>(sensor));
				const std::string&dtype = std::get<4>(std::get<1>(sensor));
				auto&intrMap = std::get<5>(std::get<1>(sensor));

				if (sensorType.compare("rgb") == 0)
				{
					if (dtype.compare("uchar") == 0 && channels == 3)
					{
						((unre::FrameRingBuffer<unsigned char>*)bufferVecP[streamIdx].data)->pop(imgs[streamIdx]->data);
					}
					else
					{
						LOG(FATAL) << "NOT SUPPORT TYPE";
					}
				}
				else if (sensorType.compare("depth") == 0)
				{
					if (dtype.compare("ushort") == 0 && channels == 1)
					{
						//((unre::FrameRingBuffer<unsigned short>*)bufferVecP[streamIdx].data)->pop(imgs[streamIdx]->data);
						((unre::FrameRingBuffer<unsigned short>*)bufferVecP[streamIdx].data)->pop();//depth neednt calib
					}
					else
					{
						LOG(FATAL) << "NOT SUPPORT TYPE";
					}
				}
				else if (sensorType.compare("infred") == 0)
				{
					if (dtype.compare("uchar") == 0 && channels == 1)
					{
						((unre::FrameRingBuffer<unsigned char>*)bufferVecP[streamIdx].data)->pop(imgs[streamIdx]->data);
					}
					else
					{
						LOG(FATAL) << "NOT SUPPORT TYPE";
					}
				}

			}
		}

		return 0;
	}

	int DataExplorer::calibAllStream()
	{
		if (!FileOP::FileExist("calib.json"))
		{
			LOG(ERROR) << "the calib.json file not exist! cannot get the chessborad parameters!";
			return UNRE_CALIB_FILE_ERROR;
		}
		rapidjson::Document calibDocRoot;
		calibDocRoot.Parse<0>(unre::StringOP::parseJsonFile2str("calib.json").c_str());
		cv::Size2f chessUnitSize;
		if (calibDocRoot.HasMember("chessUnitSize"))
		{
			float this_value = calibDocRoot["chessUnitSize"].GetFloat();
			chessUnitSize = cv::Size2f(this_value, this_value);
		}
		else if (calibDocRoot.HasMember("chessUnitSize_h") && calibDocRoot.HasMember("chessUnitSize_w"))
		{
			float this_value1 = calibDocRoot["chessUnitSize_w"].GetFloat();
			float this_value2 = calibDocRoot["chessUnitSize_h"].GetFloat();
			chessUnitSize = cv::Size2f(this_value1, this_value2);
		}
		else{
			LOG(ERROR) << "important parameter missing!!";
			return UNRE_CALIB_FILE_ERROR;
		}
		cv::Size chessBoradSize;
		if (calibDocRoot.HasMember("chessBoard_h") && calibDocRoot.HasMember("chessBoard_w"))
		{
			float this_value1 = calibDocRoot["chessBoard_w"].GetFloat();
			float this_value2 = calibDocRoot["chessBoard_h"].GetFloat();
			chessBoradSize = cv::Size2f(this_value1, this_value2);
		}
		else {
			LOG(ERROR) << "important parameter missing!!";
			return UNRE_CALIB_FILE_ERROR;
		}
		
		std::vector<cv::Point3f> true3DPointSet;
		for (int i = 0; i < chessBoradSize.height; i++)
		{
			for (int j = 0; j < chessBoradSize.width; j++)
			{
				cv::Point3f tempPoint;
				tempPoint.x = j * chessUnitSize.width;
				tempPoint.y = i * chessUnitSize.height;
				tempPoint.z = 0;
				true3DPointSet.push_back(tempPoint);
			}
		}

		std::vector<cv::Mat*> imgs;
		initMatVect(imgs);
		while (true)
		{


			pop2Mats(imgs);
			for (size_t i = 0; i < imgs.size(); i++)
			{
				cv::Mat imageGray;
				if (imgs[i]->type() == CV_16UC1)
				{
					continue;//depth no need calibration, rgb and infred need
					double min_, max_;
					cv::minMaxLoc(imageGray, &min_, &max_, NULL, NULL);;
					imageGray.convertTo(imageGray, CV_32FC1);
					imageGray = (imageGray - min_) / (max_ - min_)*255.;
					imageGray.convertTo(imageGray, CV_8UC1);
				}
				if (imgs[i]->channels() == 3)
				{
					cvtColor(*imgs[i], imageGray, CV_RGB2GRAY);
				}
				else if (imgs[i]->channels() == 1)
				{
					imageGray = imgs[i]->clone();
				}
				else
				{
					LOG(FATAL) << "CALIB TYPE ERR";
				}
				
				std::vector<cv::Point2f> srcCandidateCorners;
				imshow("123", *imgs[i]);
				cv::waitKey(1);
				bool patternfound = cv::findChessboardCorners(imageGray, cv::Size(6, 9), srcCandidateCorners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
				if (patternfound)
				{
					cv::cornerSubPix(imageGray, srcCandidateCorners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
					cv::Mat intr = stream2Intr[i]->clone();
					cv::Mat Rvect;
					cv::Mat t;
					cv::solvePnP(true3DPointSet, srcCandidateCorners, intr, cv::Mat::zeros(1, 5, CV_32FC1), Rvect, t);
					cv::Mat Rmetrix;
					cv::Rodrigues(Rvect, Rmetrix);
					std::get<0>(stream2Extr[i]) = new cv::Mat(3, 3, CV_64FC1);
					std::get<1>(stream2Extr[i]) = new cv::Mat(3, 1, CV_64FC1);
					Rmetrix.copyTo(*std::get<0>(stream2Extr[i]));
					t.copyTo(*std::get<1>(stream2Extr[i]));
				}
				else
				{					
					//continue;
					//std::cout << "Detect Failed.\n";
				}	
			}
			pop2Mats(imgs);//多弹几张图，避免队列的慢放，因为找内点已经够慢了
			pop2Mats(imgs);//
			pop2Mats(imgs);//
		}
		return 0;
	}
}