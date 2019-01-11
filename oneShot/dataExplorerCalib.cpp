#include<algorithm>
#include<chrono>
#include"stringOp.h"
#include"logg.h"
#include"iofile.h"
#include"dataExplorer.h"

#include"unreGpu.h"
#include "opencvAssistant.h"
//标定时候的数量，首先会记录rgb和inferd的点的坐标，记录calibCnt份，然后再记录calibCnt份depthmat
//整个标定的过程不能移动
const int calibCnt = 200;

namespace unre
{

	int DataExplorer::calibAllStream_noLaser()
	{
		CHECK(doCalib_) << "not the calibration model!";
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
		else {
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

		std::map<int, int>depthIdx2infredIdx;
		std::map<int, int>infredIdx2depthIdx;
		auto devicesInfo = je.getSensorAssignmentInfo();
		for (auto&this_device : devicesInfo)
		{
			auto&thisDeviceSensorMap = std::get<1>(this_device);
			if ((thisDeviceSensorMap.count("depth") == 1 && thisDeviceSensorMap.count("infred") == 0) ||
				(thisDeviceSensorMap.count("depth") == 0 && thisDeviceSensorMap.count("infred") == 1))
			{
				LOG(FATAL) << "Wrong config,the 'depth' and 'infred' must be together";
			}
			if (thisDeviceSensorMap.count("depth") == 1 && thisDeviceSensorMap.count("infred") == 1)
			{
				int depthStreamIdx = std::get<0>(thisDeviceSensorMap["depth"]);
				int infredStreamIdx = std::get<0>(thisDeviceSensorMap["infred"]);
				depthIdx2infredIdx[depthStreamIdx] = infredStreamIdx;
				infredIdx2depthIdx[infredStreamIdx] = depthStreamIdx;
			}
		}
		
		//目前不需要提前设定3d空间点，后续会从depth中算得
		std::vector<cv::Point3f> true3DPointSet;
		//float true3DPointSet_cx = 0;
		//float true3DPointSet_cy = 0;
		//for (int i = 0; i < chessBoradSize.height; i++)
		//{
		//	for (int j = 0; j < chessBoradSize.width; j++)
		//	{
		//		cv::Point3f tempPoint;
		//		tempPoint.x = (j)* chessUnitSize.width*0.001;
		//		tempPoint.y = (chessUnitSize.height - i - 1) * chessUnitSize.height*0.001;
		//		tempPoint.z = 0;// VOLUME_SIZE_Z*(0.5);
		//		true3DPointSet.push_back(tempPoint);
		//		true3DPointSet_cx += tempPoint.x;
		//		true3DPointSet_cy += tempPoint.y;
		//	}
		//}
		//true3DPointSet_cx /= true3DPointSet.size();
		//true3DPointSet_cy /= true3DPointSet.size();
		//float x_offset = VOLUME_SIZE_X*0.5 - true3DPointSet_cx;
		//float y_offset = VOLUME_SIZE_Y*0.5 - true3DPointSet_cy;
		//for (size_t i = 0; i < true3DPointSet.size(); i++)
		//{
		//	true3DPointSet[i].x += x_offset;
		//	true3DPointSet[i].y += y_offset;
		//}

		std::vector<cv::Mat*> imgs;
		initMatVect(imgs);
		std::vector<cv::Mat> grayCalibImgs(imgs.size(), cv::Mat());

		while (true)
		{
			pop2Mats(imgs);
			for (size_t i = 0; i < imgs.size(); i++)
			{
				cv::Mat imageGray;
				if (imgs[i]->type() == CV_16UC1)
				{
					continue;//depth no need calibration, rgb and infred need
				}
				if (imgs[i]->channels() == 3)
				{
					cvtColor(*imgs[i], imageGray, CV_RGB2GRAY);
					grayCalibImgs[i] = imageGray.clone();
				}
				else if (imgs[i]->channels() == 1)
				{
					imageGray = imgs[i]->clone();
					grayCalibImgs[i] = imageGray.clone();
				}
				else
				{
					LOG(FATAL) << "CALIB TYPE ERR";
				}
				imshow("streamIdx=" + std::to_string(i), imageGray);
			}
			if (cv::waitKey(1) == 'o')
			{
				bool calibDone = true;
				cv::Mat calibCooInfo = cv::Mat(chessBoradSize.height, chessBoradSize.width, CV_32FC2);
				int calibIdx = -1;
				while (calibIdx<calibCnt-1)
				{
					calibIdx++;
					pop2Mats(imgs);
					for (size_t i = 0; i < imgs.size(); i++)
					{
						cv::Mat imageGray;
						if (imgs[i]->type() == CV_16UC1)
						{
							continue;//depth no need calibration, rgb and infred need
						}
						if (imgs[i]->channels() == 3)
						{
							cvtColor(*imgs[i], imageGray, CV_RGB2GRAY);
							grayCalibImgs[i] = imageGray.clone();
						}
						else if (imgs[i]->channels() == 1)
						{
							imageGray = imgs[i]->clone();
							grayCalibImgs[i] = imageGray.clone();
						}
						else
						{
							LOG(FATAL) << "CALIB TYPE ERR";
						}
					}
					for (size_t i = 0; i < grayCalibImgs.size(); i++)
					{
						if (grayCalibImgs[i].cols < 1)
						{
							continue;
						}
						std::vector<cv::Point2f> srcCandidateCorners;
						bool patternfound = cv::findChessboardCorners(grayCalibImgs[i], chessBoradSize, srcCandidateCorners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
						if (patternfound)
						{
							cv::cornerSubPix(grayCalibImgs[i], srcCandidateCorners, cv::Size(15, 15), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
							if (checkCandidateCornersOrder(srcCandidateCorners, chessBoradSize)<0)
							{
								calibIdx--;
								LOG(INFO) << "this time calib checkCandidateCornersOrder fail.";
								continue;
							}
							cv::Mat calibCooInfo = cv::Mat(chessBoradSize.height, chessBoradSize.width, CV_32FC2);
							for (int h = 0; h < chessBoradSize.height; h++)
							{
								for (int w = 0; w < chessBoradSize.width; w++)
								{
									calibCooInfo.at<cv::Vec2f>(h, w)[0] = srcCandidateCorners[h*chessBoradSize.width + w].x;
									calibCooInfo.at<cv::Vec2f>(h, w)[1] = srcCandidateCorners[h*chessBoradSize.width + w].y;
								}
							}
							cv::FileStorage fs("./CalibData/" + std::to_string(i) + "_" + std::to_string(calibIdx) + ".yml", cv::FileStorage::WRITE);
							fs << "calibCooInfo" << calibCooInfo;
							fs.release();

							//cv::Mat intr = stream2Intr[i]->clone();
							//cv::Mat Rvect;
							//cv::Mat t;
							//cv::solvePnP(true3DPointSet, srcCandidateCorners, intr, cv::Mat::zeros(1, 5, CV_32FC1), Rvect, t);
							//cv::Mat Rmetrix;
							//cv::Rodrigues(Rvect, Rmetrix);
							//Rmetrix.copyTo(std::get<0>(stream2Extr[i]));
							//t.copyTo(std::get<1>(stream2Extr[i]));
							//LOG(INFO) << Rmetrix;
							//LOG(INFO) << t;
							//cv::imwrite(std::to_string(i) + ".jpg", grayCalibImgs[i]);
						}
						else
						{
							calibIdx--;
							LOG(INFO) << "this time calib point found fail.";
							break;
							//calibDone = false;
							//break;
						}
					}

				}
				if (calibDone)
				{
					break;
				}
			}
			else if (cv::waitKey(1) == 'c')
			{
				
			}
			else
			{
				pop2Mats(imgs);//多弹几张图，避免队列的慢放，因为找内点已经够慢了
			}
		}
		
		return 0;
	}


	int DataExplorer::calibAllStream_Laser()
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
		else {
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

		std::map<int, int>depthIdx2infredIdx;
		std::map<int, int>infredIdx2depthIdx;
		auto devicesInfo = je.getSensorAssignmentInfo();
		for (auto&this_device : devicesInfo)
		{
			auto&thisDeviceSensorMap = std::get<1>(this_device);
			if ((thisDeviceSensorMap.count("depth") == 1 && thisDeviceSensorMap.count("infred") == 0) ||
				(thisDeviceSensorMap.count("depth") == 0 && thisDeviceSensorMap.count("infred") == 1))
			{
				LOG(FATAL) << "Wrong config,the 'depth' and 'infred' must be together";
			}
			if (thisDeviceSensorMap.count("depth") == 1 && thisDeviceSensorMap.count("infred") == 1)
			{
				int depthStreamIdx = std::get<0>(thisDeviceSensorMap["depth"]);
				int infredStreamIdx = std::get<0>(thisDeviceSensorMap["infred"]);
				depthIdx2infredIdx[depthStreamIdx] = infredStreamIdx;
				infredIdx2depthIdx[infredStreamIdx] = depthStreamIdx;
			}
		}


		std::vector<cv::Mat*> imgs;
		initMatVect(imgs);
		std::vector<cv::Mat> grayCalibImgs(imgs.size(), cv::Mat());

		for (int calibIdx = 0; calibIdx < calibCnt; calibIdx++)
		{
			pop2Mats(imgs);
			cv::Mat imageGray;
			for (size_t i = 0; i < imgs.size(); i++)
			{
				if (imgs[i]->type() == CV_16UC1)
				{
					//only needs this
					cv::Mat test = imgs[i]->clone();
					cv::FileStorage fs("./CalibData/"+ std::to_string(i)+"_" + std::to_string(calibIdx) + ".yml", cv::FileStorage::WRITE);
					fs << "depthMat" << (*imgs[i]);
					fs.release();
				}
				else
				{
					continue;
				}
			}
		}

		return 0;
	}


	int DataExplorer::calibData()//目前只是支持3个sensor： 1个rgb1个deep1个inferd
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
		else {
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

		std::map<int, int>depthIdx2infredIdx;
		std::map<int, int>infredIdx2depthIdx;
		auto devicesInfo = je.getSensorAssignmentInfo();
		for (auto&this_device : devicesInfo)
		{
			auto&thisDeviceSensorMap = std::get<1>(this_device);
			if ((thisDeviceSensorMap.count("depth") == 1 && thisDeviceSensorMap.count("infred") == 0) ||
				(thisDeviceSensorMap.count("depth") == 0 && thisDeviceSensorMap.count("infred") == 1))
			{
				LOG(FATAL) << "Wrong config,the 'depth' and 'infred' must be together";
			}
			if (thisDeviceSensorMap.count("depth") == 1 && thisDeviceSensorMap.count("infred") == 1)
			{
				int depthStreamIdx = std::get<0>(thisDeviceSensorMap["depth"]);
				int infredStreamIdx = std::get<0>(thisDeviceSensorMap["infred"]);
				depthIdx2infredIdx[depthStreamIdx] = infredStreamIdx;
				infredIdx2depthIdx[infredStreamIdx] = depthStreamIdx;
			}
		}

		std::vector<cv::Mat*> imgs;
		initMatVect(imgs);
		std::vector<cv::Mat> grayCalibImgs(imgs.size(), cv::Mat());
		for (size_t i = 0; i < imgs.size(); i++)
		{
			for (size_t j = 0; j < calibCnt; j++)
			{
				std::string thisCalibFilePath = "./CalibData/" + std::to_string(i) + "_" + std::to_string(j) + ".yml";
				CHECK(unre::FileOP::FileExist(thisCalibFilePath)) << "MISSING CalibFilePath";
			}
		}
		std::vector<long long*> calibData(imgs.size(), NULL);//坐标点保留1位小数，然后*10，存入long long
		for (size_t i = 0; i < imgs.size(); i++)
		{
			if (imgs[i]->type() == CV_16UC1)
			{
				calibData[i] = new long long[imgs[i]->rows*imgs[i]->cols];
				memset(calibData[i],0, imgs[i]->rows*imgs[i]->cols *sizeof(long long));
			}
			else
			{
				calibData[i] = new long long[chessBoradSize.height*chessBoradSize.width * 2];
				memset(calibData[i], 0, chessBoradSize.height*chessBoradSize.width * 2 * sizeof(long long));
			}
			for (size_t j = 0; j < calibCnt; j++)
			{
				if (imgs[i]->type() == CV_16UC1)
				{
					cv::Mat depthMat;
					cv::FileStorage fs("./CalibData/" + std::to_string(i) + "_" + std::to_string(j) + ".yml", cv::FileStorage::READ);
					fs["depthMat"] >> depthMat;
					fs.release();
					for (int eleIdx = 0; eleIdx < depthMat.rows*depthMat.cols; eleIdx++)
					{
						calibData[i][eleIdx] += ((unsigned short*)depthMat.data)[eleIdx];
					}
				}
				else
				{
					cv::Mat coos;
					cv::FileStorage fs("./CalibData/" + std::to_string(i) + "_" + std::to_string(j) + ".yml", cv::FileStorage::READ);
					fs["calibCooInfo"] >> coos;
					fs.release();
					for (int eleIdx = 0; eleIdx < chessBoradSize.height*chessBoradSize.width; eleIdx++)
					{
						int u_ = eleIdx%chessBoradSize.width;
						int v_ = eleIdx/chessBoradSize.width;
						calibData[i][2 * eleIdx] += static_cast<long long>(coos.at<cv::Vec2f>(v_, u_)[0] * 10);
						calibData[i][2 * eleIdx+1] += static_cast<long long>(coos.at<cv::Vec2f>(v_, u_)[1] * 10);
					}
				}
			}
		}
		
		{
			CHECK(imgs.size() == 3) << "SUPPORT NOLY THIS MODEL";
			std::vector<cv::Point2f> rgb_points;
			std::vector<cv::Point2f> inferd_points;
			std::vector<cv::Point3f> gt_points;
			for (size_t i = 0; i < imgs.size(); i++)
			{
				if (imgs[i]->type() == CV_8UC3)
				{
					for (int eleIdx = 0; eleIdx < chessBoradSize.height*chessBoradSize.width; eleIdx++)
					{
						int u_ = eleIdx%chessBoradSize.width;
						int v_ = eleIdx/chessBoradSize.height;
						float thisPoint_x = calibData[i][2 * eleIdx] * 0.1 / calibCnt;
						float thisPoint_y = calibData[i][2 * eleIdx + 1] * 0.1 / calibCnt;
						rgb_points.push_back(cv::Point2f(thisPoint_x, thisPoint_y));
					}
				}
				else if (imgs[i]->type() == CV_8UC1)
				{
					for (int eleIdx = 0; eleIdx < chessBoradSize.height*chessBoradSize.width; eleIdx++)
					{
						int u_ = eleIdx%chessBoradSize.width;
						int v_ = eleIdx/chessBoradSize.height;
						float thisPoint_x = calibData[i][2 * eleIdx] * 0.1 / calibCnt;
						float thisPoint_y = calibData[i][2 * eleIdx + 1] * 0.1 / calibCnt;
						inferd_points.push_back(cv::Point2f(thisPoint_x, thisPoint_y));
					}
				}
			}
			for (size_t i = 0; i < imgs.size(); i++)
			{			
				if (imgs[i]->type() == CV_16UC1)
				{
					auto intr = stream2Intr[i];
					for (int p_idx = 0; p_idx < inferd_points.size(); p_idx++)
					{
						int infred_x = static_cast<float>(inferd_points[p_idx].x) + .5;
						int infred_y = static_cast<float>(inferd_points[p_idx].y) + .5;
						int pos = infred_y*imgs[i]->cols + infred_x;
						float z = 0.001*static_cast<float>(1.0 / calibCnt*calibData[i][pos]);
						float x = z * (infred_x - intr->ptr<double>(0)[2]) / intr->ptr<double>(0)[0];
						float y = z * (infred_y - intr->ptr<double>(1)[2]) / intr->ptr<double>(1)[1];
						gt_points.push_back(cv::Point3f(x,y,z));
					}				
				}
			}
			for (size_t i = 0; i < imgs.size(); i++)
			{
				if (imgs[i]->type() == CV_8UC3)
				{
					cv::Mat intr = stream2Intr[i]->clone();
					cv::Mat Rvect;
					cv::Mat t;
					cv::solvePnP(gt_points, rgb_points, intr, cv::Mat::zeros(1, 5, CV_32FC1), Rvect, t);
					cv::Mat Rmetrix;
					cv::Rodrigues(Rvect, Rmetrix);
					Rmetrix.copyTo(std::get<0>(stream2Extr[i]));
					t.copyTo(std::get<1>(stream2Extr[i]));
					LOG(INFO) << Rmetrix;
					LOG(INFO) << t;
				}
				if (imgs[i]->type() == CV_8UC1)
				{
					//对于realsense，用depth的点来标定infred的点，得到的rt，理论上都是 eye和zeros
					cv::Mat eye_ = cv::Mat::eye(3, 3, CV_64FC1);
					cv::Mat zeros_ = cv::Mat::zeros(3, 1, CV_64FC1);
					eye_.copyTo(std::get<0>(stream2Extr[i]));
					zeros_.copyTo(std::get<1>(stream2Extr[i]));
					continue;
					cv::Mat intr = stream2Intr[i]->clone();
					cv::Mat Rvect;
					cv::Mat t;
					cv::solvePnP(gt_points, inferd_points, intr, cv::Mat::zeros(1, 5, CV_32FC1), Rvect, t);
					cv::Mat Rmetrix;
					cv::Rodrigues(Rvect, Rmetrix);
					Rmetrix.copyTo(std::get<0>(stream2Extr[i]));
					t.copyTo(std::get<1>(stream2Extr[i]));
				}
			}
			//{
			//	cv::Mat drawPlane = cv::Mat::zeros(1080,1920,CV_8UC3);
			//	for (size_t i = 0; i < rgb_points.size(); i++)
			//	{
			//		cv::circle(drawPlane, rgb_points[i], 3, cv::Scalar(255,0,0),2);
			//		cv::circle(drawPlane, inferd_points[i], 3, cv::Scalar(0,255,0),2);
			//		LOG(INFO) << gt_points[i];
			//		LOG(INFO) << "";
			//	}
			//}
		}
		{
			//将infred的外参写给depth！！！
			//std::map<int, int>depthIdx2infredIdx;
			//std::map<int, int>infredIdx2depthIdx;
			for (size_t extrIdx = 0; extrIdx < stream2Extr.size(); extrIdx++)
			{
				if (infredIdx2depthIdx.count(extrIdx) == 1)
				{
					int infredIdx = extrIdx;
					int depthIdx = infredIdx2depthIdx[extrIdx];
					std::get<0>(stream2Extr[infredIdx]).copyTo(std::get<0>(stream2Extr[depthIdx]));
					std::get<1>(stream2Extr[infredIdx]).copyTo(std::get<1>(stream2Extr[depthIdx]));
				}
			}

		}
		{
			rapidjson::Document::AllocatorType& allocator = calibDocRoot.GetAllocator();
			rapidjson::Value info_array(rapidjson::kArrayType);
			for (int i = 0; i < stream2Extr.size(); i++) {
				rapidjson::Value info_object(rapidjson::kObjectType);
				info_object.SetObject();
				info_object.AddMember("streamIdx", i, allocator);

				rapidjson::Value R_array(rapidjson::kArrayType);
				info_object.AddMember("R_rows", std::get<0>(stream2Extr[i]).rows, allocator);
				info_object.AddMember("R_cols", std::get<0>(stream2Extr[i]).cols, allocator);
				for (int r = 0; r < std::get<0>(stream2Extr[i]).rows; r++) {
					for (int c = 0; c < std::get<0>(stream2Extr[i]).cols; c++) {
						R_array.PushBack(std::get<0>(stream2Extr[i]).ptr<double>(r)[c], allocator);
					}
				}
				info_object.AddMember("R", R_array, allocator);

				rapidjson::Value t_array(rapidjson::kArrayType);
				info_object.AddMember("t_rows", std::get<1>(stream2Extr[i]).rows, allocator);
				info_object.AddMember("t_cols", std::get<1>(stream2Extr[i]).cols, allocator);
				for (int r = 0; r < std::get<1>(stream2Extr[i]).rows; r++) {
					for (int c = 0; c < std::get<1>(stream2Extr[i]).cols; c++) {
						t_array.PushBack(std::get<1>(stream2Extr[i]).ptr<double>(r)[c], allocator);
					}
				}
				info_object.AddMember("t", t_array, allocator);

				info_array.PushBack(info_object, allocator);

			}

			//calibDocRoot.AddMember("caeraExtParams", info_array, allocator);
			calibDocRoot["caeraExtParams"] = info_array;
			//calibDocRoot["caeraExtParams"].Set(info_array);


			// 3. Stringify the DOM
			rapidjson::StringBuffer buffer;
			rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
			calibDocRoot.Accept(writer);
			std::fstream fout("calib.json", std::ios::out);
			fout << buffer.GetString() << std::endl;
			fout.close();
		}
		return 0;
	}

}