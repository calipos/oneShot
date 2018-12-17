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
const int calibCnt = 20;

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

		std::vector<cv::Point3f> true3DPointSet;
		float true3DPointSet_cx = 0;
		float true3DPointSet_cy = 0;
		for (int i = 0; i < chessBoradSize.height; i++)
		{
			for (int j = 0; j < chessBoradSize.width; j++)
			{
				cv::Point3f tempPoint;
				tempPoint.x = (j)* chessUnitSize.width*0.001;
				tempPoint.y = (chessUnitSize.height - i - 1) * chessUnitSize.height*0.001;
				tempPoint.z = 0;// VOLUME_SIZE_Z*(0.5);
				true3DPointSet.push_back(tempPoint);
				true3DPointSet_cx += tempPoint.x;
				true3DPointSet_cy += tempPoint.y;
			}
		}
		true3DPointSet_cx /= true3DPointSet.size();
		true3DPointSet_cy /= true3DPointSet.size();
		float x_offset = VOLUME_SIZE_X*0.5 - true3DPointSet_cx;
		float y_offset = VOLUME_SIZE_Y*0.5 - true3DPointSet_cy;
		for (size_t i = 0; i < true3DPointSet.size(); i++)
		{
			true3DPointSet[i].x += x_offset;
			true3DPointSet[i].y += y_offset;
		}

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
					double min_, max_;
					cv::minMaxLoc(imageGray, &min_, &max_, NULL, NULL);;
					imageGray.convertTo(imageGray, CV_32FC1);
					imageGray = (imageGray - min_) / (max_ - min_)*255.;
					imageGray.convertTo(imageGray, CV_8UC1);
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
							double min_, max_;
							cv::minMaxLoc(imageGray, &min_, &max_, NULL, NULL);;
							imageGray.convertTo(imageGray, CV_32FC1);
							imageGray = (imageGray - min_) / (max_ - min_)*255.;
							imageGray.convertTo(imageGray, CV_8UC1);
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
							checkCandidateCornersOrder(srcCandidateCorners, chessBoradSize);

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
				//for test
				bool calibDone = true;
				for (size_t i = 0; i < grayCalibImgs.size(); i++)
				{
					if (grayCalibImgs[i].cols<1)
					{
						continue;
					}

					cv::Mat testCalibImg = cv::imread(std::to_string(i) + ".jpg", 0);
					std::vector<cv::Point2f> srcCandidateCorners;
					bool patternfound = cv::findChessboardCorners(testCalibImg, chessBoradSize, srcCandidateCorners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
					if (patternfound)
					{
						cv::cornerSubPix(testCalibImg, srcCandidateCorners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
						checkCandidateCornersOrder(srcCandidateCorners, chessBoradSize);
						cv::Mat intr = stream2Intr[i]->clone();
						cv::Mat Rvect;
						cv::Mat t;
						cv::solvePnP(true3DPointSet, srcCandidateCorners, intr, cv::Mat::zeros(1, 5, CV_32FC1), Rvect, t);
						cv::Mat Rmetrix;
						cv::Rodrigues(Rvect, Rmetrix);
						Rmetrix.copyTo(std::get<0>(stream2Extr[i]));
						//t *= 0.001;
						t.copyTo(std::get<1>(stream2Extr[i]));
					}
					else
					{
						calibDone = false;
						break;
					}
				}
				if (calibDone)
				{
					break;
				}
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

		std::vector<cv::Point3f> true3DPointSet;
		float true3DPointSet_cx = 0;
		float true3DPointSet_cy = 0;
		for (int i = 0; i < chessBoradSize.height; i++)
		{
			for (int j = 0; j < chessBoradSize.width; j++)
			{
				cv::Point3f tempPoint;
				tempPoint.x = (j)* chessUnitSize.width*0.001;
				tempPoint.y = (chessUnitSize.height - i - 1) * chessUnitSize.height*0.001;
				tempPoint.z = 0;// VOLUME_SIZE_Z*(0.5);
				true3DPointSet.push_back(tempPoint);
				true3DPointSet_cx += tempPoint.x;
				true3DPointSet_cy += tempPoint.y;
			}
		}
		true3DPointSet_cx /= true3DPointSet.size();
		true3DPointSet_cy /= true3DPointSet.size();
		float x_offset = VOLUME_SIZE_X*0.5 - true3DPointSet_cx;
		float y_offset = VOLUME_SIZE_Y*0.5 - true3DPointSet_cy;
		for (size_t i = 0; i < true3DPointSet.size(); i++)
		{
			true3DPointSet[i].x += x_offset;
			true3DPointSet[i].y += y_offset;
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
		std::vector<cv::Mat> calibData(imgs.size(), cv::Mat());
		{
			for (size_t i = 0; i < imgs.size(); i++)
			{
				for (size_t j = 0; j < calibCnt; j++)
				{
					if (imgs[i]->type() == CV_16UC1)
					{
						cv::Mat depthMat;
						cv::FileStorage fs("./CalibData/" + std::to_string(i) + "_" + std::to_string(j) + ".yml", cv::FileStorage::READ);
						fs["depthMat"] >> depthMat;
						fs.release();
						if (j == 0)
						{
							calibData[i] = depthMat.clone();
							calibData[i].convertTo(calibData[i], CV_64FC1);
						}
						else
						{
							cv::Mat temp;
							depthMat.convertTo(temp, CV_64FC1);
							calibData[i] += temp;
						}
					}
					else
					{
						cv::Mat coos;
						cv::FileStorage fs("./CalibData/" + std::to_string(i) + "_" + std::to_string(j) + ".yml", cv::FileStorage::READ);
						fs["calibCooInfo"] >> coos;
						fs.release();
						if (j == 0)
						{
							calibData[i] = coos.clone();
							calibData[i].convertTo(calibData[i], CV_64FC1);
						}
						else
						{
							cv::Mat temp;
							coos.convertTo(temp, CV_64FC1);
							calibData[i] += temp;
						}
					}					
				}
			}

			for (size_t i = 0; i < imgs.size(); i++)
			{
				calibData[i] /= calibCnt;
				if (imgs[i]->type() == CV_16UC1)
				{

				}
				else
				{
					std::vector<cv::Point2f> srcCandidateCorners;
					for (size_t r = 0; r < chessBoradSize.height; r++)
					{
						for (size_t c = 0; c < chessBoradSize.width; c++)
						{
							double x = calibData[i].at<cv::Vec2d>(r, c)[0];
							double y = calibData[i].at<cv::Vec2d>(r, c)[1];
							srcCandidateCorners.push_back(cv::Point2f(x, y));
						}
					}
					cv::Mat intr = stream2Intr[i]->clone();
					cv::Mat Rvect;
					cv::Mat t;
					cv::solvePnP(true3DPointSet, srcCandidateCorners, intr, cv::Mat::zeros(1, 5, CV_32FC1), Rvect, t);
					cv::Mat Rmetrix;
					cv::Rodrigues(Rvect, Rmetrix);
					Rmetrix.copyTo(std::get<0>(stream2Extr[i]));
					t.copyTo(std::get<1>(stream2Extr[i]));
					LOG(INFO) << Rmetrix;
					LOG(INFO) << t;
					//cv::imwrite(std::to_string(i) + ".jpg", grayCalibImgs[i]);
				}				
			}
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