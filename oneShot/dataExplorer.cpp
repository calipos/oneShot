﻿#include<algorithm>
#include<chrono>
#include"stringOp.h"
#include"logg.h"
#include"iofile.h"
#include"dataExplorer.h"

#include"unreGpu.h"
#include "opencvAssistant.h"

#ifdef OPENCV_SHOW
#include "opencv2/opencv.hpp"
#endif
//#define CHECK_CUDA_DOWNSAMPLE
//#define CHECK_CUDA_VOXEL //和CHECK_CUDA_RAYCAST不能同时开启，因为会有变量重定义
//#define CHECK_CUDA_RAYCAST


#ifdef PCL_SHOW
#include "pcl/visualization/cloud_viewer.h"
#endif // PCL_SHOW


extern short2*volume;
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

			int key = cv::waitKey(10);

			//cv::FileStorage fs("data.yml", cv::FileStorage::WRITE);
			//fs << "color" << show1;
			//fs << "depth" << show2;
			//fs.release();
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

	int DataExplorer::doTsdf()
	{
		readExtrParams();
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
		short*depth_dev = NULL;//用以接受host 设备的深度图
		float*scaledDepth = NULL;//scale的深度图
		float3*dev_vmap = NULL;//用以接受从体素模型中扫出的点云
		short*depth_midfiltered = NULL;//用来接受中值滤波的结果
		short*depth_filled = NULL;//用来接受填充的结果
		short2*depth_2 = NULL;//用来接受2阶下采样
		short2*depth_3 = NULL;//用来接受3阶下采样

#ifdef DOWNSAMPLE3TIMES
		int downsample_h4 = show2.rows / 64;
		int downsample_w4 = show2.cols / 64;
		short2*depth_4 = creatGpuData<short2>(downsample_h3*downsample_w3);
#endif
		initVolu(depth_dev, scaledDepth, dev_vmap, depth_midfiltered, depth_filled, depth_2, depth_3, show2.rows, show2.cols);
		float3* host_vmap_m = new float3[show2.rows* show2.cols];//把raycast出来的点云拷到host

#ifdef CHECK_CUDA_VOXEL
		short2* volume___ = new short2[VOLUME_X * VOLUME_Y * VOLUME_Z];
#endif
#ifdef CHECK_CUDA_RAYCAST
		short2* volume___ = new short2[VOLUME_X * VOLUME_Y * VOLUME_Z];
		cv::Mat vmap2map = cv::Mat::zeros(show2.rows, show2.cols, CV_32FC3);
#endif
#ifdef PCL_SHOW
		pcl::visualization::PCLVisualizer cloud_viewer_;
		cloud_viewer_.setBackgroundColor(0, 0, 0.15);
		cloud_viewer_.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1);
		cloud_viewer_.addCoordinateSystem(1.0, "global");
		cloud_viewer_.initCameraParameters();
		cloud_viewer_.setPosition(0, 0);
		cloud_viewer_.setSize(640, 360);
		cloud_viewer_.setCameraClipDistances(0.01, 10.01);
#endif
		int time__ = -500;
		float truct = 0.03;
		while (time__--)
		{
			auto xxx = ((FrameRingBuffer<unsigned char>*)bufferVecP[0].data)->pop(show1.data);
			auto yyy = ((FrameRingBuffer<unsigned short>*)bufferVecP[1].data)->pop(show2.data);
			auto zzz = ((FrameRingBuffer<unsigned char>*)bufferVecP[2].data)->pop(show3.data);			
	/*		
			cv::FileStorage fs("testDepthMat.xml", cv::FileStorage::READ);
			cv::Mat testDepthMat;
			fs["testDepthMat"] >> testDepthMat;
			fs.release();

			cv::Mat deepImg_gray_;

#ifdef SHOW_AFFINE
			cv::Mat test_affine;
			cv::resize(deepImg_gray, test_affine, cv::Size(1080, 720));
			for (int z = 0; z < 1; z++)for (int x = 0; x < VOLUME_X; x++)for (int y = 0; y < VOLUME_Y; y++)
			{
				cv::Mat testP = cv::Mat::zeros(3, 1, CV_64FC1);
				testP.ptr<double>(0)[0] = x;
				testP.ptr<double>(1)[0] = y;
				testP.ptr<double>(2)[0] = z;
				cv::Mat camP = (std::get<0>(stream2Extr[2])*testP + std::get<1>(stream2Extr[2]));
				cv::Mat y_ = cv::Mat(3, 1, CV_64F);
				y_.ptr <double>(0)[0] = camP.ptr<double>(0)[0] * stream2Intr[1]->ptr<double>(0)[0] +
					camP.ptr<double>(2)[0] * stream2Intr[1]->ptr<double>(0)[2];
				y_.ptr <double>(1)[0] = camP.ptr<double>(1)[0] * stream2Intr[1]->ptr<double>(1)[1] +
					camP.ptr<double>(2)[0] * stream2Intr[1]->ptr<double>(1)[2];
				y_.ptr <double>(2)[0] = camP.ptr<double>(2)[0];
				y_ = y_ / y_.ptr<double>(2)[0];				
				cv::circle(test_affine, cv::Point(y_.ptr<double>(0)[0], y_.ptr<double>(1)[0]), 3, cv::Scalar(0, 0, 255), -1);
			}
#endif // SHOW_AFFINE
									
			cv::resize(testDepthMat, show2, cv::Size(1080,720));			
*/			

			Mat33 R_(std::get<0>(stream2Extr[1]).ptr<double>(0),
				std::get<0>(stream2Extr[1]).ptr<double>(1),
				std::get<0>(stream2Extr[1]).ptr<double>(2));
			cv::Mat R_inv_cv = std::get<0>(stream2Extr[1]).inv();
			Mat33 R_inv(R_inv_cv.ptr<double>(0),
				R_inv_cv.ptr<double>(1),
				R_inv_cv.ptr<double>(2));
			float3 t_;
			t_.x = std::get<1>(stream2Extr[1]).ptr<double>(0)[0] ;
			t_.y = std::get<1>(stream2Extr[1]).ptr<double>(1)[0] ;
			t_.z = std::get<1>(stream2Extr[1]).ptr<double>(2)[0] ;
			cv::Mat deepCameraPos_cv = R_inv_cv*(0 - std::get<1>(stream2Extr[1]));
			float3 deepCameraPos_;
			deepCameraPos_.x = deepCameraPos_cv.ptr<double>(0)[0];
			deepCameraPos_.y = deepCameraPos_cv.ptr<double>(1)[0];
			deepCameraPos_.z = deepCameraPos_cv.ptr<double>(2)[0];

			cudaMemcpy((void*)depth_dev, (void*)show2.data, show2.rows*show2.cols*sizeof(unsigned short),cudaMemcpyHostToDevice);
			int downsample_h2 = show2.rows / 4;
			int downsample_w2 = show2.cols / 4;
			int downsample_h3 = show2.rows / 16;
			int downsample_w3 = show2.cols / 16;
		
#ifdef DOWNSAMPLE3TIMES
			midfilter33AndFillHoles44_downsample3t(depth_dev, show2.rows,show2.cols,
				depth_midfiltered, depth_filled,
				depth_2, downsample_h2, downsample_w2,
				depth_3, downsample_h3, downsample_w3,
				depth_4, downsample_h4, downsample_w4
				);
#else 			
			//midfilter33AndFillHoles44_downsample2t(depth_dev, show2.rows, show2.cols,
			//	depth_midfiltered, depth_filled,
			//	depth_2, downsample_h2, downsample_w2,
			//	depth_3, downsample_h3, downsample_w3
			//);
#endif
#ifdef CHECK_CUDA_DOWNSAMPLE
			cv::Mat midfiltered_cvmat = cv::Mat(show2.rows, show2.cols, CV_16SC1);
			cudaMemcpy((void*)midfiltered_cvmat.data, (void*)depth_midfiltered, show2.rows*show2.cols * sizeof(short), cudaMemcpyDeviceToHost);
			cv::Mat filled_cvmat = cv::Mat(show2.rows, show2.cols, CV_16SC1);
			cudaMemcpy((void*)filled_cvmat.data, (void*)depth_filled, show2.rows*show2.cols * sizeof(short), cudaMemcpyDeviceToHost);
			short2*hostDownSample2 = new short2[show2.rows* show2.cols/16];
			cudaMemcpy((void*)hostDownSample2, (void*)depth_2, show2.rows*show2.cols * sizeof(short2)/16, cudaMemcpyDeviceToHost);
			cv::Mat hostDownSample2_cvmat = cv::Mat(show2.rows/4, show2.cols/4, CV_16SC1);
			for (int i = 0; i < show2.rows / 4; i++)for (int j = 0; j < show2.cols / 4; j++)
			{
				hostDownSample2_cvmat.ptr<short>(i)[j] = hostDownSample2[i*show2.cols / 4 + j].x;
			}
			short2*hostDownSample3 = new short2[show2.rows* show2.cols / 256];
			cudaMemcpy((void*)hostDownSample3, (void*)depth_3, show2.rows*show2.cols * sizeof(short2) / 256, cudaMemcpyDeviceToHost);
			cv::Mat hostDownSample3_cvmat = cv::Mat(show2.rows / 16, show2.cols / 16, CV_16SC1);
			for (int i = 0; i < show2.rows / 16; i++)for (int j = 0; j < show2.cols / 16; j++)
			{
				hostDownSample3_cvmat.ptr<short>(i)[j] = hostDownSample3[i*show2.cols / 16 + j].x;
			}
#ifdef DOWNSAMPLE3TIMES
			short2*hostDownSample4 = new short2[show2.rows* show2.cols / 4096];
			cudaMemcpy((void*)hostDownSample4, (void*)depth_4, show2.rows*show2.cols * sizeof(short2) / 4096, cudaMemcpyDeviceToHost);
			cv::Mat hostDownSample4_cvmat = cv::Mat(show2.rows / 64, show2.cols / 64, CV_16SC1);
			for (int i = 0; i < show2.rows / 64; i++)for (int j = 0; j < show2.cols / 64; j++)
			{
				hostDownSample4_cvmat.ptr<short>(i)[j] = hostDownSample4[i*show2.cols / 64 + j].x;
			}
#endif
#endif // CHECK_CUDA_DOWNSAMPLE





			// depth_filled depth_dev depth_midfiltered
			integrateTsdfVolume(depth_dev, show2.rows, show2.cols,
				stream2Intr[1]->ptr<double>(0)[2], stream2Intr[1]->ptr<double>(1)[2], 
				stream2Intr[1]->ptr<double>(0)[0], stream2Intr[1]->ptr<double>(1)[1],
				R_, t_, deepCameraPos_, truct, volume, scaledDepth);
			{
#ifdef CHECK_CUDA_VOXEL				
				cv::Mat cpu_data(show2.rows, show2.cols, CV_32FC1);
				cudaMemcpy(cpu_data.data, scaledDepth, show2.rows* show2.cols * sizeof(float), cudaMemcpyDeviceToHost);
				float3 cell_size; 
				cell_size.x = 1.*VOLUME_SIZE_X/ VOLUME_X; 
				cell_size.y = 1.*VOLUME_SIZE_Y / VOLUME_Y; 
				cell_size.z = 1.*VOLUME_SIZE_Z / VOLUME_Z;
				auto& R = R_;
				auto& t = t_;
				if (1)//cpu模拟cuda体素过程
				{				
					for (int y = VOLUME_Y/2; y < VOLUME_Y / 2+1; y++)for (int x = VOLUME_X/2; x < VOLUME_X / 2+1; x++)
					{
						float v_g_x = (x + 0.5f) * cell_size.x;
						float v_g_y = (y + 0.5f) * cell_size.y;
						float v_g_z = (0.5f) * cell_size.z;

						float v_x = (R.data[0].x * v_g_x + R.data[0].y * v_g_y + R.data[0].z * v_g_z) + t.x;
						float v_y = (R.data[1].x * v_g_x + R.data[1].y * v_g_y + R.data[1].z * v_g_z) + t.y;
						float v_z = (R.data[2].x * v_g_x + R.data[2].y * v_g_y + R.data[2].z * v_g_z) + t.z;
						float diff_x = v_g_x - deepCameraPos_.x;
						float diff_y = v_g_y - deepCameraPos_.y;
						float v_part_norm = diff_x * diff_x + diff_y * diff_y;
						//v_z = v_z + t.z;
						v_x = (v_x) * stream2Intr[1]->ptr<double>(0)[2];
						v_y = (v_y) * stream2Intr[1]->ptr<double>(1)[2];

						float z_scaled = 0;

						float Rcurr_inv_0_z_scaled = R.data[0].z * cell_size.z * stream2Intr[1]->ptr<double>(0)[0];
						float Rcurr_inv_1_z_scaled = R.data[1].z * cell_size.z * stream2Intr[1]->ptr<double>(1)[1];
						float tranc_dist = truct;
						float tranc_dist_inv = 1.0f / tranc_dist;

						int elem_step = VOLUME_X * VOLUME_Y;
						short2* pos = volume___ + y*VOLUME_X + x;
						

						//#pragma unroll
						for (int z = 0; z <VOLUME_Z;
							++z,
							v_g_z += cell_size.z,
							z_scaled += R.data[2].z*cell_size.z,
							v_x += Rcurr_inv_0_z_scaled,
							v_y += Rcurr_inv_1_z_scaled,
							pos += elem_step)
						{
							float this_v_z = v_z + z_scaled;
							float inv_z = 1.0f / (this_v_z);
							if (inv_z < 0)
								continue;

							// project to current cam
							int2 coo =
							{
								int(v_x * inv_z + stream2Intr[1]->ptr<double>(0)[2]),
								int(v_y * inv_z + stream2Intr[1]->ptr<double>(1)[2])
							};

							if (coo.x >= 0 && coo.y >= 0 && coo.x < show2.cols && coo.y < show2.rows)         //6
							{
								//v_part_norm += this_v_z*this_v_z;
								float distance_sqr = v_part_norm+ (v_g_z- deepCameraPos_.z)*(v_g_z - deepCameraPos_.z);
								float weight = 1. / (coo.x - 1080 / 2)*(coo.x - 1080 / 2) + (coo.y - 720 / 2)*(coo.y - 720 / 2);
								float Dp_scaled = cpu_data.ptr<float>(coo.y)[coo.x]; //meters
								float distance = sqrtf(distance_sqr);
								float sdf = (Dp_scaled - distance);																
								LOG(INFO) << sdf;
								if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
								{
									
									pos->x = std::max(-32767, std::min(32767, int(fmin(1.f,sdf/ tranc_dist) * 32767)));
									pos->y = weight * 32767;
									
								}
							}
						}
						//break;
					}
				}
				if (1)//cpu显示cuda的体素结果
				{
					cv::Mat shows1 = cv::Mat::zeros(VOLUME_Y, VOLUME_X, CV_16SC1);
					cv::Mat shows2 = cv::Mat::zeros(VOLUME_Y, VOLUME_X, CV_16SC1);
					cudaMemcpy(volume___, volume, VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(short2), cudaMemcpyDeviceToHost);
					for (int z = 0; z<VOLUME_Z; z++)
					{
						
						short*temp1 = (short*)shows1.data;
						short*temp2 = (short*)shows2.data;
						
						for (size_t i = 0; i < VOLUME_X*VOLUME_Y; i++)
						{
							
							temp1[i] = volume___[z * VOLUME_X * VOLUME_Y + i].x;
							temp2[i] = volume___[z * VOLUME_X * VOLUME_Y + i].y;
						}
						cv::imshow("123", shows1);
						cv::waitKey(50);
						//break;
					}
				//	cv::imshow("123", shows1);
					//cv::waitKey(20);
				}			
#endif			
			}
			
			raycastPoint(volume, dev_vmap, show2.rows, show2.cols,
				stream2Intr[1]->ptr<double>(0)[2], stream2Intr[1]->ptr<double>(1)[2],
				stream2Intr[1]->ptr<double>(0)[0], stream2Intr[1]->ptr<double>(1)[1],
				R_inv, t_, deepCameraPos_,truct);

#ifdef CHECK_CUDA_RAYCAST					
			if (1) //cpu仿真raycast
			{
				cudaMemcpy(volume___, volume, VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(short2), cudaMemcpyDeviceToHost);

				for (int i = show2.rows*.5; i < show2.rows*.5+1; i++)for (int j = show2.cols*.5; j < show2.cols*.5+1; j++)
				//for (int i = show2.rows*.5; i < show2.rows; i++)for (int j = show2.cols*.5; j < show2.cols; j++)
				{
					if ((i+j)%2)
					{
						continue;
					}
					cv::Mat ray_start = deepCameraPos_cv.clone();
					//ray_start.ptr<double>(0)[0] = t_.x;
					//ray_start.ptr<double>(1)[0] = t_.y;
					//ray_start.ptr<double>(2)[0] = t_.z;

					cv::Mat cmos_pos = cv::Mat(3, 1, CV_64FC1);
					cmos_pos.ptr<double>(0)[0] = 0.5*(j - stream2Intr[1]->ptr<double>(0)[2]) / stream2Intr[1]->ptr<double>(0)[0];
					cmos_pos.ptr<double>(1)[0] = 0.5*(i - stream2Intr[1]->ptr<double>(1)[2]) / stream2Intr[1]->ptr<double>(1)[1];
					cmos_pos.ptr<double>(2)[0] = 0.5;
					
					//cv::Mat ray_next = std::get<0>(stream2Extr[2]).inv()*(cmos_pos - ray_start);
					cv::Mat ray_next = R_inv_cv*(cmos_pos - std::get<1>(stream2Extr[1]));
					cv::Mat ray_next_back = std::get<0>(stream2Extr[2])*ray_next + ray_start;

					cv::Mat ray_dir = cv::Mat(ray_next - ray_start);
					double mod = ray_dir.ptr<double>(0)[0] * ray_dir.ptr<double>(0)[0]
						+ ray_dir.ptr<double>(1)[0] * ray_dir.ptr<double>(1)[0]
						+ ray_dir.ptr<double>(2)[0] * ray_dir.ptr<double>(2)[0];
					mod = sqrtf(mod);
					ray_dir /= mod;
					ray_dir.ptr<double>(0)[0] = (ray_dir.ptr<double>(0)[0] == 0.f) ? 1e-15 : ray_dir.ptr<double>(0)[0];
					ray_dir.ptr<double>(0)[1] = (ray_dir.ptr<double>(0)[1] == 0.f) ? 1e-15 : ray_dir.ptr<double>(0)[1];
					ray_dir.ptr<double>(0)[2] = (ray_dir.ptr<double>(0)[2] == 0.f) ? 1e-15 : ray_dir.ptr<double>(0)[2];

					float txmin = 0;
					float tymin = 0;
					float tzmin = 0;
					float txmax = 0;
					float tymax = 0;
					float tzmax = 0;
					if (ray_dir.ptr<double>(0)[0]>0)
					{
						if (ray_start.ptr<double>(0)[0]>VOLUME_SIZE_X)
						{
							txmin = FLT_MAX;
							txmax = FLT_MAX;
						}
						else if (ray_start.ptr<double>(0)[0]>0)
						{
							txmin = 0;
							txmax = (VOLUME_SIZE_X - ray_start.ptr<double>(0)[0]) / ray_dir.ptr<double>(0)[0];
						}
						else
						{
							txmin = (-ray_start.ptr<double>(0)[0]) / ray_dir.ptr<double>(0)[0];
							txmax = (VOLUME_SIZE_X - ray_start.ptr<double>(0)[0]) / ray_dir.ptr<double>(0)[0];
						}
					}
					else
					{
						if (ray_start.ptr<double>(0)[0]<0)
						{
							txmin = FLT_MAX;
							txmax = FLT_MAX;
						}
						else if (ray_start.ptr<double>(0)[0]<VOLUME_SIZE_X)
						{
							txmin = 0;
							txmax = (0 - ray_start.ptr<double>(0)[0]) / ray_dir.ptr<double>(0)[0];
						}
						else
						{
							txmin = (VOLUME_SIZE_X - ray_start.ptr<double>(0)[0]) / ray_dir.ptr<double>(0)[0];
							txmax = (-ray_start.ptr<double>(0)[0]) / ray_dir.ptr<double>(0)[0];
						}
					}
					if (ray_dir.ptr<double>(1)[0]>0)
					{
						if (ray_start.ptr<double>(1)[0]>VOLUME_SIZE_Y)
						{
							tymin = FLT_MAX;
							tymax = FLT_MAX;
						}
						else if (ray_start.ptr<double>(1)[0]>0)
						{
							tymin = 0;
							tymax = (VOLUME_SIZE_Y - ray_start.ptr<double>(1)[0]) / ray_dir.ptr<double>(1)[0];
						}
						else
						{
							tymin = (-ray_start.ptr<double>(1)[0]) / ray_dir.ptr<double>(1)[0];
							tymax = (VOLUME_SIZE_Y - ray_start.ptr<double>(1)[0]) / ray_dir.ptr<double>(1)[0];
						}
					}
					else
					{
						if (ray_start.ptr<double>(1)[0]<0)
						{
							tymin = FLT_MAX;
							tymax = FLT_MAX;
						}
						else if (ray_start.ptr<double>(1)[0]<VOLUME_SIZE_Y)
						{
							tymin = 0.;
							tymax = (0 - ray_start.ptr<double>(1)[0]) / ray_dir.ptr<double>(1)[0];;
						}
						else
						{
							tymin = (VOLUME_SIZE_Y - ray_start.ptr<double>(1)[0]) / ray_dir.ptr<double>(1)[0];
							tymax = (-ray_start.ptr<double>(1)[0]) / ray_dir.ptr<double>(1)[0];
						}
					}
					if (ray_dir.ptr<double>(2)[0]>0)
					{
						if (ray_start.ptr<double>(2)[0]>VOLUME_SIZE_Z)
						{
							tzmin = FLT_MAX;
							tzmax = FLT_MAX;
						}
						else if (ray_start.ptr<double>(2)[0]>0)
						{
							tzmin = 0;
							tzmax = (VOLUME_SIZE_Z - ray_start.ptr<double>(2)[0]) / ray_dir.ptr<double>(2)[0];
						}
						else
						{
							tzmin = (-ray_start.ptr<double>(2)[0]) / ray_dir.ptr<double>(2)[0];
							tzmax = (VOLUME_SIZE_Z - ray_start.ptr<double>(2)[0]) / ray_dir.ptr<double>(2)[0];
						}
					}
					else
					{
						if (ray_start.ptr<double>(2)[0]<0)
						{
							tzmin = FLT_MAX;
							tzmax = FLT_MAX;
						}
						else if (ray_start.ptr<double>(2)[0]<VOLUME_SIZE_Z)
						{
							tzmin = 0;
							tzmax = (0 - ray_start.ptr<double>(2)[0]) / ray_dir.ptr<double>(2)[0];;
						}
						else
						{
							tzmin = (VOLUME_SIZE_Z - ray_start.ptr<double>(2)[0]) / ray_dir.ptr<double>(2)[0];
							tzmax = (-ray_start.ptr<double>(2)[0]) / ray_dir.ptr<double>(2)[0];
						}
					}

					double time_start_volume = fmax(fmax(txmin, tymin), tzmin);
					double time_exit_volume = fmin(fmin(txmax, tymax), tzmax);

					const double min_dist = 0.f;         //in meters
					time_start_volume = fmax(time_start_volume, min_dist);
					if (time_start_volume >= time_exit_volume)
						continue;

					double time_curr = time_start_volume;
					cv::Mat voxelPos = (ray_start + ray_dir * time_curr);
					int g_x = std::max(0, std::min(int(voxelPos.ptr<double>(0)[0] / VOLUME_SIZE_X* VOLUME_X), VOLUME_X - 1));
					int g_y = std::max(0, std::min(int(voxelPos.ptr<double>(1)[0] / VOLUME_SIZE_Y*VOLUME_Y), VOLUME_Y - 1));
					int g_z = std::max(0, std::min(int(voxelPos.ptr<double>(2)[0] / -VOLUME_SIZE_Z*VOLUME_Z), VOLUME_Z - 1));

					double tsdf = volume___[g_z*VOLUME_X*VOLUME_Y + g_y*VOLUME_X + g_x].x*1.0 / SHRT_MAX;
					//infinite loop guard
					const double max_time = 3 * (VOLUME_SIZE_X + VOLUME_SIZE_Y + VOLUME_SIZE_Z);

					double time_step = 0.8 * 0.03;//**
					for (; time_curr < max_time; time_curr += time_step)
					{
						double tsdf_prev = tsdf;
						LOG(INFO) << tsdf_prev;
						cv::Mat voxelPos = (ray_start + ray_dir * (time_curr + time_step));
						int g_x = std::max(0, std::min(int(voxelPos.ptr<double>(0)[0] / VOLUME_SIZE_X*VOLUME_X), VOLUME_X - 1));
						int g_y = std::max(0, std::min(int(voxelPos.ptr<double>(1)[0] / VOLUME_SIZE_Y*VOLUME_Y), VOLUME_Y - 1));
						int g_z = std::max(0, std::min(int(voxelPos.ptr<double>(2)[0] / -VOLUME_SIZE_Z*VOLUME_Z), VOLUME_Z - 1));
						if (!(g_x >= 0 && g_y >= 0 && g_z >= 0 && g_x < VOLUME_X && g_y < VOLUME_Y && g_z < VOLUME_Z))
							continue;

						tsdf = volume___[g_z*VOLUME_X*VOLUME_Y + g_y*VOLUME_X + g_x].x*1.0 / SHRT_MAX;;

						if (tsdf_prev < 0.f && tsdf > 0.f)
							break;

						if (tsdf_prev > 0.f && tsdf < 0.f)           //zero crossing
						{
							double Ftdt = 0.0;
							cv::Mat thisVoxelPos = (ray_start + ray_dir * (time_curr + time_step));
							int this_g_x = std::max(0, std::min(int(voxelPos.ptr<double>(0)[0] / VOLUME_SIZE_X*VOLUME_X), VOLUME_X - 1));
							int this_g_y = std::max(0, std::min(int(voxelPos.ptr<double>(1)[0] / VOLUME_SIZE_Y*VOLUME_Y), VOLUME_Y - 1));
							int this_g_z = std::max(0, std::min(int(voxelPos.ptr<double>(2)[0] / -VOLUME_SIZE_Z*VOLUME_Z), VOLUME_Z - 1));
							if (this_g_x <= 0 || this_g_x >= VOLUME_X - 1)
								Ftdt = 0.0;
							if (this_g_y <= 0 || this_g_y >= VOLUME_Y - 1)
								Ftdt = 0.0;
							if (this_g_z <= 0 || this_g_z >= VOLUME_Z - 1)
								Ftdt = 0.0;
							double this_vx = (this_g_x + 0.5f) * VOLUME_SIZE_X/VOLUME_X;
							double this_vy = (this_g_y + 0.5f) * VOLUME_SIZE_Y/VOLUME_Y;
							double this_vz = -(this_g_z + 0.5f) * VOLUME_SIZE_Z/VOLUME_Z;

							this_g_x = (thisVoxelPos.ptr<double>(0)[0] < this_vx) ? (this_g_x - 1) : this_g_x;
							this_g_y = (thisVoxelPos.ptr<double>(1)[0] < this_vy) ? (this_g_y - 1) : this_g_y;
							this_g_z = (thisVoxelPos.ptr<double>(2)[0] > this_vz) ? (this_g_z - 1) : this_g_z;

							double a = (thisVoxelPos.ptr<double>(0)[0] - (this_g_x + 0.5f) * VOLUME_SIZE_X / VOLUME_X) / VOLUME_SIZE_X * VOLUME_X;
							double b = (thisVoxelPos.ptr<double>(1)[0] - (this_g_y + 0.5f) * VOLUME_SIZE_Y / VOLUME_Y) / VOLUME_SIZE_Y * VOLUME_Y;
							double c = (-thisVoxelPos.ptr<double>(2)[0] - (this_g_z + 0.5f) * VOLUME_SIZE_Z / VOLUME_Z) / VOLUME_SIZE_Z * VOLUME_Z;

							Ftdt =
								volume___[this_g_z*VOLUME_X*VOLUME_Y + this_g_y*VOLUME_X + this_g_x].x*1.0 / SHRT_MAX * (1 - a) * (1 - b) * (1 - c) +
								volume___[(this_g_z + 1)*VOLUME_X*VOLUME_Y + this_g_y*VOLUME_X + this_g_x].x*1.0 / SHRT_MAX * (1 - a) * (1 - b) * c +
								volume___[this_g_z*VOLUME_X*VOLUME_Y + (this_g_y + 1)*VOLUME_X + this_g_x].x*1.0 / SHRT_MAX * (1 - a) * b * (1 - c) +
								volume___[(this_g_z + 1)*VOLUME_X*VOLUME_Y + (this_g_y + 1)*VOLUME_X + this_g_x].x*1.0 / SHRT_MAX * (1 - a) * b * c +
								volume___[this_g_z*VOLUME_X*VOLUME_Y + this_g_y*VOLUME_X + this_g_x + 1].x*1.0 / SHRT_MAX * a * (1 - b) * (1 - c) +
								volume___[(this_g_z + 1)*VOLUME_X*VOLUME_Y + this_g_y*VOLUME_X + this_g_x + 1].x*1.0 / SHRT_MAX * a * (1 - b) * c +
								volume___[this_g_z*VOLUME_X*VOLUME_Y + (this_g_y + 1)*VOLUME_X + this_g_x + 1].x*1.0 / SHRT_MAX * a * b * (1 - c) +
								volume___[(this_g_z + 1)*VOLUME_X*VOLUME_Y + (this_g_y + 1)*VOLUME_X + this_g_x + 1].x*1.0 / SHRT_MAX * a * b * c;
							if (Ftdt == 0)
							{
								continue;
							}
							double Ft = 0.0;
							cv::Mat thisVoxelPos2 = (ray_start + ray_dir * (time_curr));
							int this2_g_x = std::max(0, std::min(int(thisVoxelPos2.ptr<double>(0)[0] / VOLUME_SIZE_X*VOLUME_X), VOLUME_X - 1));
							int this2_g_y = std::max(0, std::min(int(thisVoxelPos2.ptr<double>(1)[0] / VOLUME_SIZE_Y*VOLUME_Y), VOLUME_Y - 1));
							int this2_g_z = std::max(0, std::min(int(thisVoxelPos2.ptr<double>(2)[0] / -VOLUME_SIZE_Z*VOLUME_Z), VOLUME_Z - 1));
							if (this2_g_x <= 0 || this2_g_x >= VOLUME_X - 1)
								Ft = 0.0;
							if (this2_g_y <= 0 || this2_g_y >= VOLUME_Y - 1)
								Ft = 0.0;
							if (this2_g_z <= 0 || this2_g_z >= VOLUME_Z - 1)
								Ft = 0.0;
							double this2_vx = (this2_g_x + 0.5f) * VOLUME_SIZE_X / VOLUME_X;
							double this2_vy = (this2_g_y + 0.5f) * VOLUME_SIZE_Y / VOLUME_Y;
							double this2_vz = -(this2_g_z + 0.5f) * VOLUME_SIZE_Z / VOLUME_Z;

							this2_g_x = (thisVoxelPos2.ptr<double>(0)[0] < this2_vx) ? (this2_g_x - 1) : this2_g_x;
							this2_g_y = (thisVoxelPos2.ptr<double>(1)[0] < this2_vy) ? (this2_g_y - 1) : this2_g_y;
							this2_g_z = (thisVoxelPos2.ptr<double>(2)[0] > this2_vz) ? (this2_g_z - 1) : this2_g_z;


							double a2 = (thisVoxelPos2.ptr<double>(0)[0] - (this2_g_x + 0.5f) * VOLUME_SIZE_X / VOLUME_X) / VOLUME_SIZE_X * VOLUME_X;
							double b2 = (thisVoxelPos2.ptr<double>(1)[0] - (this2_g_y + 0.5f) * VOLUME_SIZE_Y / VOLUME_Y) / VOLUME_SIZE_Y * VOLUME_Y;
							double c2 = (-thisVoxelPos2.ptr<double>(2)[0] - (this2_g_z + 0.5f) * VOLUME_SIZE_Z / VOLUME_Z) / VOLUME_SIZE_Z * VOLUME_Z;

							Ft =
								volume___[this2_g_z*VOLUME_X*VOLUME_Y + this2_g_y*VOLUME_X + this2_g_x].x*1.0 / SHRT_MAX * (1 - a2) * (1 - b2) * (1 - c2) +
								volume___[(this2_g_z + 1)*VOLUME_X*VOLUME_Y + this2_g_y*VOLUME_X + this2_g_x].x*1.0 / SHRT_MAX * (1 - a2) * (1 - b2) * c2 +
								volume___[this2_g_z*VOLUME_X*VOLUME_Y + (this2_g_y + 1)*VOLUME_X + this2_g_x].x*1.0 / SHRT_MAX * (1 - a2) * b2 * (1 - c2) +
								volume___[(this2_g_z + 1)*VOLUME_X*VOLUME_Y + (this2_g_y + 1)*VOLUME_X + this2_g_x].x*1.0 / SHRT_MAX * (1 - a2) * b2 * c2 +
								volume___[this2_g_z*VOLUME_X*VOLUME_Y + this2_g_y*VOLUME_X + this2_g_x + 1].x*1.0 / SHRT_MAX * a2 * (1 - b2) * (1 - c2) +
								volume___[(this2_g_z + 1)*VOLUME_X*VOLUME_Y + this2_g_y*VOLUME_X + this2_g_x + 1].x*1.0 / SHRT_MAX * a2 * (1 - b2) * c2 +
								volume___[this2_g_z*VOLUME_X*VOLUME_Y + (this2_g_y + 1)*VOLUME_X + this2_g_x + 1].x*1.0 / SHRT_MAX * a2 * b2 * (1 - c2) +
								volume___[(this2_g_z + 1)*VOLUME_X*VOLUME_Y + (this2_g_y + 1)*VOLUME_X + this2_g_x + 1].x*1.0 / SHRT_MAX * a2 * b2 * c2;
							if (Ft == 0)
							{
								continue;
							}
							double Ts = time_curr - time_step * Ft / (Ftdt - Ft);

							cv::Mat temp = (ray_start + ray_dir * Ts);
							vmap2map.at<cv::Vec3f>(i, j)[0] = static_cast<float>(temp.ptr<double>(0)[0]);
							vmap2map.at<cv::Vec3f>(i, j)[1] = static_cast<float>(temp.ptr<double>(1)[0]);
							vmap2map.at<cv::Vec3f>(i, j)[2] = static_cast<float>(temp.ptr<double>(2)[0]);					
						}
					}
				}
			}
#ifdef CHECK_CUDA_RAYCAST
			vmap2map.setTo(cv::Scalar(0., 0., 0.));
			cudaMemcpy((void*)vmap2map.data, dev_vmap, show2.rows*show2.cols * sizeof(float) * 3, cudaMemcpyDeviceToHost);
			cv::namedWindow("123",0);
			cv::imshow("123", vmap2map);
			cv::waitKey(10);
#endif


#endif // CHECK_CUDA_RAYCAST

#ifdef PCL_SHOW
			//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sample(new pcl::PointCloud<pcl::PointXYZ>);
			//cloud_sample->width = 50;
			//cloud_sample->height = 10;
			//cloud_sample->is_dense = false;
			//cloud_sample->points.resize(cloud_sample->width * cloud_sample->height);
			//for (size_t i = 0; i < cloud_sample->points.size(); ++i)
			//{
			//	cloud_sample->points[i].x = 32 * rand() / (RAND_MAX + 1.0f);
			//	cloud_sample->points[i].y = 32 * rand() / (RAND_MAX + 1.0f);
			//	cloud_sample->points[i].z = 32.0 * rand() / (RAND_MAX + 1.0f);
			//}
			//pcl::visualization::CloudViewer viewer_sample("Viewer_sample");
			//viewer_sample.showCloud(cloud_sample);
			//while (!viewer_sample.wasStopped())
			//{
			//}
									
			cudaMemcpy(host_vmap_m, dev_vmap, show2.cols * show2.rows * sizeof(float3), cudaMemcpyDeviceToHost);
			int point_cnt = 0;
			for (size_t i = 0; i < show2.cols * show2.rows; i++)
			{
				//if (!(host_vmap_m[i].x == std::numeric_limits<float>::quiet_NaN()))
				if (!(host_vmap_m[i].x == 0.f&&host_vmap_m[i].y == 0.f&&host_vmap_m[i].z == 0.f))
				{
					point_cnt++;
				}
			}
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
			cloud->width = 1;
			cloud->height =  point_cnt;
			cloud->is_dense = false;
			cloud->points.resize(cloud->width * cloud->height);
			point_cnt = 0;
			
			for (size_t i = 0; i < show2.cols * show2.rows; i++)
			{
				//if (!(host_vmap_m[i].x == std::numeric_limits<float>::quiet_NaN()))
				if (!(host_vmap_m[i].x == 0.f&&host_vmap_m[i].y == 0.f&&host_vmap_m[i].z == 0.f))
				{
					cloud->points[point_cnt].x = host_vmap_m[i].x;
					cloud->points[point_cnt].y = host_vmap_m[i].y;
					cloud->points[point_cnt].z = host_vmap_m[i].z;

					point_cnt++;
				}
			}
			//pcl::visualization::CloudViewer viewer("Viewer");			
			//viewer.showCloud(cloud);
			//while (!viewer.wasStopped())
			//{
			//}
			
			cloud_viewer_.removeAllPointClouds();
			cloud_viewer_.addPointCloud<pcl::PointXYZ>(cloud);
			cloud_viewer_.spinOnce(10);
			//std::cout <<1 << std::endl;
#endif
		}
		return 0;
	}

	int DataExplorer::readExtrParams()
	{
		if (FileOP::FileExist("calib.json"))
		{
			rapidjson::Document calibDocRoot;
			calibDocRoot.Parse<0>(unre::StringOP::parseJsonFile2str("calib.json").c_str());
			cv::Size2f chessUnitSize;
			if (calibDocRoot.HasMember("caeraExtParams") && calibDocRoot["caeraExtParams"].IsArray())
			{
				int stream_num_from_calib = calibDocRoot["caeraExtParams"].Size();
				CHECK(stream_num_from_calib == getExactStreamCnt())<<"The extern param must match stream!!!";
				stream2Extr.clear();
				for (size_t stream_from_calib = 0; stream_from_calib < stream_num_from_calib; stream_from_calib++)
				{
					int this_stream_idx = JsonExplorer::getValue<int>(calibDocRoot["caeraExtParams"][stream_from_calib], "streamIdx");
					CHECK(calibDocRoot["caeraExtParams"][stream_from_calib]["R"].IsArray() && calibDocRoot["caeraExtParams"][stream_from_calib]["t"].IsArray());
					int R_r = JsonExplorer::getValue<int>(calibDocRoot["caeraExtParams"][stream_from_calib], "R_rows");
					int R_c = JsonExplorer::getValue<int>(calibDocRoot["caeraExtParams"][stream_from_calib], "R_cols");
					int t_r = JsonExplorer::getValue<int>(calibDocRoot["caeraExtParams"][stream_from_calib], "t_rows");
					int t_c = JsonExplorer::getValue<int>(calibDocRoot["caeraExtParams"][stream_from_calib], "t_cols");
					cv::Mat R = cv::Mat::zeros(R_r, R_c, CV_64FC1);
					cv::Mat t = cv::Mat::zeros(t_r, t_c, CV_64FC1);
					for (size_t i = 0; i < R_r; i++)
					{
						for (size_t j = 0; j < R_c; j++)
						{
							R.ptr<double>(i)[j] = calibDocRoot["caeraExtParams"][stream_from_calib]["R"][i*R_c + j].GetDouble();
						}
					}
					for (size_t i = 0; i < t_r; i++)
					{
						for (size_t j = 0; j < t_c; j++)
						{
							t.ptr<double>(i)[j] = calibDocRoot["caeraExtParams"][stream_from_calib]["t"][i*t_c + j].GetDouble();
						}
					}
					stream2Extr[this_stream_idx] = std::make_tuple(R, t);
				}
				
			}
		}
		else
		{
			LOG(FATAL) << "No calib.json, so cant read the extern params!!";
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
						stream2Extr[streamIdx] = std::make_tuple(cv::Mat(0, 0, CV_64FC1), cv::Mat(0, 0, CV_64FC1));
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
						stream2Extr[streamIdx] = std::make_tuple(cv::Mat(0, 0, CV_64FC1), cv::Mat(0, 0, CV_64FC1));
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
						stream2Extr[streamIdx] = std::make_tuple(cv::Mat(0, 0, CV_64FC1), cv::Mat(0, 0, CV_64FC1));
					}
					else
					{
						LOG(FATAL) << "NOT SUPPORT TYPE";
					}
				}

			}
		}
		readExtrParams();
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
						((unre::FrameRingBuffer<unsigned short>*)bufferVecP[streamIdx].data)->pop(imgs[streamIdx]->data);
						//((unre::FrameRingBuffer<unsigned short>*)bufferVecP[streamIdx].data)->pop();//depth neednt calib
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
	int DataExplorer::pop2Mats_noInfred(std::vector<cv::Mat*>&imgs)
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
						((unre::FrameRingBuffer<unsigned short>*)bufferVecP[streamIdx].data)->pop(imgs[streamIdx]->data);
						//((unre::FrameRingBuffer<unsigned short>*)bufferVecP[streamIdx].data)->pop();//depth neednt calib
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
						((unre::FrameRingBuffer<unsigned char>*)bufferVecP[streamIdx].data)->pop();
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

}