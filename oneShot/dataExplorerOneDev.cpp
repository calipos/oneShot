#include<algorithm>
#include<chrono>
#include"stringOp.h"
#include"logg.h"
#include"iofile.h"
#include"dataExplorer.h"

#include"unreGpu.h"

#ifdef PCL_SHOW
#include "pcl/visualization/cloud_viewer.h"
#endif // PCL_SHOW




void transProc(
	const cv::Mat*img_, const cv::Mat*intr1_, const cv::Mat*R1_, const cv::Mat*t1_,
	const cv::Mat*depth_, const cv::Mat*intr2_, const cv::Mat*R2_, const cv::Mat*t2_,
	cv::Mat*colorized_depth
)
{
	cv::Mat img = img_->clone();
	cv::Mat intr1 = intr1_->clone();
	cv::Mat R1 = R1_->clone();
	cv::Mat t1 = t1_->clone();
	cv::Mat depth = depth_->clone();
	cv::Mat intr2 = intr2_->clone();
	cv::Mat R2 = R2_->clone();
	cv::Mat t2 = t2_->clone();

	int width = depth.cols;
	int height = depth.rows;

	cv::Mat R2_inv = R2.inv();
	cv::Mat depth_as_color = cv::Mat::zeros(img.rows, img.cols, CV_16UC1);
	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			double cmos_x = (j - intr2.ptr<double>(0)[2]) / intr2.ptr<double>(0)[0];
			double cmos_y = (i - intr2.ptr<double>(1)[2]) / intr2.ptr<double>(1)[1];
			double cmos_z = 1.;
			double z = depth.ptr<unsigned short>(i)[j] * 0.001;
			if (z<0.001)
			{
				continue;
			}

			cv::Mat xyw_M = R2.clone();
			xyw_M.ptr<double>(0)[2] = -cmos_x;
			xyw_M.ptr<double>(1)[2] = -cmos_y;
			xyw_M.ptr<double>(2)[2] = -1.;

			cv::Mat xyw__ = t2*-1.0;
			xyw__.ptr<double>(0)[0] = xyw__.ptr<double>(0)[0] - R2.ptr<double>(0)[2] * z;
			xyw__.ptr<double>(1)[0] = xyw__.ptr<double>(1)[0] - R2.ptr<double>(1)[2] * z;
			xyw__.ptr<double>(2)[0] = xyw__.ptr<double>(2)[0] - R2.ptr<double>(2)[2] * z;

			cv::Mat xyw = xyw_M.inv()*xyw__;
			cv::Mat xyz = xyw.clone();
			xyz.ptr<double>(2)[0] = z;

			cv::Mat test = R2*xyz + t2;
			test.ptr<double>(0)[0] = test.ptr<double>(0)[0] / test.ptr<double>(2)[0] * intr2.ptr<double>(0)[0] + intr2.ptr<double>(0)[2];
			test.ptr<double>(1)[0] = test.ptr<double>(1)[0] / test.ptr<double>(2)[0] * intr2.ptr<double>(1)[1] + intr2.ptr<double>(1)[2];
			test.ptr<double>(2)[0] = test.ptr<double>(2)[0] / test.ptr<double>(2)[0];

			cv::Mat pointInColor = R1*xyz + t1;
			pointInColor /= pointInColor.ptr<double>(2)[0];
			int this_x = 0.5 + pointInColor.ptr<double>(0)[0] * intr1.ptr<double>(0)[0] + intr1.ptr<double>(0)[2];
			int this_y = 0.5 + pointInColor.ptr<double>(1)[0] * intr1.ptr<double>(1)[1] + intr1.ptr<double>(1)[2];
			if (this_x<0 || this_y<0 || this_x >= img.cols || this_y >= img.rows)
			{
				continue;
			}
			depth_as_color.ptr<unsigned short>(this_y)[this_x] = static_cast<unsigned short>(z * 1000);
		}
	}
	depth_as_color.copyTo(*colorized_depth);
}



namespace unre
{
	int DataExplorer::oneDevShow()
	{
		
		std::vector<cv::Mat*> imgs;
		initMatVect(imgs);


		short*depth_dev_input = NULL;
		float*depth_dev_output = NULL;
		short*depth_dev_bila = NULL;
		float*depth_dev_med = NULL;//用来接受中值滤波的结果
		float*depth_filled = NULL;//用来接受填充的结果
		float2*depth_2 = NULL;//用来接受2阶下采样
		float2*depth_3 = NULL;//用来接受3阶下采样
		float*nmap = NULL;
		float*nmap_average = NULL;
		float*vmap = NULL;
		unsigned char*rgbData = NULL;
		unsigned char*newRgbData = NULL;
		initOneDevDeep(depth_dev_input,depth_dev_output, depth_dev_bila,depth_dev_med, depth_filled, depth_2, depth_3,vmap, nmap, nmap_average,rgbData, newRgbData, imgs[1]->rows, imgs[1]->cols, imgs[0]->rows, imgs[0]->cols);
#if N2MAP			
		float*n2map = NULL;
		initN2map<float>(n2map, imgs[0]->rows, imgs[0]->cols);
#endif // N2MAP

#if AVERAGE_DEEP_3		
		short*deep_average0 = NULL, *deep_average1 = NULL, *deep_average2 = NULL;
		float*deep_average_out = NULL;
		initAverageDeep<float>(deep_average0, deep_average1, deep_average2, deep_average_out,
			imgs[1]->rows, imgs[1]->cols);
#elif AVERAGE_DEEP_5
		short*deep_average0 = NULL, *deep_average1 = NULL, *deep_average2 = NULL, *deep_average3 = NULL, *deep_average4 = NULL;
		float*deep_average_out = NULL;
		initAverageDeep(deep_average0, deep_average1, deep_average2, deep_average3, deep_average4,
			deep_average_out,
			imgs[1]->rows, imgs[1]->cols);
#endif // AVERAGE_DEEP_3
		
#if AVERAGE_DEEP_3_UPDATA
		short*deep_average0 = NULL, *deep_average1 = NULL, *deep_average2 = NULL;
		float*deep_average_out = NULL;
		initAverageDeep<float>(deep_average0, deep_average1, deep_average2, deep_average_out,
			imgs[1]->rows, imgs[1]->cols);
#elif AVERAGE_DEEP_5_UPDATA
		short*deep_average0 = NULL, *deep_average1 = NULL, *deep_average2 = NULL, *deep_average3 = NULL, *deep_average4 = NULL;
		float*deep_average_out = NULL;
		initAverageDeep(deep_average0, deep_average1, deep_average2, deep_average3, deep_average4,
			deep_average_out,
			imgs[1]->rows, imgs[1]->cols);
#elif AVERAGE_DEEP_15_UPDATA
		short*deep_average15 = NULL;
		float*deep_average_out = NULL;
		initAverageDeep(deep_average15, deep_average_out, imgs[1]->rows, imgs[1]->cols);
#endif // AVERAGE_DEEP_3_UPDATA		

#if FITDEEP_WITHNORMAL
		float*vmap_in_out = NULL;
		float*vmap0 = NULL;
		float*vmap1 = NULL;
		float*nmap0 = NULL;
		float*nmap1 = NULL;
		initFitdeep(vmap_in_out, nmap0, nmap1, vmap0, vmap1, imgs[0]->rows, imgs[0]->cols);
#endif // FITDEEP_WITHNORMAL


		double4 intr_depth;//fx,fy,cx,cy
		intr_depth.w = stream2Intr[1]->ptr<double>(0)[0];
		intr_depth.x = stream2Intr[1]->ptr<double>(1)[1];
		intr_depth.y = stream2Intr[1]->ptr<double>(0)[2];
		intr_depth.z = stream2Intr[1]->ptr<double>(1)[2];
		Mat33d R_depth(std::get<0>(stream2Extr[1]).ptr<double>(0),
			std::get<0>(stream2Extr[1]).ptr<double>(1),
			std::get<0>(stream2Extr[1]).ptr<double>(2));
		double3 t_depth;
		t_depth.x = std::get<1>(stream2Extr[1]).ptr<double>(0)[0];
		t_depth.y = std::get<1>(stream2Extr[1]).ptr<double>(1)[0];
		t_depth.z = std::get<1>(stream2Extr[1]).ptr<double>(2)[0];

		double4 intr_color;//fx,fy,cx,cy
		intr_color.w = stream2Intr[0]->ptr<double>(0)[0];
		intr_color.x = stream2Intr[0]->ptr<double>(1)[1];
		intr_color.y = stream2Intr[0]->ptr<double>(0)[2];
		intr_color.z = stream2Intr[0]->ptr<double>(1)[2];
		Mat33d R_color(std::get<0>(stream2Extr[0]).ptr<double>(0),
			std::get<0>(stream2Extr[0]).ptr<double>(1),
			std::get<0>(stream2Extr[0]).ptr<double>(2));
		double3 t_color;
		t_color.x = std::get<1>(stream2Extr[0]).ptr<double>(0)[0];
		t_color.y = std::get<1>(stream2Extr[0]).ptr<double>(1)[0];
		t_color.z = std::get<1>(stream2Extr[0]).ptr<double>(2)[0];

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
		int frameIdx = -1;
		while (1)
		{
			frameIdx++;
			//pop2Mats_noInfred(imgs);
			pop2Mats(imgs);
			
			//cv::Mat showDev0(imgs[0]->rows, imgs[0]->cols, CV_8UC3);
			//memcpy(showDev0.data, imgs[0]->data, imgs[0]->rows*imgs[0]->cols * sizeof(char)*3);

			cv::Mat showDep(imgs[1]->rows, imgs[1]->cols, CV_16SC1);
			memcpy(showDep.data, imgs[1]->data, imgs[1]->rows*imgs[1]->cols*sizeof(short));
			//cv::imshow("123", showDep);
			//cv::waitKey(10);
			//continue;

			//cv::Mat colorized;
			//transProc(imgs[0], stream2Intr[0], &std::get<0>(stream2Extr[0]), &std::get<1>(stream2Extr[0]),
			//	imgs[1], stream2Intr[1], &std::get<0>(stream2Extr[1]), &std::get<1>(stream2Extr[1]),
			//	&colorized);
			


#ifdef AVERAGE_DEEP_3
			frameIdx %= 3;
			if (frameIdx==0)
			{
				cudaMemcpy(deep_average0, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
			else if (frameIdx == 1)
			{
				cudaMemcpy(deep_average1, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
			else
			{
				cudaMemcpy(deep_average2, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
#elif AVERAGE_DEEP_5
			frameIdx %= 5;
			if (frameIdx == 0)
			{
				cudaMemcpy(deep_average0, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
			else if (frameIdx == 1)
			{
				cudaMemcpy(deep_average1, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
			else if (frameIdx == 2)
			{
				cudaMemcpy(deep_average2, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
			else if (frameIdx == 3)
			{
				cudaMemcpy(deep_average3, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
			else
			{
				cudaMemcpy(deep_average4, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
#endif // AVERAGE_DEEP_3

#ifdef AVERAGE_DEEP_3_UPDATA
			frameIdx %= 3;
			if (frameIdx == 0)
			{
				cudaMemcpy(deep_average0, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
			else if (frameIdx == 1)
			{
				cudaMemcpy(deep_average1, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
			else
			{
				cudaMemcpy(deep_average2, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			}
#elif AVERAGE_DEEP_5_UPDATA
			frameIdx %= 5;
			if (frameIdx == 0)
			{
				cudaMemcpy(deep_average0, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
				bilateralFilter<short>(deep_average0, depth_dev_bila, imgs[0]->rows, imgs[0]->cols);
				auto temp = deep_average0;
				deep_average0 = depth_dev_bila;
				depth_dev_bila = temp;
			}
			else if (frameIdx == 1)
			{
				cudaMemcpy(deep_average1, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
				bilateralFilter<short>(deep_average1, depth_dev_bila, imgs[0]->rows, imgs[0]->cols);
				auto temp = deep_average1;
				deep_average1 = depth_dev_bila;
				depth_dev_bila = temp;
			}
			else if (frameIdx == 2)
			{
				cudaMemcpy(deep_average2, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
				bilateralFilter<short>(deep_average2, depth_dev_bila, imgs[0]->rows, imgs[0]->cols);
				auto temp = deep_average2;
				deep_average2 = depth_dev_bila;
				depth_dev_bila = temp;
			}
			else if (frameIdx == 3)
			{
				cudaMemcpy(deep_average3, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
				bilateralFilter<short>(deep_average3, depth_dev_bila, imgs[0]->rows, imgs[0]->cols);
				auto temp = deep_average3;
				deep_average3 = depth_dev_bila;
				depth_dev_bila = temp;
			}
			else
			{
				cudaMemcpy(deep_average4, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
				bilateralFilter<short>(deep_average4, depth_dev_bila, imgs[0]->rows, imgs[0]->cols);
				auto temp = deep_average4;
				deep_average4 = depth_dev_bila;
				depth_dev_bila = temp;
			}
#elif AVERAGE_DEEP_15_UPDATA
			frameIdx %= 15;
			cudaMemcpy(deep_average15+ frameIdx*imgs[1]->rows * imgs[1]->cols * sizeof(short), imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
#endif
			
#if AVERAGE_DEEP_3
			combineAverageDeep<short,float>(deep_average0, deep_average1, deep_average2, deep_average_out, imgs[1]->rows , imgs[1]->cols);
#elif AVERAGE_DEEP_5
			combineAverageDeep<short, float>(deep_average0, deep_average1, deep_average2, deep_average3, deep_average4, deep_average_out, imgs[1]->rows, imgs[1]->cols);
#elif AVERAGE_DEEP_3_UPDATA
			combineAverageDeep(deep_average0, deep_average1, deep_average2, deep_average_out, imgs[1]->rows, imgs[1]->cols);
#elif AVERAGE_DEEP_5_UPDATA
			combineAverageDeep(deep_average0, deep_average1, deep_average2, deep_average3, deep_average4, deep_average_out, imgs[1]->rows, imgs[1]->cols);
#elif AVERAGE_DEEP_15_UPDATA
			combineAverageDeep(deep_average15, deep_average_out, imgs[1]->rows, imgs[1]->cols);
#else
			cudaMemcpy(depth_dev_input, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			//bilateralFilter<short>(depth_dev_input, depth_dev_bila, imgs[0]->rows, imgs[0]->cols);
#endif // !AVERAGE_DEEP_3


			{
#ifdef PCL_SHOW
				createVMap<short,double>(depth_dev_bila, vmap, intr_color.w, intr_color.x, intr_color.y, intr_color.z, imgs[0]->rows, imgs[0]->cols);
				cv::Mat onlyVmap(imgs[1]->rows, imgs[1]->cols, CV_32FC3);
				cudaMemcpy(onlyVmap.data, vmap, imgs[1]->rows*imgs[1]->cols * sizeof(float)*3, cudaMemcpyDeviceToHost);
				
				int point_cnt = 0;
				for (int i = 0; i < imgs[1]->rows; i++)
					for (int j = 0; j < imgs[1]->cols; j++)
					{
						if (!(onlyVmap.at<cv::Vec3f>(i,j)[0] == std::numeric_limits<float>::quiet_NaN()))
						{
							point_cnt++;
						}
					}
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
				cloud->width = 1;
				cloud->height = point_cnt;
				cloud->is_dense = false;
				cloud->points.resize(cloud->width * cloud->height);
				point_cnt = 0;

				for (int i = 0; i < imgs[1]->rows; i++)
					for (int j = 0; j < imgs[1]->cols; j++)
					{
						if (!(onlyVmap.at<cv::Vec3f>(i, j)[0] == std::numeric_limits<float>::quiet_NaN()))
						{
							cloud->points[point_cnt].x = onlyVmap.at<cv::Vec3f>(i, j)[0];
							cloud->points[point_cnt].y = onlyVmap.at<cv::Vec3f>(i, j)[1];
							cloud->points[point_cnt].z = onlyVmap.at<cv::Vec3f>(i, j)[2];
							point_cnt++;
						}
					}
				cloud_viewer_.removeAllPointClouds();
				cloud_viewer_.addPointCloud<pcl::PointXYZ>(cloud);
				cloud_viewer_.spinOnce(10);
				continue;
#endif //PCL_SHOW
			}
			cv::Mat showAve(imgs[1]->rows, imgs[1]->cols, CV_16SC1);
			cudaMemcpy(showAve.data, depth_dev_input, imgs[1]->rows*imgs[1]->cols * sizeof(short), cudaMemcpyDeviceToHost);


			
			colorize_deepMat(deep_average_out,
				imgs[1]->rows, imgs[1]->cols, imgs[0]->rows, imgs[0]->cols,
				intr_depth,
				R_depth, t_depth,
				intr_color,
				R_color, t_color,
				depth_dev_output
			);



			//cv::Mat showDevBeforeeMed(imgs[0]->rows, imgs[0]->cols, CV_32FC1);
			//cudaMemcpy(showDevBeforeeMed.data, depth_dev_output, imgs[0]->rows*imgs[0]->cols * sizeof(float), cudaMemcpyDeviceToHost);

			int downsample_h2 = imgs[0]->rows / 4;
			int downsample_w2 = imgs[0]->cols / 4;
			int downsample_h3 = imgs[0]->rows / 16;
			int downsample_w3 = imgs[0]->cols / 16;
			medfilter33_forOneDev(depth_dev_output, imgs[0]->rows, imgs[0]->cols,
				depth_dev_med, depth_filled,
				depth_2, downsample_h2, downsample_w2,
				depth_3, downsample_h3, downsample_w3);

			


//#define SHOW_DOWNSAMPLE
#ifdef SHOW_DOWNSAMPLE
			float2*hostDownSample2 = new float2[imgs[0]->rows* imgs[0]->cols / 16];
			cudaMemcpy((void*)hostDownSample2, (void*)depth_2, imgs[0]->rows*imgs[0]->cols * sizeof(float2) / 16, cudaMemcpyDeviceToHost);
			cv::Mat hostDownSample2_cvmat = cv::Mat(imgs[0]->rows / 4, imgs[0]->cols / 4, CV_32FC1);
			for (int i = 0; i < imgs[0]->rows / 4; i++)for (int j = 0; j < imgs[0]->cols / 4; j++)
			{
				hostDownSample2_cvmat.ptr<float>(i)[j] = hostDownSample2[i*imgs[0]->cols / 4 + j].x;
			}

			float2*hostDownSample3 = new float2[imgs[0]->rows/16* imgs[0]->cols / 16];	
			cudaMemcpy((void*)hostDownSample3, (void*)depth_3, imgs[0]->rows / 16 * imgs[0]->cols / 16*sizeof(float2), cudaMemcpyDeviceToHost);
			cv::Mat hostDownSample3_cvmat = cv::Mat(imgs[0]->rows / 16, imgs[0]->cols / 16, CV_32FC1);
			LOG(INFO) << hostDownSample3_cvmat.isContinuous();
			for (int i = 0; i < imgs[0]->rows / 16; i++)for (int j = 0; j < imgs[0]->cols / 16; j++)
			{
				hostDownSample3_cvmat.ptr<float>(i)[j] = hostDownSample3[i*imgs[0]->cols / 16 + j].x;
			}
#endif // SHOW_DOWNSAMPLE			
//#define SHOW_FILLRESULT
#ifdef SHOW_FILLRESULT
			cv::Mat showDev0(imgs[0]->rows, imgs[0]->cols, CV_32FC1);
			cudaMemcpy(showDev0.data, depth_dev_output, imgs[0]->rows*imgs[0]->cols * sizeof(float), cudaMemcpyDeviceToHost);
			cv::imshow("123", showDev0);
			cv::waitKey(10);
			//continue;
			cv::Mat showDev1(imgs[0]->rows, imgs[0]->cols, CV_32FC1);
			cudaMemcpy(showDev1.data, depth_dev_med, imgs[0]->rows*imgs[0]->cols * sizeof(float), cudaMemcpyDeviceToHost);
			cv::imshow("1234", showDev1);
			cv::waitKey(10);
			//continue;
			cv::Mat showDev2(imgs[0]->rows, imgs[0]->cols, CV_32FC1);
			cudaMemcpy(showDev2.data, depth_filled, imgs[0]->rows*imgs[0]->cols * sizeof(float), cudaMemcpyDeviceToHost);
			cv::imshow("12345", showDev2);
			cv::waitKey(10);
			continue;
#endif // SHOW_FILLRESULT
			cv::Mat showDevBeforeeVmap(imgs[0]->rows, imgs[0]->cols, CV_32FC1);
			cudaMemcpy(showDevBeforeeVmap.data, depth_filled, imgs[0]->rows*imgs[0]->cols * sizeof(float), cudaMemcpyDeviceToHost);


			createVMap<float,double>(depth_filled, vmap, intr_color.w, intr_color.x, intr_color.y, intr_color.z, imgs[0]->rows, imgs[0]->cols);

//#define SHOW_VMAP
#ifdef SHOW_VMAP
			cv::Mat showDev3(imgs[0]->rows, imgs[0]->cols, CV_32FC3);
			cudaMemcpy(showDev3.data, vmap, imgs[0]->rows*imgs[0]->cols * sizeof(float) * 3, cudaMemcpyDeviceToHost);
			cv::imshow("123", showDev3);
			cv::waitKey(10);
			//continue;
//#define SIMPLE_NMAP
#ifdef SIMPLE_NMAP
			int scope_k = 13;
			cv::Mat simple_nmap=cv::Mat::zeros(imgs[0]->rows, imgs[0]->cols, CV_32FC3);
			for (int i = scope_k; i < simple_nmap.rows- scope_k; i++)
				for (int j = scope_k; j < simple_nmap.cols- scope_k; j++)
				{
					if (std::isnan(showDev3.at<cv::Vec3f>(i + scope_k, j + scope_k)[0])||
						std::isnan(showDev3.at<cv::Vec3f>(i- scope_k, j- scope_k)[0]))
					{
						simple_nmap.at<cv::Vec3f>(i, j)[0] = 0;
						simple_nmap.at<cv::Vec3f>(i, j)[1] = 0;
						simple_nmap.at<cv::Vec3f>(i, j)[2] = 0;
						continue;
					}
					float n_x = showDev3.at<cv::Vec3f>(i + scope_k, j + scope_k)[0] - showDev3.at<cv::Vec3f>(i - scope_k, j - scope_k)[0];
					float n_y = showDev3.at<cv::Vec3f>(i + scope_k, j + scope_k)[1] - showDev3.at<cv::Vec3f>(i - scope_k, j - scope_k)[1];
					float n_z = showDev3.at<cv::Vec3f>(i + scope_k, j + scope_k)[2] - showDev3.at<cv::Vec3f>(i - scope_k, j - scope_k)[2];
					float n_mod = std::sqrtf(n_x*n_x+ n_y*n_y+ n_z*n_z);

					simple_nmap.at<cv::Vec3f>(i, j)[0] = -n_x / n_mod;
					simple_nmap.at<cv::Vec3f>(i, j)[1] = -n_y / n_mod;
					simple_nmap.at<cv::Vec3f>(i, j)[2] = n_z / n_mod;
				}
#endif // SIMPLE_NMAP
			//continue;
#endif // SHOW_VMAP

#if FITDEEP_WITHNORMAL
			fitVmap<short, float>(vmap, nmap0, nmap1, vmap0, vmap1, imgs[0]->rows, imgs[0]->cols);
			cv::Mat show_vmap0(imgs[0]->rows, imgs[0]->cols, CV_32FC3);
			cudaMemcpy(show_vmap0.data, vmap0, 3*imgs[0]->rows*imgs[0]->cols * sizeof(float), cudaMemcpyDeviceToHost);
			cv::Mat show_vmap1(imgs[0]->rows, imgs[0]->cols, CV_32FC3);
			cudaMemcpy(show_vmap1.data, vmap1, 3*imgs[0]->rows*imgs[0]->cols * sizeof(float), cudaMemcpyDeviceToHost);
#endif // FITDEEP_WITHNORMAL

			computeNormalsEigen<float>(vmap, nmap, nmap_average, imgs[0]->rows, imgs[0]->cols);
#if FITDEEP_WITHNORMAL
			computeN2ormalsEigen<float>(nmap, n2map, nmap_average, imgs[0]->rows, imgs[0]->cols);
#endif // FITDEEP_WITHNORMAL
#define SHOW_NMAP
#ifdef SHOW_NMAP
			cv::Mat showDev4(imgs[0]->rows, imgs[0]->cols, CV_32FC3);
			cudaMemcpy(showDev4.data, nmap, imgs[0]->rows*imgs[0]->cols * sizeof(float) * 3, cudaMemcpyDeviceToHost);
#if FITDEEP_WITHNORMAL
			cv::Mat showDev4_2(imgs[0]->rows, imgs[0]->cols, CV_32FC3);
			cudaMemcpy(showDev4_2.data, n2map, imgs[0]->rows*imgs[0]->cols * sizeof(float) * 3, cudaMemcpyDeviceToHost);
			cv::imshow("showDev4", showDev4_2);
#endif // FITDEEP_WITHNORMAL
			cv::imshow("showDev4", showDev4);			
			cv::waitKey(10);
			continue;
#endif // SHOW_NMAP
			


			cudaMemcpy(rgbData, imgs[0]->data, imgs[0]->rows * imgs[0]->cols * 3, cudaMemcpyHostToDevice);
			combineNmap2Rgb(rgbData, nmap,newRgbData, imgs[0]->rows, imgs[0]->cols);


			cv::Mat vis = cv::Mat(imgs[0]->rows, imgs[0]->cols,CV_8UC3);
			cudaMemcpy(vis.data, newRgbData, imgs[0]->rows*imgs[0]->cols * 3, cudaMemcpyDeviceToHost);
			cv::imshow("123", vis);
			cv::waitKey(10);

		}
		
		return 0;
	}
}