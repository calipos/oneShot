#include<algorithm>
#include<chrono>
#include"stringOp.h"
#include "logg.h"
#include "iofile.h"
#include "dataExplorer.h"
#include "unreGpu.h"
#include "unityStruct.h"
#define OUT_BINARY
//#define PCL_SHOW
#ifdef PCL_SHOW
#include "pcl/visualization/cloud_viewer.h"
#endif // PCL_SHOW


bool IsNaN(float& dat)
{
	int & ref = *(int *)&dat;
	return (ref & 0x7F800000) == 0x7F800000 && (ref & 0x7FFFFF) != 0;
}




int writeAtxt(float *vmap, unsigned char *rgb, float *nmap, int rows, int cols, int frameIdx)
{
	std::vector<long long> indexs(rows*cols,-1);
	std::vector<cv::Point3f> v;
	std::vector<cv::Point3i> vcolor;
	std::vector<cv::Point3f> n;
	std::vector<long long> triSet;
	v.reserve(rows*cols);
	n.reserve(rows*cols);
	triSet.reserve(rows*cols);

	for (int i = 1; i <rows - 1; i++)
	{
		for (int j = 1; j < cols-1; j++)
		{
			int thisPos = i*cols + j;
			if (IsNaN(vmap[3* thisPos]) || (nmap[3 * thisPos] == 0 && nmap[3 * thisPos+1] == 0 && nmap[3 * thisPos+2] == 0 ))
			{
				continue;
			}
			v.push_back(cv::Point3f(vmap[3 * thisPos], -vmap[3 * thisPos+1], vmap[3 * thisPos+2]-0.5));			
			vcolor.push_back(cv::Point3i(rgb[3 * thisPos+2], rgb[3 * thisPos + 1], rgb[3 * thisPos]));
			n.push_back(cv::Point3f(nmap[3 * thisPos], nmap[3 * thisPos + 1], nmap[3 * thisPos + 2]));
			indexs[thisPos] = v.size() - 1;
		}
	}
	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			int thisPos0 = i*cols + j;
			int thisPos1 = i*cols + j+1;
			int thisPos2 = i*cols + j + cols;
			int thisPos3 = i*cols + j + cols + 1;
			if (indexs[thisPos0]>0 && indexs[thisPos1]>0 && indexs[thisPos2]>0 && indexs[thisPos3]>0)
			{
				triSet.push_back(indexs[thisPos0]);
				triSet.push_back(indexs[thisPos1]);
				triSet.push_back(indexs[thisPos2]);
				triSet.push_back(indexs[thisPos1]);
				triSet.push_back(indexs[thisPos3]);
				triSet.push_back(indexs[thisPos2]);
			}
			else if (indexs[thisPos0]>0 && indexs[thisPos1]>0 && indexs[thisPos2]>0 && indexs[thisPos3]<0)
			{
				triSet.push_back(indexs[thisPos0]);
				triSet.push_back(indexs[thisPos1]);
				triSet.push_back(indexs[thisPos2]);
			}
			else if (indexs[thisPos0]>0 && indexs[thisPos1]>0 && indexs[thisPos2]<0 && indexs[thisPos3]>0)
			{
				triSet.push_back(indexs[thisPos0]);
				triSet.push_back(indexs[thisPos1]);
				triSet.push_back(indexs[thisPos3]);
			}
			else if (indexs[thisPos0]>0 && indexs[thisPos1]<0 && indexs[thisPos2]>0 && indexs[thisPos3]>0)
			{
				triSet.push_back(indexs[thisPos0]);
				triSet.push_back(indexs[thisPos3]);
				triSet.push_back(indexs[thisPos2]);
			}
			else if (indexs[thisPos0]<0 && indexs[thisPos1]>0 && indexs[thisPos2]>0 && indexs[thisPos3]>0)
			{
				triSet.push_back(indexs[thisPos1]);
				triSet.push_back(indexs[thisPos3]);
				triSet.push_back(indexs[thisPos2]);
			}
		}
	}
	std::fstream fout(std::to_string(frameIdx)+".txt",std::ios::out);
	for (int i = 0; i < v.size(); i++)
	{
		fout << i << " " << v[i].x << " " << v[i].y << " " << v[i].z << " " << vcolor[i].x << " " << vcolor[i].y << " " << vcolor[i].z << " " << n[i].x << " " << n[i].y << " " << n[i].z<< std::endl;
	}
	for (int i = 0; i < triSet.size()/3; i++)
	{
		fout << "t " << triSet[3 * i] << " " << triSet[3 * i+1] << " " << triSet[3 * i+2] << std::endl;
	}
	fout.close();
	return 0;
}
int writeAtxt2(float *vmap, unsigned char *rgb, float *nmap, int rows, int cols, int frameIdx)
{
	std::vector<long long> indexs(rows*cols, -1);
	std::vector<cv::Point3f> v;
	std::vector<cv::Point3i> vcolor;
	std::vector<cv::Point3f> n;
	std::vector<long long> triSet;
	v.reserve(rows*cols);
	n.reserve(rows*cols);
	triSet.reserve(rows*cols);

	for (int i = 1; i <rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			if (i%3!=0 || j % 3 != 0)
			{
				continue;
			}
			int thisPos = i*cols + j;
			if (IsNaN(vmap[3 * thisPos]) || (nmap[3 * thisPos] == 0 && nmap[3 * thisPos + 1] == 0 && nmap[3 * thisPos + 2] == 0))
			{
				continue;
			}
			v.push_back(cv::Point3f(vmap[3 * thisPos], -vmap[3 * thisPos + 1], vmap[3 * thisPos + 2] - 0.5));
			vcolor.push_back(cv::Point3i(rgb[3 * thisPos + 2], rgb[3 * thisPos + 1], rgb[3 * thisPos]));
			n.push_back(cv::Point3f(nmap[3 * thisPos], nmap[3 * thisPos + 1], nmap[3 * thisPos + 2]));
			indexs[thisPos] = v.size() - 1;
		}
	}
	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			int thisPos0 = i*cols + j;
			int thisPos1 = i*cols + j + 3;
			int thisPos2 = i*cols + j + 3*cols;
			int thisPos3 = i*cols + j + 3*cols + 3;
			if (indexs[thisPos0]>0 && indexs[thisPos1]>0 && indexs[thisPos2]>0 && indexs[thisPos3]>0)
			{
				triSet.push_back(indexs[thisPos0]);
				triSet.push_back(indexs[thisPos1]);
				triSet.push_back(indexs[thisPos2]);
				triSet.push_back(indexs[thisPos1]);
				triSet.push_back(indexs[thisPos3]);
				triSet.push_back(indexs[thisPos2]);
			}
			else if (indexs[thisPos0]>0 && indexs[thisPos1]>0 && indexs[thisPos2]>0 && indexs[thisPos3]<0)
			{
				triSet.push_back(indexs[thisPos0]);
				triSet.push_back(indexs[thisPos1]);
				triSet.push_back(indexs[thisPos2]);
			}
			else if (indexs[thisPos0]>0 && indexs[thisPos1]>0 && indexs[thisPos2]<0 && indexs[thisPos3]>0)
			{
				triSet.push_back(indexs[thisPos0]);
				triSet.push_back(indexs[thisPos1]);
				triSet.push_back(indexs[thisPos3]);
			}
			else if (indexs[thisPos0]>0 && indexs[thisPos1]<0 && indexs[thisPos2]>0 && indexs[thisPos3]>0)
			{
				triSet.push_back(indexs[thisPos0]);
				triSet.push_back(indexs[thisPos3]);
				triSet.push_back(indexs[thisPos2]);
			}
			else if (indexs[thisPos0]<0 && indexs[thisPos1]>0 && indexs[thisPos2]>0 && indexs[thisPos3]>0)
			{
				triSet.push_back(indexs[thisPos1]);
				triSet.push_back(indexs[thisPos3]);
				triSet.push_back(indexs[thisPos2]);
			}
		}
	}
	std::fstream fout(std::to_string(frameIdx) + ".txt", std::ios::out);
	for (int i = 0; i < v.size(); i++)
	{
		fout << i << " " << v[i].x << " " << v[i].y << " " << v[i].z << " " << vcolor[i].x << " " << vcolor[i].y << " " << vcolor[i].z << " " << n[i].x << " " << n[i].y << " " << n[i].z << std::endl;
	}
	for (int i = 0; i < triSet.size() / 3; i++)
	{
		fout << "t " << triSet[3 * i] << " " << triSet[3 * i + 1] << " " << triSet[3 * i + 2] << std::endl;
	}
	fout.close();
	return 0;
}

int generTris(const char*triBinaryFilePath, const int&rowPoints,const int &colPoints)
{
	int triNum = (rowPoints - 1)*(colPoints - 1) * 2;
	unsigned int*indexs = new unsigned int[triNum * 3];
	int ii = 0;
	for (int i = 0; i < rowPoints-1; i++)
	{
		for (int j = 0; j < colPoints - 1; j++)
		{
			int leftTop = i*colPoints + j;
			int rightTop = i*colPoints + j+1;
			int leftBottom = i*colPoints + colPoints+ j;
			int rightBottom = i*colPoints + colPoints + j+1;
			indexs[ii++] = leftTop;
			indexs[ii++] = rightTop;
			indexs[ii++] = leftBottom;
			indexs[ii++] = rightTop;
			indexs[ii++] = rightBottom;
			indexs[ii++] = leftBottom;
		}
	}
	if (1)//out binary file
	{
		std::fstream fout(triBinaryFilePath, std::ios::out | std::ios::binary);
		fout.write((char*)indexs, sizeof(unsigned int)*triNum * 3);
		fout.close();
	}
	delete[]indexs;
	return triNum * 3;
}
int generRgbNormPos(float *vmap, unsigned char *rgb, float *nmap)
{
	return 0;
}

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
		float*nmap_filled = NULL;
		float*vmap = NULL;
		unsigned char*rgbData = NULL;
		unsigned char*newRgbData = NULL;
		initOneDevDeep(depth_dev_input,depth_dev_output, depth_dev_bila,depth_dev_med, depth_filled, depth_2, depth_3,vmap, nmap, nmap_filled,rgbData, newRgbData, imgs[1]->rows, imgs[1]->cols, imgs[0]->rows, imgs[0]->cols);
				
#ifdef OUT_BINARY
		std::vector<std::tuple<float*, float*, float*>> frameBuffer;
		int sample_h = imgs[0]->rows / SAMPLE_H + (imgs[0]->rows % SAMPLE_H > 0 ? 1 : 0);
		int sample_w = imgs[0]->cols / SAMPLE_W + (imgs[0]->cols % SAMPLE_W > 0 ? 1 : 0);
		float *binaryPoints = new float[3 * sample_h*sample_w];
		float *binaryNorms = new float[3 * sample_h*sample_w];
		float *binaryRgba = new float[4 * sample_h*sample_w];
		initUnityData(sample_h, sample_w);
#endif // OUT_BINARY


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
		
		double4 intr_depth;//fx,fy,cx,cy
		Mat33d R_depth(std::get<0>(stream2Extr[1]).ptr<double>(0),
			std::get<0>(stream2Extr[1]).ptr<double>(1),
			std::get<0>(stream2Extr[1]).ptr<double>(2));
		double3 t_depth;
		double4 intr_color;//fx,fy,cx,cy
		Mat33d R_color(std::get<0>(stream2Extr[0]).ptr<double>(0),
			std::get<0>(stream2Extr[0]).ptr<double>(1),
			std::get<0>(stream2Extr[0]).ptr<double>(2));
		double3 t_color;
		{
			intr_depth.w = stream2Intr[1]->ptr<double>(0)[0];
			intr_depth.x = stream2Intr[1]->ptr<double>(1)[1];
			intr_depth.y = stream2Intr[1]->ptr<double>(0)[2];
			intr_depth.z = stream2Intr[1]->ptr<double>(1)[2];
			t_depth.x = std::get<1>(stream2Extr[1]).ptr<double>(0)[0];
			t_depth.y = std::get<1>(stream2Extr[1]).ptr<double>(1)[0];
			t_depth.z = std::get<1>(stream2Extr[1]).ptr<double>(2)[0];
			intr_color.w = stream2Intr[0]->ptr<double>(0)[0];
			intr_color.x = stream2Intr[0]->ptr<double>(1)[1];
			intr_color.y = stream2Intr[0]->ptr<double>(0)[2];
			intr_color.z = stream2Intr[0]->ptr<double>(1)[2];
			t_color.x = std::get<1>(stream2Extr[0]).ptr<double>(0)[0];
			t_color.y = std::get<1>(stream2Extr[0]).ptr<double>(1)[0];
			t_color.z = std::get<1>(stream2Extr[0]).ptr<double>(2)[0];
		}
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
		int frameIdx2write = 0;
		while (1)
		{
			frameIdx++;
			//pop2Mats_noInfred(imgs);
			pop2Mats(imgs);
			
			//cv::Mat showDev0(imgs[0]->rows, imgs[0]->cols, CV_8UC3);
			//memcpy(showDev0.data, imgs[0]->data, imgs[0]->rows*imgs[0]->cols * sizeof(char)*3);

			cv::Mat showDep(imgs[1]->rows, imgs[1]->cols, CV_16SC1);
			memcpy(showDep.data, imgs[1]->data, imgs[1]->rows*imgs[1]->cols*sizeof(short));


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
				createVMap<float, double>(deep_average_out, vmap, intr_depth.w, intr_depth.x, intr_depth.y, intr_depth.z, imgs[1]->rows, imgs[1]->cols);
				cv::Mat onlyVmap(imgs[1]->rows, imgs[1]->cols, CV_32FC3);
				cudaMemcpy(onlyVmap.data, vmap, imgs[1]->rows*imgs[1]->cols * sizeof(float) * 3, cudaMemcpyDeviceToHost);

				int point_cnt = 0;
				for (int i = 0; i < imgs[1]->rows; i++)
					for (int j = 0; j < imgs[1]->cols; j++)
					{
						if (!(onlyVmap.at<cv::Vec3f>(i, j)[0] == std::numeric_limits<float>::quiet_NaN()))
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


			cv::Mat showAve(imgs[1]->rows, imgs[1]->cols, CV_32FC1);
			cudaMemcpy(showAve.data, deep_average_out, imgs[1]->rows*imgs[1]->cols * sizeof(float), cudaMemcpyDeviceToHost);
						
			colorize_deepMat(deep_average_out,
				imgs[1]->rows, imgs[1]->cols, imgs[0]->rows, imgs[0]->cols,
				intr_depth,
				R_depth, t_depth,
				intr_color,
				R_color, t_color,
				depth_dev_output
			);
			cv::Mat showDevBeforeeMed(imgs[0]->rows, imgs[0]->cols, CV_32FC1);
			cudaMemcpy(showDevBeforeeMed.data, depth_dev_output, imgs[0]->rows*imgs[0]->cols * sizeof(float), cudaMemcpyDeviceToHost);
			//continue;

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
			
			createVMap<float,double>(depth_dev_med, vmap, intr_color.w, intr_color.x, intr_color.y, intr_color.z, imgs[0]->rows, imgs[0]->cols);
			
//#define SHOW_VMAP
#ifdef SHOW_VMAP
			cv::Mat showDev3(imgs[0]->rows, imgs[0]->cols, CV_32FC3);
			cudaMemcpy(showDev3.data, vmap, imgs[0]->rows*imgs[0]->cols * sizeof(float) * 3, cudaMemcpyDeviceToHost);
			//cv::imshow("123", showDev3);
			//cv::waitKey(10);
			//continue;

			//continue;
#endif // SHOW_VMAP
			
			computeNormalsEigen<float>(vmap, nmap, nmap_filled, imgs[0]->rows, imgs[0]->cols);
			
			cudaMemcpy(rgbData, imgs[0]->data, imgs[0]->rows * imgs[0]->cols * 3, cudaMemcpyHostToDevice);
#ifdef OUT_BINARY			
			sampleUnityData(vmap, nmap, rgbData,intr_color,sample_h, sample_w, imgs[0]->cols);
			float *binaryPoints_new = new float[3 * sample_h*sample_w];
			float *binaryNorms_new = new float[3 * sample_h*sample_w];
			float *binaryRgba_new = new float[4 * sample_h*sample_w];
			device2Host(binaryPoints_new, binaryNorms_new, binaryRgba_new);
			frameBuffer.push_back(std::make_tuple(binaryPoints_new, binaryNorms_new, binaryRgba_new));


			if (frameBuffer.size()>15)
			{
				break;
			}
#endif // OUT_BINARY
#define SHOW_NMAP
#ifdef SHOW_NMAP
			cv::Mat showDev4(imgs[0]->rows, imgs[0]->cols, CV_32FC3);
			cudaMemcpy(showDev4.data, nmap, imgs[0]->rows*imgs[0]->cols * sizeof(float) * 3, cudaMemcpyDeviceToHost);
			cv::imshow("showDev4", showDev4);
			cv::waitKey(10);
			continue;
#endif // SHOW_NMAP

			combineNmap2Rgb(rgbData, nmap,newRgbData, imgs[0]->rows, imgs[0]->cols);


			cv::Mat vis = cv::Mat(imgs[0]->rows, imgs[0]->cols,CV_8UC3);
			cudaMemcpy(vis.data, newRgbData, imgs[0]->rows*imgs[0]->cols * 3, cudaMemcpyDeviceToHost);
			cv::imshow("123", vis);
			cv::waitKey(10);

		}
		generTris("tris",240,427);
		return 0;
		for (size_t i = 0; i < frameBuffer.size(); i++)
		{
			char a1 = i / 26 / 26 + 97;
			char a2 = i / 26 + 97;
			char a3 = i % 26 + 97;
			std::string thisFrameName = std::string("") + a1 + a2 + a3;
			std::fstream fout1(thisFrameName + ".point", std::ios::out | std::ios::binary);
			fout1.write((char*)std::get<0>(frameBuffer[i]), sizeof(float) * sample_h*sample_w * 3);
			fout1.close();
			std::fstream fout2(thisFrameName + ".norm", std::ios::out | std::ios::binary);
			fout2.write((char*)std::get<1>(frameBuffer[i]), sizeof(float) * sample_h*sample_w * 3);
			fout2.close();
			std::fstream fout3(thisFrameName + ".rgba", std::ios::out | std::ios::binary);
			fout3.write((char*)std::get<2>(frameBuffer[i]), sizeof(float) * sample_h*sample_w * 4);
			fout3.close();
			//cv::Mat points = cv::Mat(240, 427, CV_32FC3);
			//cv::Mat norms = cv::Mat(240, 427, CV_32FC3);
			//cv::Mat rgba = cv::Mat(240, 427, CV_32FC4);
			//memcpy(points.data, std::get<0>(frameBuffer[i]), sizeof(float) * 3 * 427 * 240);
			//memcpy(norms.data, std::get<1>(frameBuffer[i]), sizeof(float) * 3 * 427 * 240);
			//memcpy(rgba.data, std::get<2>(frameBuffer[i]), sizeof(float) * 4 * 427 * 240);
		}
		return 0;
	}
}