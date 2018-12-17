#include<algorithm>
#include<chrono>
#include"stringOp.h"
#include"logg.h"
#include"iofile.h"
#include"dataExplorer.h"

#include"unreGpu.h"





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
		initOneDevDeep(depth_dev_input,depth_dev_output, imgs[1]->rows, imgs[1]->cols, imgs[0]->rows, imgs[0]->cols);

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

	
		while (1)
		{
			//pop2Mats_noInfred(imgs);
			pop2Mats(imgs);
			//for (size_t i = 0; i < imgs.size(); i++)
			//{
			//	if (imgs[i]->type()==CV_8UC1)
			//	{
			//		continue;
			//	}
			//	cv::imshow(std::to_string(i), *(imgs[i]));
			//}
			//cv::waitKey(12);
			//cudaMemcpy(volume___, volume, VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(short2), cudaMemcpyDeviceToHost);
			cv::Mat showDev0(imgs[0]->rows, imgs[0]->cols, CV_8UC3);
			memcpy(showDev0.data, imgs[0]->data, imgs[0]->rows*imgs[0]->cols * sizeof(char)*3);

			cv::Mat showDev1(imgs[1]->rows, imgs[1]->cols, CV_16SC1);
			memcpy(showDev1.data, imgs[1]->data, imgs[1]->rows*imgs[1]->cols*sizeof(short));
			cudaMemcpy(depth_dev_input, imgs[1]->data, imgs[1]->rows * imgs[1]->cols * sizeof(short), cudaMemcpyHostToDevice);
			colorize_deepMat(depth_dev_input,
				imgs[1]->rows, imgs[1]->cols, imgs[0]->rows, imgs[0]->cols,
				intr_depth,
				R_depth, t_depth,
				intr_color,
				R_color, t_color,
				depth_dev_output
			);
			cv::Mat showDev2(imgs[0]->rows, imgs[0]->cols,CV_32FC1);
			cudaMemcpy(showDev2.data, depth_dev_output, imgs[0]->rows*imgs[0]->cols * sizeof(float), cudaMemcpyDeviceToHost);
		}
		
		return 0;
	}
}