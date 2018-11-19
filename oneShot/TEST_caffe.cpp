#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

//#include <gflags/gflags.h>
//#include <glog/logging.h>
//#include<caffe\util\logg.h>
#include "../../../logg.h"
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <tuple>
#include<time.h>
#include<opencv\cv.hpp>
#include <google/protobuf/text_format.h>
//#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
//using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;


#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include"caffe/mtcnn.h"
using cv::Rect;
using cv::Mat;
using cv::imread;
using namespace std;

std::string FLAGS_gpu = "0";
std::string FLAGS_phase = "TEST";
// Parse GPU ids or use all available devices
struct Bbox {
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
	float area;
	float ppoint[10];
	float regreCoord[4];
};
static void get_gpus(vector<int>* gpus)
{
	if (FLAGS_gpu == "all")
	{
		int count = 0;
#ifndef CPU_ONLY
		CUDA_CHECK(cudaGetDeviceCount(&count));
#else
		NO_GPU;
#endif
		for (int i = 0; i < count; ++i) {
			gpus->push_back(i);
		}
	}
	else if (FLAGS_gpu.size())
	{
		vector<string> strings;
		//boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
		//for (int i = 0; i < strings.size(); ++i) {
		//	gpus->push_back(boost::lexical_cast<int>(strings[i]));
		//}
		gpus->push_back(0);
	}
	else {
		CHECK_EQ(gpus->size(), 0);
	}
}


int device_query()
{
	LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
	vector<int> gpus;
	get_gpus(&gpus);
	for (int i = 0; i < gpus.size(); ++i)
	{
		caffe::Caffe::SetDevice(gpus[i]);
		caffe::Caffe::DeviceQuery();
	}
	return 0;
}

const float threshold[3] = { 0.8f, 0.8f, 0.6f };
void generateBbox(const std::vector<float>&score,
	const std::vector<float>&location,
	std::vector<Bbox> &boundingBox_, float scale)
{
	const int stride = 2;
	const int cellsize = 12;
}

int main3()//mtcnn
{
	auto d = caffe::LayerRegistry<float>::LayerTypeList();
	for (auto dd : d) LOG(INFO) << dd;

	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size() != 0) {
		LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, gpus[0]);
		LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
		Caffe::SetDevice(gpus[0]);
		Caffe::set_mode(Caffe::GPU);
	}
	else {
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}
	std::vector<float> pixels;
	pixels.resize(3 * 1 * 800 * 800);

	Net<float> * pnet = new Net<float>("D:/repo/ncnn-MTCNN/mtcnn/det1.prototxt", caffe::TEST);
	Blob<float>* pnet_input_layer = pnet->input_blobs()[0];
	pnet->CopyTrainedLayersFrom("D:/repo/ncnn-MTCNN/mtcnn/det1.caffemodel");


	//cv::Mat img = cv::imread("D:/BaiduNetdiskDownload/lightCaffe/lbl (4).jpg");
	//cv::Mat img = cv::imread("D:/BaiduNetdiskDownload/lightCaffe/timg.jpg");
	cv::Mat img = cv::imread("D:/BaiduNetdiskDownload/lightCaffe/test1.jpg");
	cv::Size scales(800, 600);
	int target_size = scales.height;
	int max_size = scales.width;
	int im_size_min = img.cols > img.rows ? img.rows : img.cols;
	int im_size_max = img.cols < img.rows ? img.rows : img.cols;
	if (im_size_min > target_size || im_size_max > max_size)
	{
		float im_scale = float(target_size) / float(im_size_min);
		if (int(im_scale * im_size_max) > max_size)
		{
			im_scale = float(max_size) / float(im_size_max);
		}
		cv::resize(img, img, cv::Size(), im_scale, im_scale, 0);
	}

	int MIN_DET_SIZE = 12;
	int minsize = 24;
	const float pre_facetor = 0.709;
	float minl = img.cols < img.rows ? img.cols : img.rows;
	float m = (float)MIN_DET_SIZE / minsize;
	minl *= m;
	float factor = pre_facetor;
	vector<float> scales_;
	//scales_.push_back(1.);
	while (minl > MIN_DET_SIZE) {
		scales_.push_back(m);
		minl *= factor;
		m = m * factor;
	}
	for (size_t i = 0; i < scales_.size(); i++) {
		int hs = (int)ceil(img.rows * scales_[i]);
		int ws = (int)ceil(img.cols * scales_[i]);
		cv::Mat this_img;
		cv::resize(img, this_img, cv::Size(ws, hs));
		pnet_input_layer->Reshape(1, 3, this_img.rows, this_img.cols);
		int pixel_idx = 0;
		for (int k = 0; k < 3; k++)
		{
			for (int i = 0; i < this_img.rows; i++)
			{
				for (int j = 0; j < this_img.cols; j++)
				{
					pixels[pixel_idx] = ((img.at<cv::Vec3b>(i, j)[2 - k] - 127.5) / 128.);
					pixel_idx++;
				}
			}
		}
		switch (Caffe::mode())
		{
		case Caffe::CPU:
			memcpy(pnet_input_layer->mutable_cpu_data(), &pixels[0],
				sizeof(float)* pnet_input_layer->count());
			break;
#ifndef CPU_ONLY
		case Caffe::GPU:
			cudaMemcpy(pnet_input_layer->mutable_gpu_data(), &pixels[0],
				sizeof(float)* pnet_input_layer->count(), cudaMemcpyHostToDevice);
			break;
#endif
		default:
			LOG(FATAL) << "Unknown Caffe mode.";
		}
		//LOG(INFO) << caffe_net->input_blobs()[0]->shape_string();
		pnet->Forward();
		LOG(INFO) << pnet->output_blobs().size();
		Blob<float>* out_blob0 = pnet->output_blobs()[0];
		LOG(INFO) << out_blob0->shape_string();
		Blob<float>* out_blob1 = pnet->output_blobs()[1];
		LOG(INFO) << out_blob1->shape_string();
		std::vector<float> score(out_blob1->width()*out_blob1->height(), 0);
		memcpy(&score[0], out_blob1->cpu_data() + out_blob1->count() / 2, out_blob1->count() / 2);
		std::vector<float> locals(out_blob0->channels()*out_blob0->width()*out_blob0->height(), 0);
		memcpy(&locals[0], out_blob0->cpu_data(), out_blob0->count());
		float highestScore = 0;
		int highestIdx = -1;
		for (int i = 0; i<score.size(); i++)
			if (score[i] > highestScore)
			{
				highestScore = score[i];
				highestIdx = i;
			}
		LOG(INFO) << highestScore;
		LOG(INFO) << highestIdx;
		int xx = highestIdx%this_img.cols;
		int yy = highestIdx / this_img.cols;
		float a1 = locals[4 * highestIdx];
		float a2 = locals[4 * highestIdx + 1];
		float a3 = locals[4 * highestIdx + 2];
		float a4 = locals[4 * highestIdx + 3];
		cv::circle(this_img, cv::Point(xx * 2, yy * 2), 12, cv::Scalar(0, 0, 255), 3);
		std::vector<Bbox> boundingBox_;
		//generateBbox(score_, location_, boundingBox_, scales_[i]);
		//nms(boundingBox_, nms_threshold[0]);
		//firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
		boundingBox_.clear();
	}





	Net<float> * rnet2 = new Net<float>("D:/repo/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det2.prototxt", caffe::TEST);
	rnet2->CopyTrainedLayersFrom("D:/repo/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det2.caffemodel");
	Net<float> * onet = new Net<float>("D:/repo/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det3.prototxt", caffe::TEST);
	onet->CopyTrainedLayersFrom("D:/repo/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det3.caffemodel");


	return 0;
}

int main4() {

	vector<string> model_file = {
		"D:/repo/MTCNN/model/det1.prototxt",
		"D:/repo/MTCNN/model/det2.prototxt",
		"D:/repo/MTCNN/model/det3.prototxt"
		//            "../model/det4.prototxt"
	};

	vector<string> trained_file = {
		"D:/repo/MTCNN/model/det1.caffemodel",
		"D:/repo/MTCNN/model/det2.caffemodel",
		"D:/repo/MTCNN/model/det3.caffemodel"
		//            "../model/det4.caffemodel"
	};

	MTCNN mtcnn(model_file, trained_file);

	vector<Rect> rectangles;
	string img_path = "D:/lightCaffe/test1.jpg";
	Mat img = imread(img_path);

	int count = 1;
	unsigned start = time(NULL);
	for (int i = 0; i < count; i++) {
		mtcnn.detection_TEST(img, rectangles);
		std::cout << mtcnn.bounding_box_.size() << std::endl;
		for (size_t j = 0; j < mtcnn.bounding_box_.size(); j++)
		{
			cv::rectangle(img,
				cv::Rect(mtcnn.bounding_box_[j].y, mtcnn.bounding_box_[j].x,
					mtcnn.bounding_box_[j].height, mtcnn.bounding_box_[j].width)
				, cv::Scalar(0, 0, 255), 2);
		}

	}
	unsigned end = time(NULL);
	unsigned ave = (end - start)*1000.0 / count;
	std::cout << "Run " << count << " times, " << "Average time:" << ave << std::endl;

	return 0;
}
