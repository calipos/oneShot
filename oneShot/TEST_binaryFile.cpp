#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
using namespace std;

int TEST_binaryFile()
{
	//float z[3] = {1.2,3.4,5.6};
	//std::fstream fout("binaryFileTest",std::ios::out|std::ios::binary);
	//fout.write((char*)z, sizeof(float) * 3);
	//fout.close();

	//float y[3];
	//fstream binary_file("binaryFileTest", ios::binary | ios::in);
	//binary_file.read(reinterpret_cast<char *>(&y), sizeof(float)*3);
	//binary_file.close();
	//cout << y[0] << endl;
	//cout <<y[1] << endl;
	//cout << y[2] << endl;


	float *binaryPoints_new = new float[3 * 427*240];
	float *binaryNorms_new = new float[3 * 427 * 240];
	float *binaryRgba_new = new float[4 * 427 * 240];

	fstream binary_file1("aah.point", ios::binary | ios::in);
	binary_file1.read(reinterpret_cast<char *>(binaryPoints_new), sizeof(float) * 3 * 427 * 240);
	binary_file1.close();

	fstream binary_file2("aah.norm", ios::binary | ios::in);
	binary_file2.read(reinterpret_cast<char *>(binaryNorms_new), sizeof(float) * 3 * 427 * 240);
	binary_file2.close();

	fstream binary_file3("aah.rgba", ios::binary | ios::in);
	binary_file3.read(reinterpret_cast<char *>(binaryRgba_new), sizeof(float) * 4 * 427 * 240);
	binary_file3.close();

	cv::Mat points = cv::Mat(240, 427, CV_32FC3);
	cv::Mat norms = cv::Mat(240, 427, CV_32FC3);
	cv::Mat rgba = cv::Mat(240, 427, CV_32FC4);
	memcpy(points.data, binaryPoints_new, sizeof(float) * 3 * 427 * 240);
	memcpy(norms.data, binaryNorms_new, sizeof(float) * 3 * 427 * 240);
	memcpy(rgba.data, binaryRgba_new, sizeof(float) * 3 * 427 * 240);



	return 0;
}