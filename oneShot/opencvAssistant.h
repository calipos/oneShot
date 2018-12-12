#ifndef _OPENCVASSISTANT_H_
#define _OPENCVASSISTANT_H_
#include<opencv2/opencv.hpp>

int checkCandidateCornersOrder(std::vector<cv::Point2f>&points, const cv::Size&chessBorad);
#endif