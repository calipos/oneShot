#include "opencvAssistant.h"
float getLineK(const cv::Point2f&p1, const cv::Point2f&p2)
{
	if (std::abs(p1.x - p2.x)<1e-4)
	{
		return FLT_MAX;
	}
	else
	{
		return   (p1.y - p2.y)/ (p1.x - p2.x);
	}
}
struct headStruct
{
	float pos_;
	int index_;
	headStruct(float pos,int index)
	{
		pos_ = pos;
		index_ = index;
	}
};
int checkCandidateCornersOrder(std::vector<cv::Point2f>&points, const cv::Size&chessBorad)
{
	if (chessBorad.width == chessBorad.height || points.size() != chessBorad.width * chessBorad.height)
	{
		return -1;
	}
	const cv::Point2f&sample1 = points[chessBorad.width-1];
	const cv::Point2f&sample2 = points[chessBorad.width - 2];
	const cv::Point2f&sample3 = points[chessBorad.width ];
	float k1 = getLineK(sample1, sample2);
	float k2 = getLineK(sample1, sample3);
	std::vector<cv::Point2f> sortedPoints;
	if (std::abs(k1 - k2)<1e-1)//no upRight
	{
		//检测出来的宽高反了，每一排其实是一列点
		std::vector<headStruct> rowIdxs;;
		for (size_t y = 0; y < chessBorad.width; y++)//y这里代表一列
		{
			float thisLineK = getLineK(points[y*chessBorad.height], points[y*chessBorad.height + 1]);
			rowIdxs.push_back(headStruct(points[y*chessBorad.height].x, y));;
			for (size_t x = 2; x < chessBorad.height; x++)
			{
				float thisShortLineK = getLineK(points[y*chessBorad.height +x- 1], points[y*chessBorad.height + x]);
				if (std::abs(thisLineK - thisShortLineK)>1e-1)
				{
					return -1;//重新标定
				}
			}
		}
		std::sort(rowIdxs.begin(), rowIdxs.end(), [](auto&i, auto&j) {return i.pos_<j.pos_; });		
		std::vector<std::vector<cv::Point2f>> eachCols;
		eachCols.resize(chessBorad.width);
		
		for (int idx = 0; idx < rowIdxs.size(); idx++)
		{
			int r=rowIdxs[idx].index_;//第几列
			for (int h = 0; h < chessBorad.height; h++)
			{
				eachCols[idx].push_back(points[chessBorad.height*r + h]);
			}
			std::sort(eachCols[r].begin(), eachCols[r].end(), [](auto i, auto j) {return i.y>j.y; });
		}
		points.clear();
		for (int h = 0; h < chessBorad.height; h++)
		{
			for (int w = 0; w < eachCols.size(); w++)
			{
				points.push_back(eachCols[w][h]);
			}
		}				
	}
	else//upRight
	{
		//检测出来的宽高正确了，每一排就是一排
		std::vector<headStruct> rowIdxs;;
		for (size_t y = 0; y < chessBorad.height; y++)//y这里代表一pai
		{
			float thisLineK = getLineK(points[y*chessBorad.width], points[y*chessBorad.width + 1]);
			rowIdxs.push_back(headStruct(points[y*chessBorad.width].y, y));;
			for (size_t x = 2; x < chessBorad.width; x++)
			{
				float thisShortLineK = getLineK(points[y*chessBorad.width + x - 1], points[y*chessBorad.width + x]);
				if (std::abs(thisLineK - thisShortLineK)>1e-1)
				{
					return -1;//重新标定
				}
			}
		}
		std::sort(rowIdxs.begin(), rowIdxs.end(), [](auto&i, auto&j) {return i.pos_>j.pos_; });//y值大的往前排
		for (int idx = 0; idx < rowIdxs.size(); idx++)
		{
			int c = rowIdxs[idx].index_;//第几列
			std::vector<cv::Point2f> thisRows;
			for (int w = 0; w < chessBorad.width; w++)
			{
				thisRows.push_back(points[chessBorad.width*c + w]);
			}
			std::sort(thisRows.begin(), thisRows.end(), [](auto i, auto j) {return i.x<j.x; });
			sortedPoints.insert(sortedPoints.end(), thisRows.begin(), thisRows.end());
		}
		points.clear();
		points.insert(points.end(), sortedPoints.begin(), sortedPoints.end());
	}	
	return 0;
}
