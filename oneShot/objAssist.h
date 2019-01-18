#ifndef _OBJ_ASSIST_H_
#define _OBJ_ASSIST_H_
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include "opencv2/opencv.hpp"
#include "logg.h"
#include "stringOp.h"


namespace unre
{
	class ObjAssist
	{
	public:		
		static int getSphereObjData(const char*objFilePath,
			std::vector<cv::Point3f>&points,
			std::vector<cv::Point3f>&norm,
			std::vector<cv::Point3i>&face,
			const float radius)
		{  
			std::fstream fin(objFilePath,std::ios::in);
			std::string aline;
			while (std::getline(fin,aline))
			{
				auto seg = StringOP::splitString(aline," ");
				if (seg[0].compare("v")==0)
				{
					CHECK(seg.size()==4);
					std::stringstream ss;
					ss << aline;
					cv::Point3f v;
					ss >> aline >> v.x >> v.y >> v.z;
					points.emplace_back(std::move(v));
				}
				else if(seg[0].compare("vn") == 0)
				{
					CHECK(seg.size() == 4);
					std::stringstream ss;
					ss << aline;
					cv::Point3f vn;
					ss >> aline >> vn.x >> vn.y >> vn.z;
					norm.emplace_back(std::move(vn));
				}
				else if (seg[0].compare("f") == 0)
				{
					CHECK(seg.size() == 4);
					cv::Point3i f;
					auto seg2 = StringOP::splitString(seg[1], "\\");
					f.x = atoi(seg2[0].c_str());
					seg2 = StringOP::splitString(seg[2], "\\");
					f.y = atoi(seg2[0].c_str());
					seg2 = StringOP::splitString(seg[3], "\\");
					f.z = atoi(seg2[0].c_str());
					face.emplace_back(std::move(f));
				}
			}
			cv::Point3f center(0, 0, 0);
			for (auto&d : points)center += d;
			center.x /= points.size();
			center.y /= points.size();
			center.z /= points.size();
			for (auto&d : points)
			{
				cv::Point3f this_p = d;
				float length = this_p.x*this_p.x
					+ this_p.y*this_p.y
					+ this_p.z*this_p.z;
				length = std::sqrtf(length);
				d=(this_p / length*radius);
			}
			return 0;
		}
	private:
		ObjAssist();
		~ObjAssist();

	};

	
}


#endif // !_OBJ_ASSIST_H_
