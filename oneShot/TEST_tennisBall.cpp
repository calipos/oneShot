#include "unreGpu.h"
#include "logg.h"
#include "objAssist.h"
#define PCL_SHOW
#ifdef PCL_SHOW
#include "pcl/visualization/cloud_viewer.h"
#endif // PCL_SHOW
#ifdef GenerMeshData
#include "pcl/visualization/cloud_viewer.h"
#endif // PCL_SHOW

int TEST_tennisBall()
{
	std::vector<cv::Point3f> points;
	std::vector<cv::Point3f> norms;
	std::vector<cv::Point3i> faces;
	unre::ObjAssist::getSphereObjData("IcoSphere.obj",points,norms,faces, 0.02);
	  

#ifdef PCL_SHOW 
	pcl::visualization::PCLVisualizer cloud_viewer_;
	cloud_viewer_.setBackgroundColor(0, 0, 0.15);
	cloud_viewer_.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1);
	cloud_viewer_.addCoordinateSystem(1.0,0,0,0, "global");
	cloud_viewer_.setCameraPosition(2.5,-1,2.5, -0.15,-0.89,-0.42, 1, 1, 1); //设置坐标原点
	cloud_viewer_.initCameraParameters();
	cloud_viewer_.setPosition(0, 0);
	cloud_viewer_.setSize(640, 360);
	cloud_viewer_.setCameraClipDistances(0,0);
#endif
#ifdef GenerMeshData 
	pcl::visualization::PCLVisualizer cloud_viewer_;
	cloud_viewer_.setBackgroundColor(0, 0, 0.15);
	cloud_viewer_.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1);
	cloud_viewer_.addCoordinateSystem(1.0, "global");
	cloud_viewer_.initCameraParameters();
	cloud_viewer_.setPosition(0, 0);
	cloud_viewer_.setSize(640, 360);
	cloud_viewer_.setCameraClipDistances(0.01, 10.01);
#endif
	int ballNum = initTennisBalls();
#ifdef PCL_SHOW
	float*pos_host = new float[ballNum * 3];
	float*vel_host = new float[ballNum * 3];
	unsigned char*rgb_host = new unsigned char[ballNum*3];
#endif
#ifdef GenerMeshData
	initMeshData_dev(points.size(), norms.size());
	float *pos_host = new float[ballNum*points.size() * 3];
	float *norm_host = new float[ballNum*norms.size() * 3];
	unsigned int *triIdx_host = new unsigned int[ballNum*norms.size() * 3];
	unsigned char *rgb_host = new unsigned char[ballNum*norms.size() * 3];
#endif // GenerMeshData
	int time = 50;
	while (time--)
	{
#ifdef PCL_SHOW
		loopProc(0.5, pos_host, vel_host, rgb_host);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud->width = 1;
		cloud->height = ballNum;
		cloud->is_dense = false;
		cloud->points.resize(cloud->width * cloud->height);
		for (size_t i = 0; i < ballNum; i++)
		{
			cloud->points[i].x = pos_host[3 * i];
			cloud->points[i].y = pos_host[3 * i + 1];
			cloud->points[i].z = pos_host[3 * i + 2];
			cloud->points[i].r = rgb_host[3 * i];
			cloud->points[i].g = rgb_host[3 * i + 1];
			cloud->points[i].b = rgb_host[3 * i + 2];
		}
		cloud_viewer_.removeAllPointClouds();
		cloud_viewer_.addPointCloud<pcl::PointXYZRGB>(cloud);
		cloud_viewer_.spinOnce(100);
		pcl::visualization::Camera camPos;
		cloud_viewer_.getCameraParameters(camPos);
#else
		loopProc(0.05);
#ifdef GenerMeshData
		getMeshData(pos_host, norm_host, triIdx_host, rgb_host,
			(float*)&points[0], (float*)&norms[0], (unsigned int*)&faces[0],
			points.size(), norms.size());
		
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		cloud->width = 1;
		cloud->height = ballNum*points.size();
		cloud->is_dense = false;
		cloud->points.resize(cloud->width * cloud->height);
		for (size_t i = 0; i < points.size()*ballNum; i++)
		{
			cloud->points[i].x = pos_host[3 * i];
			cloud->points[i].y = pos_host[3 * i + 1];
			cloud->points[i].z = pos_host[3 * i + 2];
			//memcpy(&(cloud->points[i].x), &pos_host[3 * i], points.size()*3*sizeof(float));
		}
		
		cloud_viewer_.removeAllPointClouds();
		cloud_viewer_.addPointCloud<pcl::PointXYZ>(cloud);
		cloud_viewer_.spinOnce(1);		
#endif //GenerMeshData
#endif
	}
	while (true)
	{
		time = 400;
		initTennisBalls();
		while (time--)
		{
#ifdef PCL_SHOW
			loopProc(0.5, pos_host, vel_host, rgb_host);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			cloud->width = 1;
			cloud->height = ballNum;
			cloud->is_dense = false;
			cloud->points.resize(cloud->width * cloud->height);
			for (size_t i = 0; i < ballNum; i++)
			{
				cloud->points[i].x = pos_host[3 * i];
				cloud->points[i].y = pos_host[3 * i + 1];
				cloud->points[i].z = pos_host[3 * i + 2];
				cloud->points[i].r = rgb_host[3 * i];
				cloud->points[i].g = rgb_host[3 * i + 1];
				cloud->points[i].b = rgb_host[3 * i + 2];
			}
			cloud_viewer_.removeAllPointClouds();
			cloud_viewer_.addPointCloud<pcl::PointXYZRGB>(cloud);
			cloud_viewer_.spinOnce(100);
			pcl::visualization::Camera camPos;
			cloud_viewer_.getCameraParameters(camPos);
#endif
		}
	}
	
	LOG(INFO) << 123;
	return 0;
}
