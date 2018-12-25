#ifndef _REALSENSE_H_
#define _REALSENSE_H_
#include <map>
#include <atomic>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#include"sensor.h"

/**
Class to encapsulate a filter alongside its options
*/
class filter_options
{
public:
	filter_options(const std::string name, rs2::processing_block& filter);
	filter_options(filter_options&& other);
	std::string filter_name;                                   //Friendly name of the filter
	rs2::processing_block& filter;                            //The filter in use
	//std::map<rs2_option, filter_slider_ui> supported_options;  //maps from an option supported by the filter, to the corresponding slider
	//std::atomic_bool is_enabled;                               //A boolean controlled by the user that determines whether to apply the filter or not
};


#endif // !_REALSENSE_H_

