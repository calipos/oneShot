#include "realSense_post.h"
filter_options::filter_options(filter_options&& other) :
	filter_name(std::move(other.filter_name)),
	filter(other.filter)
	//,is_enabled(other.is_enabled.load())
{
}

/**
Constructor for filter_options, takes a name and a filter.
*/
filter_options::filter_options(const std::string name, rs2::processing_block& filter) :
	filter_name(name),
	filter(filter)
	//,is_enabled(true)
{
	const std::array<rs2_option, 3> possible_filter_options = {
		RS2_OPTION_FILTER_MAGNITUDE,
		RS2_OPTION_FILTER_SMOOTH_ALPHA,
		RS2_OPTION_FILTER_SMOOTH_DELTA
	};

	//Go over each filter option and create a slider for it
	for (rs2_option opt : possible_filter_options)
	{
		if (filter.supports(opt))
		{
			rs2::option_range range = filter.get_option_range(opt);
			std::string opt_name = rs2_option_to_string(opt);
			std::string prefix = "Filter ";
		}
	}
}