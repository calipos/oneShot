#include "caffe/util/hdf5.hpp"

#include <string>
#include <vector>

namespace caffe {

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob) 
{

}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) 
{

}



template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string& dataset_name, const Blob<float>& blob,
    bool write_diff) 
{
}



string hdf5_load_string(hid_t loc_id, const string& dataset_name) 
{
	return "";
}

void hdf5_save_string(hid_t loc_id, const string& dataset_name,
                      const string& s) 
{
}

int hdf5_load_int(hid_t loc_id, const string& dataset_name) 
{

	return 0;
}

void hdf5_save_int(hid_t loc_id, const string& dataset_name, int i) 
{
}

int hdf5_get_num_links(hid_t loc_id) 
{ 
	return 0;
}

string hdf5_get_name_by_idx(hid_t loc_id, int idx) 
{
	return "";
}

}  // namespace caffe
