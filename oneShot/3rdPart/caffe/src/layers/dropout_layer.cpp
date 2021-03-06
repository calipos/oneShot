// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		NeuronLayer<Dtype>::LayerSetUp(bottom, top);
		threshold_ = this->layer_param_.dropout_param().dropout_ratio();
		DCHECK(threshold_ > 0.);
		DCHECK(threshold_ < 1.);
		scale_ = 1. / (1. - threshold_);
		uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
		scale_train_ = this->layer_param_.dropout_param().scale_train();
	}

	template <typename Dtype>
	void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		NeuronLayer<Dtype>::Reshape(bottom, top);
		// Set up the cache for random number generation
		// ReshapeLike does not work because rand_vec_ is of Dtype uint
		rand_vec_.Reshape(bottom[0]->shape());
	}

	template <typename Dtype>
	void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		unsigned int* mask = rand_vec_.mutable_cpu_data();
		const int count = bottom[0]->count();
		if (this->phase_ == TRAIN) {
			// Create random numbers
			LOG(FATAL) << "TRAIN FORWARD NOT IMPLEMENT";
		}
		else {
			caffe_copy(bottom[0]->count(), bottom_data, top_data);
			if (!scale_train_) {
				caffe_scal<Dtype>(count, 1. / scale_, top_data);
			}
		}
	}



#ifdef CPU_ONLY
	STUB_GPU(DropoutLayer);
#endif

	INSTANTIATE_CLASS(DropoutLayer);
	REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
