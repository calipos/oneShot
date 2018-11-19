#include "caffe/util/db.hpp"


#include <string>

namespace caffe { namespace db {

DB* GetDB(DataParameter::DB backend) {
  switch (backend) {


  default:
    LOG(FATAL) << "Unknown database backend";
    return NULL;
  }
}

DB* GetDB(const string& backend) {


  LOG(FATAL) << "Unknown database backend";
  return NULL;
}

}  // namespace db
}  // namespace caffe
