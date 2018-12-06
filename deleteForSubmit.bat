cd %cd%
rd /q /s oneShot\3rdPart\Boost\include
del oneShot\3rdPart\Boost\lib\*.lib
rd /q /s oneShot\3rdPart\Eigen\eigen3
rd /q /s oneShot\3rdPart\pcl\include
rd /q /s oneShot\3rdPart\VTK\include
del oneShot\3rdPart\VTK\lib\*.lib
del oneShot\3rdPart\openni2\OpenNI2.dll
del oneShot\3rdPart\caffe\dependence\protobuf_v140\lib\Release\libprotobuf.lib
del oneShot\3rdPart\caffe\dependence\protobuf_v140\lib\Debug\libprotobufd.lib

pause