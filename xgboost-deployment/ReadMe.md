# Xgboost

## 安装前配置

###1. conda / minmiconda 环境配置
在命令行输入以下命令：

	conda create -n xgb python=3
	source activate xgb

	conda install numpy scipy scikit-learn scikit-image matplotlib tifffile
	conda install -c conda-forge gdal
	
	
###2. Ubuntu下直接配置
	pip install numpy scipy scikit-learn scikit-image tifffile matplotlib
	
	apt-get install libgdal-dev
	export CPLUS_INCLUDE_PATH=/usr/include/gdal/
	export C_INCLUDE_PATH=/usr/include/gdal/
	pip install GDAL==1.11.2
	
	
##xgboost 安装

	git clone --recursive https://github.com/dmlc/xgboost
	cd xgboost

###ubuntu for cpu: 

	make -j4
	cd ./python-package
	python setup.py install	
	export PYTHONPATH=~/xgboost/python-package:$PYTHONPATH
	
###ubuntu for gpu: 

	mkdir build
	cd build
	cmake .. -DUSE_CUDA=ON
	make -j4
	cd ../python-package
	python setup.py install
	export PYTHONPATH=~/xgboost/python-package:$PYTHONPATH
 
###mac
 
	brew install gcc
	cp make/config.mk ./config.mk
	gcc --version
	
	Open config.mk and uncomment these two lines：
	export CC = gcc
	export CXX = g++
	然后修改成安装的gcc版本，如：
	export CC = gcc-7
	export CXX = g++-7
	make -j4
	
	cd ./python-package
	python setup.py install	
	export PYTHONPATH=~/xgboost/python-package:$PYTHONPATH

###windows for cpu
	
	

	
	
	

	










