# Xgboost for Ground Object Classification

## Installation


###1. conda / minmiconda 
在命令行输入以下命令：

	conda create -n xgb python=3.6

	source activate xgb

	conda install numpy, scipy, scikit-learn, scikit-image, matplotlib

	conda install -c conda-forge gdal
	
xgboost 安装

	git clone --recursive https://github.com/dmlc/xgboost
	cd xgboost
	make -j4
	cd ../python-package
	python setup.py install	
	










