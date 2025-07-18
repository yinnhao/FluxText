export CC=gcc-9
export CXX=g++-9
git clone https://github.com/NVIDIA/apex
cd apex
# 清理之前可能的缓存
python setup.py clean
python setup.py install --cpp_ext --cuda_ext