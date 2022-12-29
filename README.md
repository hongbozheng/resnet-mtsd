# ResNet-50 (TensorFlow)

## Project Setup
#### 1. Package
This project depends on the installation of the following essential packages:
```
sudo apt-get install python3-venv
```
#### 2. Virtual Environment
Create a `Python` virtual environment
```
python3 -m venv ./venv
```
Activating the virtual environment
```
source venv/bin/activate
```
Leaving the virtual environment
```
deactivate
```
#### 3. Install dependencies
In the activated virtual environment
```
cd script
./setup.bash
```
Test `tensorflow`
```
tf.test.is_gpu_available()
```
```
tf.config.list_physical_devices('GPU')
```
```
print("# GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
If below info appeared, check the [NUMA node problem](https://gist.github.com/zrruziev/b93e1292bf2ee39284f834ec7397ee9f)
```
2022-12-26 21:02:28.791057: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937]
successful NUMA node read from SysFS had negative value (-1),
but there must be at least one NUMA node, so returning NUMA node zero
```
If below warnings appeared, check [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) &
[NVIDIA TENSORRT DOCUMENTATION](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#install)

**Note: NVIDIA TensorRT is `OPTIONAL`**
```
2022-12-26 21:31:58.505309: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64]
Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory;
LD_LIBRARY_PATH: /usr/local/cuda/lib64:/usr/local/cuda-11.8/lib64
2022-12-26 21:31:58.505374: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64]
Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory;
LD_LIBRARY_PATH: /usr/local/cuda/lib64:/usr/local/cuda-11.8/lib64
2022-12-26 21:31:58.505383: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38]
TF-TRT Warning: Cannot dlopen some TensorRT libraries.
If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
```
Check tensorflow compatible version at [Tested build configurations](https://www.tensorflow.org/install/source#gpu)

#### CUDA helpful Links
* [cuDNN Download](https://developer.nvidia.com/rdp/cudnn-download)
* [NVIDIA CUDNN DOCUMENTATION - Tar File Installation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar)
* [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
* Add the following lines into `~/.bashrc`
```
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
* [Install CUDA on Ubuntu 20.04 Focal Fossa Linux](https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux)
* [Could not load dynamic library `libcudart.so.*`](https://stackoverflow.com/questions/64193633/could-not-load-dynamic-library-libcublas-so-10-dlerror-libcublas-so-10-cann)