pycharm remote connect AWS ipython <br>
need to configure CUDA_Path,  LD_LIBRARY_PATH   oetherwise, it will prompt up "ImportError: libmklml_intel.so"
open pycharm-settings-build,execution,deployment- python console- add ENV(dont do it in Run-configuration, that might work with python, but not with ipython)


Configure Pycharm with Ipython
https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d
1. configure python interpreter(deployment or SSH with key pair)
2. set ENV in python console, add LD_LIBRARY
3. configure automatic upload
