pycharm remote connect AWS ipython <br>
need to configure CUDA_Path,  LD_LIBRARY_PATH   oetherwise, it will prompt up "ImportError: libmklml_intel.so"
open pycharm-settings-build,execution,deployment- python console- add ENV(dont do it in Run-configuration, that might work with python, but not with ipython)
