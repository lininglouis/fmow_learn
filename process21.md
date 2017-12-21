pycharm remote connect AWS ipython <br>
need to configure CUDA_Path,  LD_LIBRARY_PATH   oetherwise, it will prompt up "ImportError: libmklml_intel.so"
open pycharm-settings-build,execution,deployment- python console- add ENV(dont do it in Run-configuration, that might work with python, but not with ipython)


Configure Pycharm with Ipython
https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d
1. configure python interpreter(deployment or SSH with key pair)
2. set ENV in python console, add LD_LIBRARY
3. configure automatic upload



pytorch transform normalize ()  <br>
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
(ref  https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683 )                                     


pytorch crossEntroyLoss(input, target) <br>
In a word: If you do multiple classifcation, you dont need to add a softmax or logsoftmax layer at the end of the network. Pytorch put this part of work to loss function crossEntroyLoss.
for example a 10-class classification, batch_size 16.
The input will be a matrix of [10, 16]
the target will be a vector with only labels(no need to one hot encode) [16]
This **crossEntroyLoss** criterion combines LogSoftMax and NLLLoss in one single class.
