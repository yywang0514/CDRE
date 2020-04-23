# CDRE-JD
environment： python3, pytorch1.1.0, JDK 

If you want to train the model：

1. Please create two empty folders in the root directory： ./dtree ./dsent

2. Please download the StanfordNLP toolkit package and put it in the root directory
    https://pan.baidu.com/s/1j4LzBPCUuFcPuXrtFPFOZw  password: avn4
   
3. Please download the BERT model and vocabulary and put them in ./models/

4. Run : python train.py proto 5 5 0

If you want to test the model：

1. After finishing the above 1-3, please download our trained model and put it in ./checkpoint
   https://pan.baidu.com/s/1pW7Dqpd2zKeAiIx0ye0CmQ  password: eb3r
   
2. Run python test.py proto 5 5 0

You can edit the training file and test file to change the data set used.
