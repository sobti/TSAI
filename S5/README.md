
**My Targets are** -<br>

99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)<br>
Less than or equal to 15 Epochs<br>
Less than 10000 Parameters<br>
Do this in minimum 5 steps<br>
Each File must have "target, result, analysis" TEXT block (either at the start or the end)<br>
You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct.<br> 
Explain your 5 steps using these target, results, and analysis with links to your GitHub files<br>
Keep Receptive field calculations handy for each of your models.<br> 

----------------------------------------------------------------------------------------
**Approach 1** <br> 

No Image Augmentation, No LR Decay, No dropout, No batch normalization.<br> 
Limit my parameters in 11k. I want to test my accuracy and need to know what my model is capable of doing. If it overfits, then i reduce my parameter further otherwise if it is underfit, then try to increase the parameters.<br>

Used parameters -10,984<br>
Human's based performance -99.7<br>
Maximum Train Accuracy Achieved -97.77<br>
Maximum Test Accuracy Achieved - 97.89<br>
Maximum - Epoch -20<br>

**Observations** - *The accuracy can be increased further. The accuracy has incremental trend. The model is underfit by looking at the human level accuracy. So i will try to increase the parameters. In the Initial training phase of the model, the loss is not decreasing as much.*

----------------------------------------------------------------------------------------
**Approach 2**<br>

*Problem* - Improve the Initial training phase of model by using only Batch Normalization. It will increase the parameters in my model up to >11k.<br> 

Used parameters -11,192<br>
Human's based performance -99.7<br>
Maximum Train Accuracy Achieved -99.72<br>
Maximum Test Accuracy Achieved - 99.14<br>
Maximum - Epoch -20<br>

**Observations** - *Wow, by introducing only batch normalization, my model's initial phase accuracy jump to 87% (in first epoch. However in previous approach it is 9.5%). By lookin at the log's. reach to conclusion my model is overfitting. The difference between my training accuracy and test accuracy is very high. By looking at logs the Learning rate may be big, need to reduce after epoch, the training accuracy decreases and then increases.*

------------------------------------------------------------------------------------------------
**Approach 3** <br>

*Problem* - Remove the overfitting.<br>
*Solutions* - The overfitting problem can be resolved by training the smaller parameter network. By using dropout we can reduce the overfitting. By increasing the data by (data augmentation, resize, crop, color jitter) images to train. <br>

*By dropout* - <br>
Starting from the values of 0.1. The overfitting problem is reduced. It also underfit the model. So tunning the parameter hold between 0.01 to 0.03. It will not help to reduce the overfitting. When we facing the problem in second decimal.<br>

*By - reducing parameter*<br>
I reduced the parameter be in 10 k. The overfit problem is reduced little bit. This approach is effective in terms of dropout.<br>

*By - data augmentation (Rotate)*<br>
With the help of rotate the overfit is reduced.<br>

*Adopted Method - Reducing parameter--> Data Augmentation*<br>

Used parameters - 9,678<br>
Human's based performance -99.7<br>
Maximum Train Accuracy Achieved -99.2<br>
Maximum Test Accuracy Achieved - 99.44<br>
Maximum - Epoch -15<br>

**Observations** - *Model is slightly overfitting the data in epoch number 9,10. The overfitting is for .20-.30. By looking at the graph's there are too much flucations in the test error and test accuracy.*<br>

---------------------------------------------------------------------------------------------------------------------------------
**Approach 4** - <br>

*Problem* - Reduce overfitting and flucations in the test error/test accuracy<br>
*Solution* - Use the different learning rate and slight data augmentation change parameter.<br>

Used parameters - 9,678<br>
Human's based performance -99.7<br>
Maximum Train Accuracy Achieved -99.2<br>
Maximum Test Accuracy Achieved - 99.44<br>
Maximum - Epoch -20<br>
Consistency in result - 4 epoch<br>

**Observtion** - *Fluctations reduced by the learning rate decay. From epoch 16 to 20, the consistency is greater than 99.4. According to the target, the results are lagging little bit. Trying to reach the target in next approach.*<br>

----------------------------------------------------------------------------------------------------------------------
**Approach 5** - <br>

*Problem* - Reach the 99.4 consistent in 15 epoch.<br> 
*Solution* - Reducing the Batch size so that it can learn more faster. To buzz my accuracy greater than 99.4, i also adjust the learning rate, probably increase the learn rate. I will reduce the step size of step lr because my model will learn more faster due to reduced batch size.<br>

Used parameters - 9,678<br>
Human's based performance -99.7<br>
Maximum Train Accuracy Achieved -99.17<br>
Maximum Test Accuracy Achieved - 99.47<br>
Maximum - Epoch -15<br>
Consistency in result - Last 9 epoch<br>

**Observation** - *Targets are achieved.*<br>

------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------
