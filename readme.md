# MNIST / ML: Python (PyTorch)
### Preparation for training
1. First you need to download the dataset from [link](https://github.com/myleott/mnist_png)
2. For training/testing, we need an scv file, where each line specifies the path to the image from 
the training sample and the class label. To convert the downloaded dataset to a csv file, [script](io.py) is written.
We pass the path to the dataset and the csv file into which we want to save the result. Example:</br> 
<code>python ./io.py --dataset_dir ./data/training/ --file_path ./data/train.csv</code>
3. Having received the csv file, we can start training. The hyperparameters for training lie in
[config file](./training_hyperparams.py). Training starts like this:</br>
<code>python mnist.py train --dataset_path ./data/train.csv --save_dir ./model</code>

### Training
After training the model, it is worth testing it. But first of all, we need a csv file of the test dataset. 
We receive it as the second point of preparation for training:</br>
<code>python ./io.py --dataset_dir ./data/testing/ --file_path ./data/test.csv</code></br>
And we can start testing. Testing is started as follows: </br>
<code>python ./mnist.py inference --model_path ./model/model.pth --dataset_path ./data/test.csv --save_dir ./model/result_test</code></br>

### Collection of metrics 
We first need a csv file again, but for the test results:</br>
<code>python ./io.py --dataset_dir ./model/result_test/ --file_path ./model/result_test.csv</code></br>
And now we can start collecting metrics:</br>
<code>python ./evalute.py --dataset_path ./data/test.csv --result_path ./model/result_test.csv</code>