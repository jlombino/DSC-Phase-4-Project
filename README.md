<h1>Image Classification with Deep Learning Project</h1>

Author: Jason Lombino

For more information on this project please see my <a href= ./slides.pdf>Presentation</a> or <a href= ./Final_Notebook.ipynb>Jupyter Notebook</a>.


<h1>Business Application</h1>

The Bentley Health Foundation is a non-profit organization with the goal of preventing infectious diseases. Childhood pneumonia is currently one of the main focuses of the Bentley Health Foundation because it kills more children than any other infectious disease worldwide according to <a href=https://data.unicef.org/topic/child-health/pneumonia/>UNICEF</a>. Identifying cases of pneumonia quickly and accurately is a vital step in preventing deaths.

While chest x-rays are simple to take and commonly used to diagnose cases of pneumonia, they need to be interpreted by specialists. Unfortunately, these specialists are frequently not available in rural areas and developing countries. In order to combat this issue, the Bentley Health Foundation is exploring using machine learning to detect signs of pneumonia in chest x-ray images. If successful, the Bentley Health Foundation's model can assist in diagnosing cases of pneumonia so patients can recieve the life-saving treatment they need without having to risk waiting for or traveling to a specialist.

Provided with a patient's chest x-ray image, the model should return the probability that the patient has pneumonia.

<h1>Data</h1>

The following <a href=https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>dataset</a> was provided for modeling. It contains thousands of chest x-rays from children with and without pneumonia. The data provided is separated into train, test, and validation folders. However, the provided validation dataset was too small to effectively use. Prior to performing any modeling <b>I manually moved 20% of the training data into the validation folder</b>. Instructions for doing this can be found in my <a href= ./Final_Notebook.ipynb>Jupyter Notebook</a>. The remaining images in the training folder were used to train models, and the test folder was used as provided.

<h2>Class Imbalance</h2>
One thing that immediately stands out is the class imbalance in the data set. There is approximately a 3:1 class imbalance in the train and validation datasets, and approximately a 2:1 class imbalance in the test dataset. While this is not optimal, It should not skew the evaluation metrics too much.

<img src=images/class_imb.png>

<h2>X-Ray Examples</h2>

Here are some examples of x-ray images used in this analysis. The top three images are healthy x-rays without pneumonia present. The middle three images are x-rays with pneumonia present. The bottom three images are examples of the preprocessing transformations for the training dataset. These transformations are zooming in or out up to 30%, rotation up to 45 degrees in either direction, and horizontal flipping.

<img src=images/example_xray.png>

<h1>Iterative Modeling</h1>

Despite the class imbalance in both the training and test data, <b>Accuracy</b> was the primary metric used to evaluate models. The model should make as many correct decisions as possible. While false negatives are more severe than false positives in this case, favoring recall too heavily will likely result in many patients undergoing treatment for a disease they do not have wasting valuable resources.

I made several models while trying to determine the best convolutional neural network (CNN) architecture and hyperparameters for classifying the x-rays. The training and validation results for each of the models can be seen in the table below. Each model has a corresponding jupyter notebook located <a href=notebooks/>here</a>.

<h3><a href=notebooks/01_basic_model.ipynb>01</a> Basic CNN</h3>
This is a very basic model that used a small neural network and did not preprocess images. This was used as a starting point for the rest of my models, but was not considered when selecting a final model.

<h3><a href=notebooks/02_image_preprocessing.ipynb>02</a> Basic CNN Image Preprocessing</h3>
This model attempts to improve on the basic CNN by adding image preprocessing to reduce overfitting. 

<h3><a href=notebooks/03_larger_network.ipynb>03</a> Larger CNN</h3>
This model attempts to improve on on the basic CNN by increasing the size of the neural network. Both the number of layers and the size of each layer is increased. 

<h3><a href=notebooks/04_tanh_activation.ipynb>04</a> Larger CNN tanh Activation</h3>
This model attempts to improve on the basic CNN by using a tanh activation function for each layer rather than relu. It performs much worse overall in testing and on validation data.

<h3><a href=notebooks/05_dropout.ipynb>05</a> Larger CNN Dropout</h3>
This model attempts to improve on the basic CNN by adding dropout layers to reduce overfitting.

<h3><a href=notebooks/06_larger_stride.ipynb>06</a> Larger CNN Larger Stride</h3>
This model attempts to improve on the basic CNN by increasing the stride of the filter in each convolutional layer to reduce overfitting.

<h3><a href=notebooks/07_double_conv.ipynb>07</a> Larger CNN Double Convolutional Layers</h3>
This model attempts to improve on the basic CNN by adding additional convolutional layers.

<h3><a href=notebooks/08_even_larger_network.ipynb>08</a> Even Larger CNN</h3>
This model attempts to improve on the basic CNN by adding additional convolutional and dense layers. Both the number of layers and the size of each layer is increased. 

<h3><a href=notebooks/09_transfer_xc.ipynb>09</a> Transfer Learning Xception</h3>
This model attempts to imrpove on the basic CNN by using the <a href=https://keras.io/api/applications/xception/>Xception</a> network as its convolutional base.

<h3><a href=notebooks/10_transfer_vgg.ipynb>10</a> Transfer Learning VGG16</h3>
This model attempts to imrpove on the basic CNN by using the <a href=https://keras.io/api/applications/vgg/>VGG16</a> network as its convolutional base.

<h3><a href=notebooks/11_vgg_decay.ipynb>11</a> Transfer Learning VGG16 Decaying Learning Rate</h3>
This model attempts to imrpove on the basic CNN by using the <a href=https://keras.io/api/applications/vgg/>VGG16</a> network as its convolutional base. It also decreases the learning rate each epoch.

<h1>Comparison of Models</h1>

This table shows the performance of each of the models listed above on the train and validation data. I explore models 8 and 11 further in my <a href= ./Final_Notebook.ipynb>Jupyter Notebook</a>, but only show the results for model 11 here. Model 11 was selected over model 10 because its results are more stable.

| Model Name                                        | % Train Accuracy | % Val Accuracy | 
|---------------------------------------------------|------------------|----------------|
| 01 (Not Considered) Basic CNN                     | 98.8             | 96.7           | 
| 02 Basic CNN Image Preprocessing                  | 93.5             | 91.2           | 
| 03 Larger CNN                                     | 93.5             | 94.0           | 
| 04 Larger CNN tanh Activation                     | 74.3             | 74.1           | 
| 05 Larger CNN Dropout                             | 93.3             | 94.0           | 
| 06 Larger CNN Larger Stride                       | 95.3             | 93.4           | 
| 07 Larger CNN Double Convolutional Layers         | 95.5             | 94.0           | 
| 08 Even Larger CNN                                | 94.8             | 94.5           | 
| 09 Transfer Learning Xception                     | 96.6             | 94.0           | 
| 10 Transfer Learning VGG16                        | 95.0             | 95.7           | 
| 11 Transfer Learning VGG16 Decaying Learning Rate | 94.7             | 94.2           |

<h1>Best Model - 11 VGG16 Decaying LR</h1>

 My best performing model was model <a href=notebooks/11_vgg_decay.ipynb>11</a>. The structure of this model is shown below. It uses VGG16 as its convolutional base with several densely connected layers to classify the images.

<img src=images/tx_model.png>

<h2>Model Results</h2>

The model achieves <b>90.4% Accuracy</b> on the test dataset. The model appears to be better at classifying positive cases than negative cases, but this makes sense because the training data had an abundance of positive cases. Despite this, the model achieves a 86% precision and 97% recall on the test set.

<img src=images/tx_confusion.png>

<h1><a href=https://github.com/marcotcr/lime>Lime</a> Explanation of Results</h1>

I used Lime to explain what the model is "seeing" when it makes a classification. I looked at explanations for images the model classified correctly and incorrectly. In both cases, the model does seem to focus near the lungs, exactly where the x-ray images would show signs of pneumonia. There is some interesting behavior in some of the predictions though where the model appears to be focusing outside the body. This is unexpected and concerning, but one possible explanation is that one's body weight has some impact on susceptibility to pneumonia.

<img src=images/tx_tp.png>

<img src=images/tx_fp.png>

<h1>Intermediate Layer Activations</h1>

It is also interesting to know what each convolutional layer in the network "sees". The following image shows the activation of the first neuron in each convolutional layer in the neural network. It becomes clear that each layer focuses on an increasingly specific pattern within the image. 

<img src=images/layers.png>

<h1>Conclusion</h1>

The final model can be used by the Bentley Health Foundation to determine whether any given chest x-ray shows signs of pneumonia. This model can be deployed in regions where pneumonia is common to assist with diagnosing cases. This will allow patients to recieve the life-saving treatment they need without having to risk waiting for or traveling to a specialist.

 <h1>Repository Information</h1>

```
├── notebooks                        <- Jupyter notebooks for various models
├── images                           <- Graphs generated from code
├── Final_Notebook.ipynb             <- Jupyter notebook with my full analysis
├── Final_Notebook.pdf               <- PDF version of project Jupyter notebook
├── slides.pdf                       <- PDF version of project presentation
└── README.md                        <- The top-level README you are currently reading
```