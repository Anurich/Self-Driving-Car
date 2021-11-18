<h2> Self-Driving-Car </h2>
<p>In this work, I try to implement simple CNN model to predict the steering angle of car given image from front camera i.e. center. 
It's an end to end learning approach from nvidia published in 2016 paper on End to End Learning for Self Driving Car
Link: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf. 
</p>
<h2> Architecture Information. </h2>
<p>
In this project I used my own architecture, I used pretrained network vgg16 trained on imagenet followed by some dense layer with elu activation function.
One can try to replicate the same architecture presented in Nvidia paper, Or can use different architecture.
</p>
<h2>Dataset Information. </h2>
<p>
A lot of self driving car dataset is available thanks to <b> Comma.ai, Udacity and many other companies </b>, But in my case due to limited resources I cannot train my model on a lot of data, So i keep it simple I trained my model on 44k images and use somewhere around 14k for validation and 4k for testing. 
The dataset I use:
<ul>
  <li>Thanks to SullyChen for making data available: https://github.com/SullyChen/driving-datasets</li>
</ul>
Other datset:
<ul>
  <li>This link include dataset from Udacity : https://github.com/udacity/self-driving-car/tree/master/datasets </li>
</ul>
</p>
<h2> Installation. </h2>
<ul>
  <li> Python==3.6.6</li>
  <li> Pytorch==1.6.0</li>
</ul>
<h2> Training Infomation & Preprocessing. </h2>
<p>
Model is trained for only 1 epoch due to limited resources, I normalize the data as well as try different approaches like calibrating a Steering Angle , Performing data Argumentation techniques. The code contain only two file one is dataset.py which contain the data loader class another is model.py file which contain model and main loop.
<h2> Prediction vs Groundtruth (Steering Angle) </h2>
Result may not be impressive because model not trained for large epoch and data it is trained on is very limited.</p>
<figure>
<img src ="Figure_1.png" heigh="300" width="500"/>
  </figure>

<h2> Alert & Contribute. </h2>
<p>
The above project is only for research purposes and using this in physical device can be dangerous. Please feel free to point any mistake in code as well any suggestion for improvement, feel free to contribute.
</p>
