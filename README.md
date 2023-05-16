# Recurrent Neural Networks for Predictive Maintenance
* Author: Umberto Griffo
* Twitter: @UmbertoGriffo

## Colab
You can try the code directly on [Colab](https://colab.research.google.com/drive/1tjIOud2Cc6smmvZsbl-QDBA6TLA2iEtd).
Save a copy in your drive and enjoy It!

## Software Environment
* Python 3.6
* numpy 1.13.3
* scipy 0.19.1
* matplotlib 2.0.2
* spyder 3.2.3
* scikit-learn 0.19.0
* h5py 2.7.0 
* Pillow 4.2.1 
* pandas 0.20.3
* TensorFlow 1.3.0
* [Keras 2.1.1](https://keras.io)

## Problem Description
In this example I build an LSTM network in order to predict remaining useful life (or time to failure) of aircraft engines <a href="https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan">[3]</a> based on scenario described at <a href="https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb">[1]</a> and <a href="https://gallery.azure.ai/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2">[2]</a>.
The network uses simulated aircraft sensor values to predict when an aircraft engine will fail in the future, so that maintenance can be planned in advance.
The question to ask is "Given these aircraft engine operation and failure events history, can we predict when an in-service engine will fail?"
We re-formulate this question into two closely relevant questions and answer them using two different types of machine learning models:

	* Regression models: How many more cycles an in-service engine will last before it fails?
	* Binary classification: Is this engine going to fail within w1 cycles?

## Data Summary
In the **Dataset** directory there are the training, test and ground truth datasets.
The training data consists of **multiple multivariate time series** with "cycle" as the time unit, together with 21 sensor readings for each cycle.
Each time series can be assumed as being generated from a different engine of the same type.
The testing data has the same data schema as the training data.
The only difference is that the data does not indicate when the failure occurs.
Finally, the ground truth data provides the number of remaining working cycles for the engines in the testing data.
The following picture shows a sample of the data: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/datasetSample.png"/>
</p>
You can find more details about the data at <a href="https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb">[1]</a> and <a href="https://gallery.azure.ai/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2">[2]</a>.

## Experimental Results
### Results of Regression model

|Mean Absolute Error|Coefficient of Determination (R^2)|
|----|----|
|12|0.7965|

The following pictures show the trend of loss Function, Mean Absolute Error, R^2 and actual data compared to predicted data: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_regression_loss.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_mae.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_r2.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_regression_verify.png"/>
</p>
         
### Results of Binary classification 

|Accuracy|Precision|Recall|F-Score|
|----|----|----|----|
|0.97|0.92|1.0|0.96|

The following pictures show trend of loss Function, Accuracy and actual data compared to predicted data: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_loss.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_accuracy.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_verify.png?raw=true"/>
</p>

## Extensions
We can also create a model to determine if the failure will occur in different time windows, for example, fails in the window (1,w0) or fails in the window (w0+1, w1) days, and so on. This will then be a multi-classification problem, and data will need to be preprocessed accordingly. 

## Who is citing this work?

* In chapter 10 of [Hands-On Artificial Intelligence for IoT](https://www.amazon.it/Hands-Artificial-Intelligence-IoT-techniques/dp/1788836065) book
	* https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-IoT/tree/master/Chapter10
* In chapter 7 of [Mobile and Wireless Communications with Practical Use-Case Scenarios](https://www.amazon.com/Wireless-Communications-Practical-Use-Case-Scenarios/dp/1032119020) book
	* https://www.google.pt/books/edition/Mobile_and_Wireless_Communications_with/lvqhEAAAQBAJ?hl=en&gbpv=1
* In "Using Recurrent Neural Networks to predict the time for an event" master's thesis (Universitat de Barcelona, Barcelona, Spain). Retrieved from http://diposit.ub.edu/dspace/bitstream/2445/134691/3/memoria.pdf

## References

- [1] Deep Learning for Predictive Maintenance https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
- [2] Predictive Maintenance: Step 2A of 3, train and evaluate regression models https://gallery.azure.ai/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2
- [3] A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan), NASA Ames Research Center, Moffett Field, CA 
- [4] Understanding LSTM Networks http://colah.github.io/posts/2015-08-Understanding-LSTMs/
