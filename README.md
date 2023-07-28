# Multitask Model to Forecast Patient's Next Unit and Remaining Length of Stay

This is the source code for building a deep learning model using multi-task learning (MTL) approach. The model is designed to train and validate on data prepared from Electronic Health Records (EHRs).

How long a patient will stay in the hospital or patient's length of stay (LOS) is a necessary piece of information for hospital resource allocation. Having a good estimation of LOS can aid in hospital planning and decision making. Prediction of LOS at the admission time might be distorted because right at the beginning, we don't have adequate information to interpret patient's status. To address this, we have designed our model to learn and predict adaptively along the patient's stay. The model appends new data about the patient as it becomes available and update the prediction value accordingly. 

Also, general hospitals where there are arrangements of multiple units for providing care, patients can get transferred from one unit to another due to a diverse set of underlying situations. Hospital logistics as well as the span of patient's stay might change depending on the unit he/she is moving. Considering the latent relationship between this two variable, we designed our model to predict these two variable simultaneously.

In this multi-task learning model, we train the model to predict:<br>
  - Task-1 --> patient's unit label in next time step
  - Task-2--> patient's remaining length of stay

The model inputs get updated on a daily basis and generates prediction accordingly.

## Running the code

The starting point of the code is main.py. It is specifically desinged to take input from the user based on the EHR data that we are using. The dataset contains six years of health records starting from January 2016 and December 2021. Considering the execution time of train and test, we have divided it for each year resulting in six datasets. Executing the main.py will ask the user whether she wants to train a model or wants to test the model that was previously trained. The user can also do both by selecting a dataset(train and test). Depending on the user input, the code will train model, save the model with its weight vectors and subsequently plot the evaluation metrics.

Since there are two different types of tasks, our evaluation metrics are set accordingly. Prediction of next unit label is a classification task, our evaluation metrics for this task are accuracy and top-k accuracy (k=2). Prediction of remaining length of stay is a regression task, our evaluation metrics in this case is mean absolute error, mean squared error and root mean squared error. The MAE and Top-k accuracy values are plotted and saved in the image folder to show the model performance on test data. The history of the training time is saved in the hist_log folder.


