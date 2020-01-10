# Intelligent-Traffic-Control-System
<b>Aim :</b> Enables passage of emergency vehicles in Traffic congestion scenario by detecting emergency siren sounds

# Abstract : 
INDIA is one of the fast growing economies in the world and it is second most populous Country. In India traffic congestion problem happens frequently. Day by day number of vehicles are increasing exponentially but to cope up with this exponential growth India does not have any traffic control mechanism. Also, the traffic in Indian is disordered and it is non lane based. For solving such problems it required very accurate traffic control system.

Homo sapiens are evolving as time moves on, making our life simpler is a part and parcel of the term evolution, satisfaction is a behavior which every individual wants, this behavior is only  exhibited  when  our  needs  are  fulfilled,  to  fulfill  these  requirements  each  and  every organization in modern world extract data from every individual and analyze the requirements on a large scale, these data gathering requires some structured and well defined process like survey, conferences and many other, by concatenating all data that are collected from various resources the firm will come to a conclusion in which the outcome exactly gives what society wants based on this result the organizations start building the solution.

In India due to high population density there are lot of challenges that needs to be addressed, in that one of major issue is “Traffic”, due to traffic congestion many problems arises the major problems are global warming, ozone layer depletion in global level, but it is also determined that due to the toxic gases that are liberated into the open environment humans are suffering from various diseases, all these occurrences will impact the standard of living which is a bad sign for overall growth of a society or nation, one of the main reasons for the traffic congestion for emergency vehicles is haphazard way of parking the vehicle in our society this has been proved by the survey that was conducted by various resources.

The project aims to develop an Intelligent Traffic Control System by recognizing siren sound of emergency vehicles. This consists of a ML service running in Raspberry Pi. The sensors installed in the Traffic Poles will pick up the siren sound and sends it to the AI system for recognition, which in turn clears the respective signal. Poles with sound sensors will be attached at farther distances from the traffic pole. The system will be trained to recognize and provide solutions for ambulance siren sounds in different traffic situations.

# Understanding files and folders in the project : 

There are a few files and folders present in the project which are necessary for working of the project or used in intermediate steps during building of the project. Understanding the purpose of each files and contents of folders will help in easy execution of the program.

<ul>
  <li>MP_model.tflite - Saved Neural Network for Embedded and Mobile Devices
  <li>MP_model.h5 - Neural Network is saved in this file. This is a Keras-API generated file.</li>
  <li>ml.py - Contains the Keras code for the program. </li>
  <li>spectrogram.py - Can generate spectrograms from a wave file. Needs to be edited by changing the wave file name and output file name
  <li>Spectrograms.zip - contains all the spectrogram images</li>
  </ul>
  
# Setting things up : 

<ul>
  Install the following libraries to get the system to working.<br /><br />
  <li>Tensorflow</li>
  <li>Numpy</li>
  <li>OpenCV (Version 2 used in project)</li>
  <li>Matplotlib</li>
  <li>Python (Version 3 used in project)</li>
</ul>
 <p>From the zip file named Spectrograms.zip extract all the sounds to the directory where the file ml.py exists.<br />
  
 # Running the program :  
  Run the ml.py file using Python Interpreter installed.
  Type <b> <i>py ml.py</i> </b> (tested on Windows 10).
The program will start execution by performing the training,testing and then predictions with pre-recorded sounds.
</p>
