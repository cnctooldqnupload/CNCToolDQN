# CNCToolDQN

CNCToolDQN implementation and TH score estimation (First uploaded in April/12/2023)


 ## CNCToolDQN Architecture
![architecture](https://user-images.githubusercontent.com/131362675/233518536-bbf521c0-8ec4-49fc-be5f-8581c7dafccb.png)

CNCToolDQN is a Deep Q-learning based framework for tool monitoring, in which the agent is trained by exploring the representation space of multivariate time-series data (state) of tool loads (i.e., Spindle loads, and Servo loads) and tool age, and gets rewards when it correctly prdicts anomalies (action).  
Then, predicted output is used to estimate TH (Tool health) score ranging from 0 to 100. 

## Dataset 

For reproduction, we provide a portion of preprocessed data. 

The sample data (OP20_2A End mill) is downloadable at https://drive.google.com/file/d/1qCz1cYYKYRI1uqwHBFPBxfq4bJeEPHNj/view?usp=share_link.

 ## How to run

How to run:

python main.py --dir='C:/' --dir_savefigure= 'C:/' --gpu='0' --maxlen=300 --L= 1 --alpha=0.005 --tau=1

--dir: Data directory. ex) "C:/Users/Desktop/Data/"

--dir_savefigure : The directory to save TH score plot. ex) "C:/Users/Desktop/figure/"

--gpu: GPU index to use (default: '0')

--maxlen: Window size (default = 300) 

--L: Lower limit to calculate p_l, which is the probability of Z score lower than -L (default = 1)

--alpha: An hyperparameter to estimate TH score (default = 0.005)

--tau: An hyperparameter to estimate TH score (default = 0.1)
