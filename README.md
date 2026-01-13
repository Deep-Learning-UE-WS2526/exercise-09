# Exercise 9: LSTM

Deadline: 21.01.26 23:59

This is the ninth exercise, and it is about long short-term memory.

## Step 1: Git-Stuff
Clone this repository to your local computer. On the command line, you would use the following command: `git clone https://github.com/Deep-Learning-UE-WS2526/exercise-09`. Create a new branch in the repository, named after your UzK-account: `git checkout -b "UZKACCOUNT"`.

## Step 2: Setup

The file `exercise.py` contains many of the things you need for the exercise: The entire preprocessing, input conversion and a basic neural network. Make sure you understand what's going on there and why -- preparing the input for a neural network is the thing you'll spend most of the time on in real life!

## Step 3: Model Training

If you run the code as it is, it'll deliver an accuracy of 0.97, which already sounds pretty good. However, it's actually very close to the baseline, because we have a huge imbalance in the data. Improve the performance of the network! A little tip: The function 'fit()', which is applied to the model, has a parameter 'sample_weight'. Use this parameter to balance the unbalanced dataset with additional weights.

## Step 4: Commit
Commit your changes to your local repository and push them to the server.