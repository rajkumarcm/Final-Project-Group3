# Instructions for Downloading Data and Running the Code

## 1. Download data set from Kaggle
###     a. Access the Kaggle page for [CNN-Daily Mail News Text Summarization](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail) 
###     b. use the "download" button on the top right of the page to download the data set to your computer and unzip the files.
###     c. Unzip the download, there sould be three cvs files, test - train - validation.
###     d. Place the three cvs files in the desired location on your computer.

<br>

## 2. Adjust data directory location
### a. Open the *NLP_project.py* code file in the ***Code*** directory
### b. On line, 64, change the variable, *DATA_DIR*, to the location of the folder on your computer that contains the three cvs files

<br>

## 4. Train and Test the model
### a. Open the *NLP_project.py* ***Code*** file in the code directory
### b. Run the file

<br>

## 6. Making personal changes to the model
-If you desire to make changes to and experiment with the model, you can find most of the **hyperparamters** from lines *72-80*  
-If you want to speed things up, you can work with a small sample of the data. To do this, go to trainer argument in the build function. Comment out lines **195-196**. Instead, uncomment lines **199-200**. In additional, navigate to the test function. Comment out line **226** and uncomment line **230**.
