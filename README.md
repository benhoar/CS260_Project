# CS260_Project
Project for CS260
The data is derived from the stock data folder provided by this source
https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs

The "Data_Preparation.ipynb" has the code for gathering data from the large folder of stocks and for the subsequent calculation of the indicators based on rules derived from investopedia.com and stockcharts.com

To Run Models:
1. Ensure that Pytorch is installed by running 'pip install torch torchvision'.

2. Uncomment lines with 'For training on GPU' to train with GPU instead of CPU.

3. Run 'python3 stocks_*.py' to execute any of the scripts for MLP or SVM models

4. Script will run and train each model with the stock indicators dataset, outputting loss and accuracy.

5. Modify 'learning_rate' variable to test different step size values. Change 'DROP_NUM' variable to test the effect of dropping a certain number of features from the dataset.