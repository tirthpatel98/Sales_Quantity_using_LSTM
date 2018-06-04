STEP1:  #load dataset >> set path of the csv file to be trained.

STEP2:  #frame as supervised learning >> change the series_to_supervised method's second and third attributes to change the amount of past and future predictions; default= 'past=2''future=4'

STEP3:  #split into train and test sets set >> set the number of rows for training by changing train_rows.

STEP4:  #fit network >> change epochs and batch_size values according to the data; default= 'epoch=250' 'batch_sz=30'. 

STEP5:  #plot history >> close the plot diagram after inspection.

STEP6:  #writing data to csv file >> change file name if needed; default="avg_predict.csv".
