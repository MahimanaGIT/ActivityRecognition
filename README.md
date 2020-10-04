# Activity Recognition 

Framework: Tensorflow
Dataset: UCF-Action Dataset (Classification problem with 10 classes in dataset)

Techniques Used:

1. Transfer Learning with 2D CNN: Used Inception Ver 3 pretrained model concatenated with couple of 2D CNN layers and Dense layers for output, achieved an accuracy of 65%.


2. 2D CNN + LSTM: Used Time Distributed input from 2D CNN layers and add LSTM layers to infer changes in motion and learn features for activity classification. Accuracy Achieved: 53%

3. Transfer Learning with LSTM: Following techniques were tried:
    i. Pretrained Inception Ver 3, non trainable layers, chopped last layer + LSTM + Dense, Accuracy Achieved: 13%

    ii. Pretrained Inception Ver 3, last 20 layers are retrained + LSTM + Dense, Accuracy Achieved: 15%

    iii. Pretrained Inception Ver 3, chopped off last 20 layers + LSTM + Dense, Accuracy Achieved: 55%

Things to be done:
1. Making transfer learning + LSTM model better to good accuracy:
    i. Adding more layers to slower down the training loss decreasing
