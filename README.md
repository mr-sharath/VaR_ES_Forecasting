VaR & ES Forecasting using Deep Learning
Indu Sai Atla

1. Project Overview
   
Objective: Implement a novel framework for forecasting Value-at-Risk (VaR) and Expected
Shortfall (ES) using:


• Mogrifier RNNs with Quantile Regression (QRMogRNN)


• GAN-based scenario generation for tail risk modeling

Key Innovations:

• Hybrid approach combining decomposition techniques with deep learning

• Joint modeling of market heterogeneity properties

• First implementation of GANs for ES estimation in financial risk

File Structure:
1 VaR_ES /
2 data /
3 raw / ( BLK . csv , SPX . csv )
4 processed / ( BLK_imfs . npy )
5 splits /
6 models /
7 best_qrmoglstm . pt ( Partial )
8 es_gan . h5 ( Pending )
9 scripts /
10 1 _fetch_data . py
11 2 _preprocess . py
12 3 _decompose . py
13 4 _train_var . py ( Partial )
14 5 _train_es . py ( In Progress )
15 6 _evaluate . py
16 results /
