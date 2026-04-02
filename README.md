# Deploying Deep Learning for 12 lead ECG Analysis on Edge Devices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### Description
- Author : Koy Daniel
- Internship Establishment : Institute of Technology of Cambodia (ITC)
- Department : Department of Electrical and Energy Engineering (GEE)
- Supervisor : Mr.Sum Rithea
- Date : 8th July 2025
- Project : Bechelor Degree Thesis Topic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Introduction
This project aims to bring deep learning–based ECG analysis to low-power edge devices. A custom 1D CNN model was trained to classify 12-lead ECG signals into 8 heart conditions. The model was optimized and deployed on ESP32-S3, Raspberry Pi 4, and PC. It enables real-time, on-device heart monitoring—making AI-powered diagnosis more accessible and efficient in resource-limited settings.

![Figure 1: Deep Learning with Cardiovascular Disease](/figure/image-1.png)
<p align="center"><i>Figure 1: Deep Learning with Cardiovascular Disease</i></p>

## Project Overview
This project explores the deployment of a lightweight 1D-CNN-ResNet deep learning model for 12-lead ECG signal classification. The model is trained and quantized, then deployed across thress hardware platforms:

- 🧠 ESP32-S3 (Primary Edge Device)
- 🍓 Raspberry Pi (Secondary Edge Device)
- 💻 Personal Computer (PC with Intel Core i7 + RTX3050)

The gold is to enable an **ECG interpretation** on resource-constrained edge devices for portable health monitoring.

**1. Project Backgroung**\
This figure shows the full pipline of the project-from collecting ECG data using the PTB database and *MS400 Simulator Machine*, through data segmentation and normalization, to training a 1D-CNN model. The trained model is quantized into TensorFlow Lite formatn and deployed on three platforms: *ESP32-S3, Raspberry Pi,* and *PC*. 

![Figure 2: Project Workflow](/figure/image-2.png)
<p align="center"><i>Figure 2: Block diagram of project workflow</i></p>

**2. Project Objective**\
This study aim to:
  - To develop 1D Convolutional Neural Network (1D-CNN) model to classify 12-lead ECG into 8 Cardiovascular classes
  - To quantize and deploy the quantized model on three platforms: ESP32-S3, Raspberry Pi, and PC.
  - To evaluate and compare the platforms based on inference performance, resource usage, and power consumption.

![Figure 3: Block diagram of study objective](/figure/image-3.png)
<p align="center"><i>Figure 3: Block diagram of study objective</i></p>

## Study Background
The goal of this study background is to critically analyze and synthesize the existing research relevant to this project. It aims to provide a comprehensive overview of key findings, identify gaps in current knowledge, and highlight opportunities for future research.

**1. Integration of Deep Learning in Cardiovascular Disease Detection**\
The Figure  explains why deep learning is used in ECG-based cardiovascular disease (CVD) diagnosis. It highlights that deep learning can automatically extract meaningful patterns from raw ECG signals, reducing the need for manual feature engineering. It also introduces key deep learning techniques—such as 1D-CNNs, RNNs, and LSTMs—that are effective for analyzing sequential ECG data and improving diagnostic accuracy.

![Figure 4: Deep Learning in ECG-Based CVD Diagnosis](/figure/image-4.png)
<p align="center"><i>Figure 4: Deep Learning in ECG-Based CVD Diagnosis</i></p>

**2. Neural Network**\
 Neural networks are *adaptive* statistical models based on an analogy with the brain’s structure. They are *adaptive* in that they can learn to estimate the parameters of some population using a small number of exemplars (one or a few) at a time.

 ![Figure 5: (a). Physical neuron, (b). Artificial Neural Network](/figure/image-5.png)
 <p align="center"><i>Figure 5: (a). Physical neuron, (b). Artificial Neural Network</i></p>

   **2.1. Activation Function**\
   Activation functions are critical in neural networks for introducing non-linearity, enabling the network to model complex patterns. They transform the input signals to outputs that propagate through the network layers. Without them, networks behave like simple linear models, limiting their adaptability to real-world non-linear errors.

   - **Commonlyused activation functions:** The most common activation functions used:

   ![Figure 6: Sigmoid function, Tanh function, RELU function](/figure/image-6.png)
   <p align="center"><i>Figure 6: Sigmoid function, Tanh function, RELU function</i></p>

   - **Softmanx:** The softmax step can be seen as a generalized logistic function that takes as input a vector of scores $x \in \real^n$ and output a vector of output a vector of outputs a vector of output of output probalility $p \in \real^n$ through a softmax function at the end of the architecture. It is defined as follows:
   $$
     p = \left( 
     \begin{array}{c}
        p_1 \\
        \vdots \\ 
        p_n
     \end{array} \right)\quad
     \text{where} \quad p_i = \dfrac{e^{x_i}}{\sum\limits_{j=1}^{n}e^{x_j}}
   $$

   **2.2. One Dimentional Convolutional Neural Network (1D-CNN)**\
   A 1D-CNN (1-Dimensional Convolutional Neural Network) is a type of convolutional neural network designed to process one-dimensional data, such as time-series signals or sequences.

   ![Figure 7: Visualization of 1D-CNN convolutional process on ECG-like sequential input](/figure/image.png)
   <p align="center"><i>Figure 7: Visualization of 1D-CNN convolutional process on ECG-like sequential input</i></p>

   *Figure 7* illustrates how a 1D Convolutional Neural Network (1D-CNN) processes sequential data. In part (a), a kernel of fixed size slides across a one-dimensional input sequence to perform convolution operations, generating output values known as feature maps. These values capture important local patterns in the signal. In part (b), multiple input sequences are processed in parallel. Each sequence is passed through the convolution and pooling layers, which reduce dimensionality while preserving key features. The final outputs are then fed into a fully connected layer for classification. This process allows the model to automatically learn relevant patterns from raw signal data such as ECGs.

   **2.3. Post Training Quantization**\
   Post-Training Quantization (PTQ) reduces a trained model's size and computation by converting float32 values to lower precision (like INT8), without retraining. It uses a small calibration dataset to compute scale and zero-point for efficient inference on edge devices.
   $$
   \begin{equation}
   X_q = round \left(\dfrac{X}{S}+Z\right)
   \end{equation}
   $$

   Where:
   <table><tr><td>

   - $X_q$ – Quantized value  
   - $S$ – Scale

   </td><td>

   - $X$ – Floating-point value  
   - $Z$ – Zero-point

   </td></tr></table>

   - **Symmetric Signed Quantization with Restricted Range:** This thesis uses symmetric signed quantization with restricted range, where the zero-point is fixed at zero and values are symmetrically distributed, e.g., $\lbrack 127, -127\rbrack$. This avoids overflow during accumulation and improves inference stability. The effective range is $\lbrack -2^{n-1}+1, 2^{n-1}-1\rbrack$; for INT8, it is $\lbrack -128, 127\rbrack$. The scale $S$ is calculated using **Equation 2**.
   $$
   \begin{equation}
   S = \dfrac{max\left(|f_{min}|, |f_{max}|\right)}{2^{n-1}-1}
   \end{equation}
   $$

   The scheme below demonstrates the mapping of the value in FP32 to the INT8 space using symmetric signed quantization with a restricted range, as depicted in *Figure 7*.

   ![Figure 8: Symmetric signed quantization scheme with restricted range of INT8](/figure/image-7.png)
   <p align="center"><i>Figure 7: Symmetric signed quantization scheme with restricted range of INT8</i></p>

   - **Dequantization:** Dequantization converts quantized integers back to floating-point values during inference using the inverse of the quantization formula:

   $$
   \begin{equation}
       X = S \times \left(X_q - Z\right) \quad\text{or}\quad X = S \times X_q
   \end{equation}
   $$

   The schema below demonstrates the mapping of the value in FP32 to the INT8 space using symmetric signed quantization with a restricted range, as depicted in *Figure 8*.

   ![Figure 8: Symmetric signed quantization scheme with restricted range of INT8](/figure/image-8.png)
   <p align="center"><i>Figure 8: Dequantization process in symmetric post-training quantization</i></p>

## Methodology
**1. Dataset for traning**\
In this project the datasets are collected from the PTB Diagnostic ECG Database. Once the dataset collected all the data is feed into the MS400 Simulator Machine than the signal is lived back on the Simulator.\
**1.1. Data Record**\
This section shows the procedure of capturing live ECG data from the MS400 ECG simulator machine.
- **Step 1: Signal selection and export**\
   Relevent ECG signal from the database were selected and exported in *.dat* format. From the *Figure* show the samples of PTB data.

   ![Figure 9: Sample of PTB Diagnostic ECG Database](/figure/image-9.png)
   <p align="center"><i>Figure 8: Sample of PTB Diagnostic ECG Database</i></p>

   1. The *s0035_re.dat* contain the raw binary ECG signal data, typically samples at 100Hz and store the value as 16-bit integers without any headers or metadata.
   2. The *s0035_re.hea* file accompanies each *.dat* file and contains metadata describing the ECG recording. 
- **Step 2: Case Conversion for Compatibility**\
   The collected *.dat* files were processed with a case conversion tool to make them compatible with the MS400 ECG Simulator.  *Figure 3.3* illustrates the step-by-step procedure for this conversion process.

   ![Figure 10: Case conversion tool main interface configuration](/figure/image-10.png)
   <p align="center"><i>Figure 10: Case conversion tool main interface configuration</i></p>
