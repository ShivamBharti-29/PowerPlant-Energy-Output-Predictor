# ⚡ Power Plant Energy Output Predictor (PyTorch)

This project implements an **Artificial Neural Network (ANN)** using **PyTorch** to predict the net hourly electrical energy output ($EP$) of a Combined Cycle Power Plant. By analyzing ambient environmental factors, the model provides high-precision regression for energy forecasting.

---

### 🚀 Features

* **PyTorch Core:** Built using `torch.nn` for modular and efficient deep learning.
* **Deep Regression:** Designed to capture non-linear relationships between weather data and power output.
* **Standardized Pipeline:** Includes full data preprocessing with `StandardScaler` for optimized neural network training.
* **Performance Tracking:** Uses Mean Squared Error (MSE) to monitor and minimize prediction variance.

---

### 📊 Model Methodology



* **Data Preprocessing:** Features ($T$, $V$, $AP$, $RH$) are scaled to a standard normal distribution before being converted into PyTorch tensors.
* **ANN Architecture:** * **Input Layer:** 4 neurons receiving environmental data.
    * **Hidden Layers:** Two dense layers (`nn.Linear`) with 6 neurons each, utilizing **ReLU** activation for non-linearity.
    * **Output Layer:** 1 neuron providing the continuous predicted value for $EP$.
* **Optimization:** The model is trained using the **Adam** optimizer to find the global minimum of the loss function efficiently.

---

### 🛠 Tech Stack

* **Language:** Python
* **Framework:** PyTorch (`torch`, `torch.nn`)
* **Libraries:** Scikit-Learn, Pandas, NumPy
* **Environment:** Jupyter Notebook / Anaconda

---

### 📈 Business Impact

* **Grid Efficiency:** Enables more accurate energy supply forecasting for grid operators.
* **Cost Reduction:** Minimizes operational overhead by automating output estimations.
* **Environmental Adaptation:** Helps plants adjust operations based on fluctuating ambient temperature and pressure.

---

### 📁 Project Structure

```text
ANN_Regression/
├── regression.ipynb       # PyTorch Model training and evaluation
├── app.py                 # Application script
├── prediction.py          # Prediction utilities
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation












