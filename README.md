# ⚡ Power Plant Energy Output Predictor (PyTorch)

This project implements an **Artificial Neural Network (ANN)** using **PyTorch** to predict the net hourly electrical energy output ($EP$) of a Combined Cycle Power Plant. The model analyzes environmental factors to provide high-accuracy regression.

---

### 🚀 Features

* **PyTorch Tensors:** Utilizes high-performance tensor computations for model training.
* **Dynamic Graphing:** Leverages PyTorch's autograd for efficient backpropagation.
* **Feature Engineering:** Implements standard scaling to normalize input variables like Temperature ($T$) and Ambient Pressure ($AP$).
* **Evaluation Metrics:** Includes loss tracking to monitor convergence during the training loop.

---

### 📊 Model Methodology



* **Data Preprocessing:** Data is split into training and testing sets, then converted into PyTorch Tensors after being scaled with `StandardScaler`.
* **ANN Architecture:** * **Input Layer:** 4 neurons for environmental features.
    * **Hidden Layers:** A Sequential stack of two Linear layers with 6 neurons each, utilizing **ReLU** activation.
    * **Output Layer:** A single Linear neuron for continuous value regression ($EP$).
* **Optimization:** The model uses the **Adam** optimizer and **MSELoss** (Mean Squared Error) to minimize prediction error.

---

### 🛠 Tech Stack

* **Language:** Python
* **Framework:** PyTorch (torch, torch.nn)
* **Data Science:** Scikit-Learn, Pandas, NumPy
* **Environment:** Jupyter Notebook / Anaconda

---

### 📈 Business Impact

Accurate energy prediction allows for:
* **Proactive Load Balancing:** Better integration of power plant output into the national grid.
* **Operational Savings:** Reducing fuel waste by predicting output based on current weather conditions.
* **Precision Analytics:** Moving from heuristic-based estimation to data-driven deep learning models.

---

### 📁 Project Structure

```text
ANN_Regression/
├── regression.ipynb       # PyTorch Model training and evaluation
├── app.py                 # Application script
├── prediction.py          # Prediction utilities
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

---

---

## 📬 Author

**Shivam Bharti**  
GitHub: https://github.com/ShivamBharti-29

---

⭐ If you found this project useful, consider giving it a star!

