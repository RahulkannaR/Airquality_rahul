# 📄 IEEE Published Research – Air Quality Analysis & Prediction in Tamil Nadu

> ✅ **Published in IEEE Xplore**
> 
> 🏛️ **Presented at**: *IEEE International Conference on Advanced Computing Technologies (ICoACT 2025)*
> 
> 🔗 **IEEE Link**: [https://ieeexplore.ieee.org/document/11005148](https://ieeexplore.ieee.org/document/11005148)
> 
> 📄 **DOI**: `10.1109/ICoACT548.2025.11005148`


This GitHub repository hosts the official implementation of our IEEE-published research titled:

> **"Air Quality Analysis and Prediction in Tamil Nadu: An ANN and CNN Approach"**

---

## 🧠 Project Overview

Air pollution is a critical issue in Tamil Nadu, affecting public health and the environment. This project proposes an intelligent and data-driven solution for **analyzing and predicting air quality** using **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)**.

The models were trained using historical data on pollutants such as **SO₂, NO₂, RSPM/PM10, PM2.5**, and **meteorological features**. Results demonstrate high accuracy and strong generalization, offering real-time AQI forecasting that surpasses traditional methods.

---

## ✨ Highlights

* 🌍 **Focus Region**: Tamil Nadu, India
* 🤖 **Models Used**: ANN and CNN
* 📈 **Results**:

  * **R² Score**: 0.9998
  * **MAE**: 0.037
  * **RMSE**: 0.4522
* 🛰️ **Dataset Includes**:

  * Pollutant concentrations: SO₂, NO₂, PM10, PM2.5
  * AQI values
  * Sampling Dates

---

## 📁 Repository Structure

```
📦 AirQuality-TamilNadu/
├── dataset/                  # Cleaned dataset of Tamil Nadu air pollution
├── models/                  # ANN and CNN model architectures
├── results/                 # Visual outputs: plots, heatmaps, residuals
├── utils/                   # Data preprocessing and visualization tools
├── README.md                # Project documentation (this file)
└── requirements.txt         # Python libraries used
```

---

## 🧪 Setup & Usage

### 🔧 Installation

```bash
git clone https://github.com/your-username/air-quality-analysis.git
cd air-quality-analysis
pip install -r requirements.txt
```

### ▶️ Running the Models

**To train and predict with ANN**:

```bash
python models/train_ann.py
```

**To run CNN**:

```bash
python models/train_cnn.py
```

Or launch the project in Jupyter:

```bash
jupyter notebook notebooks/air_quality_analysis.ipynb
```

---

## 📊 Visual Outputs

* 📈 Scatter plots for each pollutant vs sampling date
* 🌫️ AQI trendline over time
* 🧮 Correlation heatmap of pollutants
* 🔁 Residual plots and prediction distribution
* 🎯 Predicted AQI vs Actual AQI scatter plot

---

## 🧠 Model Insights

### 🔸 ANN

* Fully connected layers
* Input: pollutant levels + weather features
* Optimized using backpropagation and gradient descent

### 🔸 CNN

* Converts feature space into image-like grid
* Uses convolution + pooling layers for spatial feature learning
* Fully connected layers for AQI prediction

---

## 📚 Evaluation Metrics

| Metric   | Value      |
| -------- | ---------- |
| MAE      | **0.037**  |
| RMSE     | **0.4522** |
| R² Score | **0.9998** |

> ✅ The models accurately predict AQI ranges and respond well to varying pollutant levels.

---

## 📜 Citation

If you use this repository, please cite our IEEE paper:

```bibtex
@inproceedings{rahul2025airquality,
  title={Air Quality Analysis and Prediction in Tamil Nadu: An ANN and CNN Approach},
  author={R. Rahul, T. Jegan, S. Yasar Arafath, and Dr. S. Palanivel Rajan},
  booktitle={2025 IEEE International Conference on Advanced Computing Technologies (ICoACT)},
  year={2025},
  doi={10.1109/ICoACT548.2025.11005148}
}
```

---

## 👥 Authors

* **Dr. S. Palanivel Rajan**
* **R. Rahul**
* **T. Jegan**
* **S. Yasar Arafath**
  
  📧 *[rahulkanna170504@gmail.com](mailto:rahulkanna170504@gmail.com)*
  
  🏫 Velammal College of Engineering and Technology, Madurai

---

## 🏆 Acknowledgements

We thank IEEE and the ICoACT 2025 committee for providing the platform to present this work. This research contributes toward enhancing **environmental monitoring and policy support** through deep learning solutions.
