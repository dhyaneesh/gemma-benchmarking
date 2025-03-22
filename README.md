---

# 🔍 Gemma Models Benchmarking Suite

A comprehensive and extensible benchmarking suite for evaluating **Gemma language models** on academic benchmarks and custom datasets.

---

## 🚀 Features

- ✅ Support for multiple **Gemma model variants**
- 📊 Integration with **academic benchmarks** (MMLU, GSM8K, and more)
- 📁 **Custom dataset** evaluation support
- ⚖️ Comparative analysis with other **open-source models**
- 📈 Automated **report generation and visualization**
- 🔌 **Modular and extensible** design for easy integration of new benchmarks or models

---

## 🛠️ Setup Instructions

### ✅ Prerequisites

Ensure you have the following installed:

- Python **3.10+**
- Git
- Windows OS (for now)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download/) – **recommended for managing environments**

---

### 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/dhyaneesh/gemma-benchmarking.git
cd gemma-benchmarking
```

2. **Set up the environment**

<details>
<summary><strong>Option 1: Using Conda (Recommended)</strong></summary>

```bash
conda env create -f environment.yml
conda activate gemma-benchmark
```
</details>

<details>
<summary><strong>Option 2: Using Python venv</strong></summary>

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
</details>

---

## ⚙️ Configuration

All benchmarking settings are controlled via JSON configuration files in the `configs/` directory.

- The **default configuration** is available at: `configs/default.json`
- You can create custom configs to tailor model selection, datasets, and evaluation settings.

---

## 📈 Usage

Run with the **default config**:

```bash
python src/main.py
```

Run with a **custom config**:

```bash
python src/main.py --config configs/custom_config.json
```

Specify models and benchmarks via CLI:

```bash
python src/main.py --models gemma-2b gemma-7b --benchmarks mmlu gsm8k
```

---

## 📁 Project Structure

```
gemma-benchmarking/
├── configs/              # Configuration files (JSON)
├── environment.yml       # Conda environment specification
├── logs/                 # Log files
├── requirements.txt      # Python dependencies
├── results/              # Benchmark output results
├── scripts/              # Utility scripts (e.g., dataset preparation)
├── src/                  # Source code
│   ├── benchmarks/       # Benchmark task implementations
│   ├── models/           # Model wrappers and loading logic
│   ├── utils/            # Helper utilities and tools
│   ├── visualization/    # Visualization and reporting tools
│   └── main.py           # Entry point for benchmarking
└── README.md             # You're here!
```

---

## 📌 Roadmap

- [ ] Add support for additional Gemma model variants
- [ ] Expand academic benchmark integration
- [ ] Improve visualization and report automation
- [ ] Add leaderboard comparison with open models (e.g., LLaMA, Mistral)
- [ ] Docker support and multiplatform compatibility
- [ ] Add CLI wizard for quick setup

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

### 🙌 Contributing

Pull requests, issues, and suggestions are welcome! Please open an issue or start a discussion if you’d like to contribute.

---
