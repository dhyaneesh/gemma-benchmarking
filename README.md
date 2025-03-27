---

# 🔍 LLM Benchmarking Suite

A comprehensive benchmarking suite for evaluating Gemma and other language models on various benchmarks including MMLU (Massive Multitask Language Understanding) and GSM8K (Grade School Math 8K).

---

## 🚀 Features

- ✅ Support for Gemma models (2B and 7B)
- 🔍 Support for Mistral models
- 📊 MMLU benchmark implementation
- 🔢 GSM8K benchmark implementation
- 🔌 Configurable model parameters
- 🔒 Secure HuggingFace authentication
- 📈 Detailed results reporting and visualization
- 📊 Interactive plots and summary reports

---

## 🛠️ Setup Instructions

### ✅ Prerequisites

Ensure you have the following installed:

- Python **3.8+**
- CUDA-capable GPU (recommended)
- HuggingFace account with access to Gemma models

---

### 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/gemma-benchmarking.git
cd gemma-benchmarking
```

2. **Create and activate a virtual environment**

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
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
</details>

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🔒 Authentication

For models that require authentication (like Gemma), you need to log in to HuggingFace:

```bash
huggingface-cli login
```

This will prompt you to enter your HuggingFace token. You can get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens).

---

## ⚙️ Configuration

All benchmarking settings are controlled via JSON configuration files in the `configs/` directory.

- The **default configuration** is available at: `configs/default.json`
- You can create custom configs to tailor model selection, datasets, and evaluation settings.

---

## 📈 Usage

### Running Benchmarks

Run with the **default config**:

```bash
python src/main.py
```

Run with a **custom config**:

```bash
python src/main.py --config path/to/config.json
```

Specify models and benchmarks via CLI:

```bash
python src/main.py --models gemma-2b mistral-7b
```

Run specific benchmarks:

```bash
python src/main.py --benchmarks mmlu gsm8k
```

Enable verbose output:

```bash
python src/main.py --verbose
```

### Generating Reports

After running benchmarks, generate visualization reports:

```bash
python scripts/generate_report.py
```

Customize report generation:

```bash
python scripts/generate_report.py --results_dir custom_results --output_dir custom_reports --output_name my_report
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
├── reports/              # Visualization reports and plots
├── scripts/              # Utility scripts
│   ├── generate_report.py  # Report generation script
│   └── prepare_data.py     # Dataset preparation scripts
├── src/                  # Source code
│   ├── benchmarks/       # Benchmark task implementations
│   │   ├── base_benchmark.py  # Base benchmark class
│   │   ├── mmlu.py           # MMLU benchmark
│   │   └── gsm8k.py          # GSM8K benchmark
│   ├── models/           # Model wrappers and loading logic
│   ├── utils/            # Helper utilities and tools
│   ├── visualization/    # Visualization and reporting tools
│   │   └── plotter.py       # Results plotting
│   └── main.py           # Entry point for benchmarking
└── README.md             # You're here!
```

---

## 📊 Available Benchmarks

### MMLU (Massive Multitask Language Understanding)
- Evaluates models across 57 subjects
- Supports few-shot learning
- Configurable number of examples per subject

### GSM8K (Grade School Math 8K)
- Tests mathematical reasoning capabilities
- Step-by-step problem solving
- Few-shot learning support
- Detailed accuracy metrics

---

## 📌 Roadmap

- [ ] Add support for additional Gemma model variants
- [ ] Expand academic benchmark integration
- [ ] Add HumanEval benchmark implementation
- [ ] Improve visualization and report automation
- [ ] Add leaderboard comparison with open models (e.g., LLaMA, Mistral)
- [ ] Docker support and multiplatform compatibility
- [ ] Add CLI wizard for quick setup

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

### 🙌 Contributing

Pull requests, issues, and suggestions are welcome! Please open an issue or start a discussion if you'd like to contribute.

---

## 📄 Acknowledgments

- Google for the Gemma models
- Mistral AI for the Mistral models
- HuggingFace for the transformers library and model hosting
- The MMLU and GSM8K benchmark creators

---
