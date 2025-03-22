# Gemma Models Benchmarking Suite

A comprehensive benchmarking solution for evaluating Gemma language models against academic benchmarks and custom datasets.

## Setup Instructions

### Prerequisites

- Python 3.10+
- Git
- Windows OS
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download/) (recommended)

### Installation

1. Clone the repository:

```
git clone https://github.com/dhyaneesh/gemma-benchmarking.git
cd gemma-benchmarking
```

2. Set up environment:

   **Option 1: Using Conda (recommended)**

   ```
   conda env create -f environment.yml
   conda activate gemma-benchmark
   ```

   **Option 2: Using Python venv**

   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Configuration

The benchmarking suite is configured using JSON files in the `configs/` directory. The default configuration is provided in `configs/default.json`.

## Usage

Run the benchmarking suite with the default configuration:

```
python src/main.py
```

Use custom configuration:

```
python src/main.py --config configs/custom_config.json
```

Specify models and benchmarks directly:

```
python src/main.py --models gemma-2b gemma-7b --benchmarks mmlu gsm8k
```

## Project Structure

```
gemma-benchmarking/
├── configs/              # Configuration files
├── environment.yml       # Conda environment specification
├── logs/                 # Log files
├── requirements.txt      # Pip requirements file
├── results/              # Benchmark results
├── scripts/              # Utility scripts
├── src/                  # Source code
│   ├── benchmarks/       # Benchmark implementations
│   ├── models/           # Model wrappers
│   ├── utils/            # Utility functions
│   ├── visualization/    # Visualization tools
│   └── main.py           # Main entry point
└── README.md             # This file
```

## Features (Planned)

- Support for multiple Gemma model variants
- Integration with academic benchmarks (MMLU, GSM8K, etc.)
- Custom dataset evaluation
- Comparative analysis with other open models
- Automated reporting and visualization
- Extensible architecture for adding new benchmarks

## License

[MIT License](LICENSE)
