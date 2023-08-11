# MDCSI

Welcome to the MCSI! This repository contains code, datasets, and documentation related to the research and experiments conducted to enhance intent classification for german cmulti-domian customer support que. The primary focus of this project is to develop and improve intent classification models, particularly in scenarios with limited labeled data, aiming to bridge the gap between academia and industry.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Experiments](#experiments)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project explores practical aspects of intent classification in industrial settings, with the goal of enhancing model performance, generalization, and real-world applicability. I investigate techniques such as Sentence Transformer Finetuning (SETFIT) and explore innovative approaches to improve few-shot intent classification, addressing challenges faced in customer support queries and similar complex intents.

## Key Features

- **SETFIT Implementation:** Code for implementing the Sentence Transformer Finetuning (SETFIT) approach for few-shot intent classification.
- **Experimental Analysis:** Jupyter notebooks and scripts for conducting experiments, evaluating model performance, and comparing results.
- **SETFITSOUP Exploration:** Investigating the weighted averaging of embeddings (SETFITSOUP) to enhance model generalization during inference.
- **Documentation:** Detailed explanations of methodologies, code usage, and research findings.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Review the documentation and code in the respective folders.
3. Experiment with the provided datasets and SETFIT implementation.
4. Explore the SETFITSOUP experiment for further enhancements in model generalization.

## Dataset

We have developed a novel Multi-domain intent classification dataset, which comprises authentic customer requests. This dataset reflects the diversity of industrial scenarios, making it valuable for evaluating intent classification models in real-world applications. Detailed information about the dataset, its format, and usage can be found in the `dataset/` folder.
FURTHERMORe the dataSET AND ITS SUBSET PER DOMIAN ARE FOUND ON HUGGINGFACE

## Code Structure

The code in this repository is organized as follows:
- `scripts`: Implementation of the Sentence Transformer Finetuning (SETFIT) approach and the setfitfusion expir .
- `notebooks/`: Exploration of the dataset and and prepsaration.

## Experiments

We have conducted experiments to evaluate the performance of intent classification models, comparing SETFIT and SETFITSOUP with traditional methods. Results, analysis, and insights can be found in the `experiments/` folder.

## Contributing

Contributions to this project are welcome! If you find bugs, have suggestions, or want to contribute improvements, please open an issue or submit a pull request. Let's work together to enhance the capabilities of intent classification in industrial settings.

## License

This project is licensed under the [MIT License](LICENSE), allowing you to use and modify the code in this repository for your own purposes. Please review the license for more details.
