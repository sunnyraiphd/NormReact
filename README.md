# NormReact

## Beyond Right and Wrong: Evaluating Second-order Social Reasoning in Large Language Models

## 📌 Overview

Most AI alignment efforts have focused on first-order social norms—teaching models what is socially acceptable or unacceptable (e.g., "do not steal").

However, human social intelligence depends not only on recognizing norms, but also on anticipating who will enforce them and how (e.g., public shame, confrontation, or inaction). These second-order expectations, known as **metanorms**, govern how people respond when social rules are broken.

This project introduces a framework for evaluating **metanorm reasoning in large language models (LLMs)** across two key dimensions:

* Emotional appraisal
* Behavioral response

We also propose two new prediction tasks:

* Self-regulation: how violators respond to their own actions
* Other-regulation: how observers respond to violations

---

## 📊 NormReact Dataset

We release NormReact, a multi-perspective dataset consisting of:

* 450 norm violation scenarios
* Hand-annotated labels for:

  * Emotional reactions
  * Behavioral responses
* Controlled variations across:

  * Violator gender
  * Observer social closeness

This dataset enables systematic study of social enforcement dynamics beyond simple norm recognition.

**Access the dataset:**
- **Hugging Face**: [NormReact/NormReact](https://huggingface.co/datasets/NormReact/NormReact)
- **This repository**: `data/processed` folder

---

## 🔍 Key Findings

Across six large language models, we find that:

* Models overpredict punitive responses, even when humans expect inaction
* They depict a harsher social world than human judgments suggest
* Alignment with human expectations deteriorates as social distance increases

These results suggest that current LLMs:

* Over-represent punishment
* Under-represent tolerance and restraint
* Mischaracterize how relationships shape norm enforcement

---

## ⚠️ Implications

These biases have important implications for AI systems deployed in socially sensitive domains, including:

* Conflict mediation
* Policy simulation
* Social decision-making systems

Without careful calibration, such systems may produce distorted representations of real-world social regulation.

---

## 📄 Paper

Coming soon.

---

## 📁 Repository Structure

```id="3ry1q1"
NormReact/
├── data/                       # LLM experiment and analysis datasets
│   ├── raw/
│   │   ├── main/               # Main LLM experiment responses
│   │   └── sensitivity/        # Sensitivity experiment responses
│   └── processed/              # Cleaned and preprocessed analysis datasets
├── human/                      # Human experiment data and R analysis
│   ├── data/                   # Human experiment datasets
│   ├── output/                 # Human analysis outputs
│   ├── 1-human-analysis.qmd
│   ├── 2-human-moral-dimention.qmd
│   └── 3-human-llm.qmd
├── notebooks/                  # Python experiment and analysis notebooks
│   ├── analysis.ipynb
│   └── llm_experiments.ipynb
├── outputs/
│   ├── tables/                 # Supplementary analysis tables
│   └── figures/                # Generated figures and plots
├── scripts/                    # Standalone scripts for model experiments
├── Appendix.pdf
├── README.md                   
├── requirements.txt            
└── .gitignore                  
```

---

## ⚙️ Setup

```bash id="f2xb6o"
git clone https://github.com/sunnyraiphd/NormReact.git
cd NormReact
```

---

## 🧪 Tasks

This repository supports evaluation of:

* Emotion prediction for norm violations
* Behavioral response classification
* Self-regulation vs. other-regulation modeling

---

## 📚 Citation

If you use NormReact in your research, please cite:

Rai S, Kuang J, Jamalova R, Lou A, Bicchieri C, Malhotra N, Orozco-Olvera VH, Munoz-Boudet AM, Ungar LH, Guntuku SC. **Beyond Right and Wrong: Evaluating Second-order Social Reasoning in Large Language Models.** *arXiv preprint* arXiv:2609.05437. 2026 Jul 17.

📄 [Read the paper on arXiv](https://arxiv.org/abs/2609.05437)

*Note: The linked arXiv preprint is an earlier version of the paper. The updated NeurIPS version is not yet publicly available.*

---


## 📄 License

CC BY-SA 4.0

---

## 📬 Contact

For questions or collaboration, please open an issue or contact:

