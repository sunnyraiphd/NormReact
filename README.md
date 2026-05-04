# NormReact

## Beyond Right and Wrong: Evaluating Second-order Social Reasoning in Large Language Models


---

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
- **This repository**: `data/` folder

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
├── data/              # NormReact dataset
├── human/             # Human Analysis R code
├── notebooks/         # Analysis/Experiment notebooks
└── README.md
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

Anonymous

---


## 📄 License

CC BY 4.0

---

## 📬 Contact

For questions or collaboration, please open an issue or contact:
Anonymous

