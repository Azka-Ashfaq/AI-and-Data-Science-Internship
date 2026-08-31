---

## 🎯 Goal
Predict if someone earns >50K/year using simple baseline models

---

## 📚 Dataset Information
| Property | Value |
|----------|-------|
| **Dataset** | UCI Adult Census Income |
| **Samples** | 48,842 |
| **Features** | 14 |
| **Target** | Binary (≤50K or >50K) |
| **Rich People** | 23.9% |
| **Not Rich** | 76.1% |

---

## 🧠 Models Implemented

### Baseline 1: Always Guess "Not Rich"
- Uses majority class prediction
- **Strategy:** Predict 0 for everyone

### Baseline 2: Simple Rule
- **Rule:** Predict rich if education >= 13 OR capital-gain > 0
- **Logic:** People with bachelor's degree or investment income

### Baseline 3: Advanced Rule
- **Rules:**
  - education-num >= 13 (high education), OR
  - capital-gain > 5000 (significant investment income), OR
  - hours-per-week > 40 AND education-num >= 12, OR
  - married AND high education

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | Rich Caught |
|-------|----------|-----------|--------|----------|-------------|
| Baseline 1 | 0.760 | 0.000 | 0.000 | 0.000 | 0 |
| Baseline 2 | 0.742 | 0.443 | 0.529 | 0.482 | 1,238 |
| Baseline 3 | 0.724 | 0.426 | 0.608 | 0.501 | 1,423 |

### 🏆 Best Model: Baseline 3 (Advanced Rule)
- **F1-Score:** 0.501
- **Rich People Caught:** 1,423 out of 2,340 (60.8%)

---

## 📈 Performance Analysis

### What Each Metric Tells Us:
| Metric | Explanation |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of those predicted rich, how many were actually rich |
| **Recall** | Of all rich people, how many did we catch |
| **F1-Score** | Harmonic mean of precision and recall |

### Confusion Matrix - Baseline 3
Predicted
                Not Rich  Rich
Actual Not Rich   5,837   1,592
       Rich         917   1,423
       - **True Negatives:** 5,837 (correctly predicted not rich)
- **False Positives:** 1,592 (incorrectly predicted rich)
- **False Negatives:** 917 (missed rich people)
- **True Positives:** 1,423 (correctly predicted rich)

---

## 🚀 How to Run

### 1. Clone the Repository
bash
git clone https://github.com/Azka-Ashfaq/Al-and-Data-Science-Internship.git
cd Al-and-Data-Science-Internship
---

## 👤 Author

*Azka Ashfaq*

- GitHub: [@Azka-Ashfaq](https://github.com/Azka-Ashfaq)
- LinkedIn: [Azka Ashfaq](https://linkedin.com/in/azka-ashfaq)

 📅 Date
August 31, 2026
