# SVM from Scratch: Minimal Implementation

This folder contains a minimal implementation of a linear Support Vector Machine (SVM) from scratch, compared directly with scikit-learn's `LinearSVC` on the Breast Cancer dataset (reduced to 2D with PCA for visualization).

## What is this?
- `svm_scratch.py` — Pure Python SVM (no ML libraries for the core logic)
- Compares our SVM with scikit-learn's SVM
- Visualizes decision boundaries and prints accuracy

## Results
- **Custom SVM Accuracy:** 0.9912
- **scikit-learn SVM Accuracy:** 0.9912

The plot below shows how both models draw their decision boundaries. Our scratch SVM matches scikit-learn's output:

![Decision Boundary Comparison](figure_1.png)

## How to Run
```bash
python svm_scratch.py
```

## Output
- Prints accuracy for both models
- Shows a plot comparing decision boundaries

---

**Note:** This code is for learning and demonstration. It is intentionally minimal and not optimized for production.
