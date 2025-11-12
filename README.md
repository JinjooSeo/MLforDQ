# BDT Training for DQ Framework (hipe4ML + XGBoost)


This script provides a **training and export pipeline** for BDT models using [**hipe4ml**](https://github.com/hipe4ml/hipe4ml), optimized for the **O2 DQ framework**.

It automates:

1. Tree loading from `DielectronsAll` tables  
2. Preselection and variable filtering  
3. Model training per **pT** and **centrality** bin  
4. Optimization of hyperparameters with **Optuna**  
5. Determination of the **best BDT cut** based on pseudo-significance  
6. Export to **ONNX** format and **JSON configuration** (Hyperloop-ready)

---

## Requirements

### Python dependencies
```bash
pip install hipe4ml hipe4ml_converter
