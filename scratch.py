import sys, platform
import numpy as np, pandas as pd, sklearn, matplotlib, seaborn as sns
import xgboost, lightgbm, shap, phik

print(f"Python   : {sys.version.split()[0]}  ({platform.platform()})")
print(f"NumPy    : {np.__version__}")
print(f"Pandas   : {pd.__version__}")
print(f"sklearn  : {sklearn.__version__}")
print(f"XGBoost  : {xgboost.__version__}")
print(f"LightGBM : {lightgbm.__version__}")
print(f"SHAP     : {shap.__version__}")
print(f"phik     : {phik.__version__}")
print("\n✅ All imports clean — environment is ready.")