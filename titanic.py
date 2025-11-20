import matplotlib.pyplot as plt
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn import (
    ensemble,
    preprocessing,
    tree,
)
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)
from yellowbrick.classifier import (
    ConfusionMatrix,
    ROCAUC,
)
from yellowbrick.model_selection import (
    LearningCurve,
)

df = pd.read_csv("train.csv")
orig_df = df.copy()

print(df.head())
print(df.dtypes)

profile = ProfileReport(df, title="Titanic Dataset Profiling Report")
profile.to_file("titanic_report.html")
