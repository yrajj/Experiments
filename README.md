```
import pandas as pd
import os

pipe = coe_pipeline

# Define output path
path = "model_params_dump"
os.makedirs(path, exist_ok=True)

### ---- Extract Vectorizer ----
vectorizer = dict(pipe.named_steps['preprocessor'].named_transformers_)['vectorizer']

# Save vectorizer config
vec_config = vectorizer.get_params()
pd.DataFrame(list(vec_config.items()), columns=["parameter", "value"]) \
    .to_csv(os.path.join(path, "vectorizer_config.csv"), index=False)

# Save vocabulary
vocab_df = pd.DataFrame(list(vectorizer.vocabulary_.items()), columns=["token", "index"])
vocab_df.to_csv(os.path.join(path, "vectorizer_vocabulary.csv"), index=False)

# Save IDF values
if hasattr(vectorizer, "idf_"):
    idf_df = pd.DataFrame({
        "token": [t for t, i in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])],
        "idf": vectorizer.idf_
    })
    idf_df.to_csv(os.path.join(path, "vectorizer_idf.csv"), index=False)

### ---- Extract Classifier ----
classifier = pipe.named_steps['classifier']

# CalibratedClassifierCV parameters
calib_config = classifier.get_params()
pd.DataFrame(list(calib_config.items()), columns=["parameter", "value"]) \
    .to_csv(os.path.join(path, "calibrator_config.csv"), index=False)

# Base estimator (LinearSVC)
base_est = classifier.base_estimator
svc_config = base_est.get_params()
pd.DataFrame(list(svc_config.items()), columns=["parameter", "value"]) \
    .to_csv(os.path.join(path, "linear_svc_config.csv"), index=False)

# LinearSVC weights (coef_ and intercept_)
if hasattr(base_est, "coef_"):
    coef_df = pd.DataFrame(base_est.coef_)
    coef_df.to_csv(os.path.join(path, "linear_svc_coef.csv"), index=False)

if hasattr(base_est, "intercept_"):
    intercept_df = pd.DataFrame(base_est.intercept_)
    intercept_df.to_csv(os.path.join(path, "linear_svc_intercept.csv"), index=False)

### ---- Calibration parameters (from calibrated_classifiers_) ----
if hasattr(classifier, "calibrated_classifiers_"):
    for i, calib_clf in enumerate(classifier.calibrated_classifiers_):
        calib = getattr(calib_clf, "calibrator", None)
        if calib is None:
            continue

        # Save calibrator hyperparameters
        calib_params = calib.get_params()
        pd.DataFrame(list(calib_params.items()), columns=["parameter", "value"]) \
            .to_csv(os.path.join(path, f"calibrator_{i}_config.csv"), index=False)

        # If sigmoid scaling (LogisticRegression)
        if hasattr(calib, "coef_"):
            coef_df = pd.DataFrame(calib.coef_)
            coef_df.to_csv(os.path.join(path, f"calibrator_{i}_coef.csv"), index=False)
        if hasattr(calib, "intercept_"):
            intercept_df = pd.DataFrame(calib.intercept_)
            intercept_df.to_csv(os.path.join(path, f"calibrator_{i}_intercept.csv"), index=False)

        # If isotonic regression scaling
        if hasattr(calib, "X_thresholds_") and hasattr(calib, "y_thresholds_"):
            iso_df = pd.DataFrame({
                "X_thresholds": calib.X_thresholds_,
                "y_thresholds": calib.y_thresholds_
            })
            iso_df.to_csv(os.path.join(path, f"calibrator_{i}_isotonic.csv"), index=False)


print(f"✅ All pipeline parameters dumped into '{path}/' folder.")

```

```
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# --------- Step 1: Load vectorizer config, vocab, IDF ----------
vec_config_df = pd.read_csv("model_params_dump/vectorizer_config.csv")
vectorizer_config = dict(zip(vec_config_df['parameter'], vec_config_df['value']))

# Convert some types that were saved as string
for key in ['max_df', 'min_df', 'sublinear_tf', 'lowercase', 'ngram_range']:
    if key in vectorizer_config:
        if key in ['sublinear_tf', 'lowercase']:
            vectorizer_config[key] = vectorizer_config[key] == 'True'
        elif key == 'ngram_range':
            # from string "(2, 2)" to tuple
            vectorizer_config[key] = tuple(map(int, vectorizer_config[key].strip("()").split(",")))
        else:
            vectorizer_config[key] = float(vectorizer_config[key])

# Load vocabulary
vocab_df = pd.read_csv("model_params_dump/vectorizer_vocabulary.csv")
vocab = dict(zip(vocab_df['token'], vocab_df['index']))

# Initialize vectorizer with config and vocab
vectorizer = TfidfVectorizer(**vectorizer_config)
vectorizer.vocabulary_ = vocab

# Load IDF values
idf_df = pd.read_csv("model_params_dump/vectorizer_idf.csv")
vectorizer.idf_ = np.array(idf_df.sort_values('token')['idf'])
vectorizer._tfidf._idf_diag = np.diag(vectorizer.idf_) if hasattr(vectorizer, "_tfidf") else None

# --------- Step 2: Load LinearSVC weights ----------
svc_config_df = pd.read_csv("model_params_dump/linear_svc_config.csv")
svc_config = dict(zip(svc_config_df['parameter'], svc_config_df['value']))

# Convert types
for key in ['C', 'max_iter']:
    if key in svc_config:
        svc_config[key] = float(svc_config[key])

if 'class_weight' in svc_config and svc_config['class_weight'] == 'balanced':
    svc_config['class_weight'] = 'balanced'

svc = LinearSVC(**svc_config)

# Load coef_ and intercept_
coef_df = pd.read_csv("model_params_dump/linear_svc_coef.csv")
intercept_df = pd.read_csv("model_params_dump/linear_svc_intercept.csv")
svc.coef_ = coef_df.values
svc.intercept_ = intercept_df.values
svc.classes_ = np.arange(svc.coef_.shape[0]) if svc.coef_.shape[0] > 1 else np.array([0, 1])

# --------- Step 3: Run inference ----------
# Assume test_df has a column 'processed_text' with text data
def run_inference(test_df, text_column='processed_text'):
    X_test = vectorizer.transform(test_df[text_column])
    preds = svc.predict(X_test)
    return preds

# Example usage
# test_df = pd.DataFrame({'processed_text': ["sample text 1", "another example"]})
# predictions = run_inference(test_df)
# print(predictions)
```
