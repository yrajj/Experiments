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
