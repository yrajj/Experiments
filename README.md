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
```
getLegalEntities

getLegalEntities

Retums the list of legal entities for given string.

legalEntity,

The legal entity value to check. This parameter is mandatory.

getByEntity

getByEntity

Returns financial metrics for a legal entity, product, or relationship.

entity

The column that needs to be filtered. This parameter is mandatory.

Allowed Values: 'legal_entity', 'product_desc', 'relationship_id.

Info: relationship_id is used to filter by SUP.

entityValue

The value of the entity filter. This parameter is mandatory.

selectDimensions

Create

The selection dimensions for the query. This parameter is optional

String Format: column name delimited by pipe

Allowed values: "legal entity, booking officer, product desc, 'sub lob, lob', 'n mis date skey, legal, entity(product desc

legal_entity/product descub Job

periodAggregationType

The aggregation period for the data. This parameter is optional

Allowed Values: LTM, YTD, QTD, MTD', 'PYLTM, PYYTD PYOTO

Default Value: LIM.

orderBy

The columns to order the results by. This parameter is optional.

Example: revenue DESC, 'ptpp ASC, etc.

dateFilter

The year, month, or date for filtering the data. This parameter is optional

Allowed Values: yearyyyy, monthyyyymm.

Example: year: 2024, "month 202301

A

getEntityByOfficer

getByEntityByOfficer

Retums financial metrics for a legal entity, product, or relationship.

officerName

Officer value provided by user in prompt. This parameter is mandatory

String Format: Last name, Fist name, in case only first or last name is given, provide only that

entity

The column that needs to be filtered. This parameter is mandatory. Allowed Values: legal entity, 'product desc', 'relationship id

info: relationship id is used to filter by SUP.

entityValue

The value of the entity filter. This parameter is mandatory.

select Dimensions

The selection dimensions for the query. This parameter is optional.

String Format: column name delimited by pipe (B.

Allowed values: legal entity', 'product desc, sub lob', 'booking officer', 'lob, mis date skey, llegal entity/product, desc

Segal entityfproduct deschub Job

periodAggregation Type

The aggregation period for the data. This parameter is optional Allowed Values: LTM, YTD, QTD, MID, PYLTM, PYYTD, PYOTO

Default Value: LTM.

orderBy

The columns to order the results by. This parameter is optional

Example: revenue DESC", 'ptpp ASC, etc.

dateFilter

The year, month, or date for filtering the data. This parameter is optional Allowed Values, yesryyyy, monthyyyymm.

Example year 2024, month 202301

getRanked Entity

getRanked Entity

Retums financial metrics and provides ability to sort the column as top or bottom.

entity

Column that needs to be ranked. This parameter is mandatory.

Allowed Values: 'legal entity, 'product desc

rankType

Rank type: accepts the sorting order. Ascending means bottom, Descending means top. This parameter is optional.

Allowed Values: 'top', 'bottom'

Default Value Top

numberOfRecords

Number of records to fetch. This parameters optional

Allowed Values: 0 to 20

Default Value: 10

Parameter DataTypes Integer

rankMetric

Metric/column on which ranking is needed. This parameter is optional

Allowed Values 'ptpp, revenue, 'sva, toe, blended equity, 'revenueline, revenuejptpp revenuessa toive, ptppator

String Format: column name delimited by pipe (

Default Value ptpp

period AggregationType

Aggregation Penod. This parameter is optional

Allowed Values: LTM, YTD, QTD, MID, PYLIM, PYYTO, PYQTD

Default Value LTM

periodAggregation Type

Aggregation Period. This parameter is optional.

Allowed Values: LTM, YTD', 'QTD, MTD, PYLIM, PYYTΟ, ΡΥΩΤΟ

Default Value: LTM

dateFilter

Year or month of filtering. This parameter is optional. Allowed Values yearyyyy, 'monthyyyymm

Example: year 2024 or month: 202301

コー

getRanked EntityByOfficer

getRanked EntityByOfficer

Returns financial metrics for a given officer and provides ability to sort the column as top or bottom.

officer Name

The name of the officer used to filter results.

entity

Column that needs to be ranked. This parameter is mandatory.

Allowed Values: legal entity, product desc

rankType

Rank type: accepts the sorting order. Ascending means bottom, Descending means top. This parameter is optional

Allowed Values: top, bottom

Default Value top

numberOfRecords

Number of records to letch. This parameter is optional.

Allowed Values: 0 to 20

Parameter DataType Integer

Default Value: 10

rankMetric

Metric/column on which ranking is needed. This parameter is optional

String Format: column name delimited by pipe ().

Allowed Values 'ptpp', 'revenue', 'wa', 'toe', 'blended, equity, revenuejice, revenuelptpp, revenuelsva, Yoejsva, ptpipisaloe etc

Default Value: ptpp

periodAggregationType

Aggregation Period. This parameter is optional

Allowed Values LTM, YTO, QTO, MTD, PYLIM, PYYTO, PYQTO

Default Value: TM

dateFilter

Wear or month of filtering. This parameter is optional.

Allowed Values: yearyyyy', 'monthyyyymm

Example: year2024 of month 202301

getDimension Mapping

getDimensionMapping

Retums related columns to the given columns filter. This method retrieves mappings or relationships for a specified dimension based on the provided filters



```
