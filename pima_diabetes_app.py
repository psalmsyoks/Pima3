from pathlib import Path

import pandas as pd
import streamlit as st
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Pima Diabetes Classifier",
    page_icon="🩺",
    layout="wide",
)

DATA_PATH = Path(__file__).with_name("Pima.csv")
TARGET = "diabetes_class"
FEATURES = [
    "preg_count",
    "glucose_concentration",
    "diastolic_bp",
    "triceps_skin_fold_thickness",
    "two_hr_serum_insulin",
    "bmi",
    "diabetes_pedi",
    "age",
]
FEATURE_LABELS = {
    "preg_count": "Pregnancies",
    "glucose_concentration": "Glucose concentration",
    "diastolic_bp": "Diastolic blood pressure",
    "triceps_skin_fold_thickness": "Triceps skin-fold thickness",
    "two_hr_serum_insulin": "2-hour serum insulin",
    "bmi": "BMI",
    "diabetes_pedi": "Diabetes pedigree function",
    "age": "Age",
}
ZERO_AS_MISSING = [
    "glucose_concentration",
    "diastolic_bp",
    "triceps_skin_fold_thickness",
    "two_hr_serum_insulin",
    "bmi",
]


@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)
    expected = set(FEATURES + [TARGET])
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Pima.csv is missing required columns: {', '.join(sorted(missing))}")
    if data.empty or data[TARGET].nunique() != 2:
        raise ValueError("Pima.csv must contain rows for both diabetes classes.")
    return data


@st.cache_resource
def train_and_evaluate():
    data = load_dataset()
    X = data[FEATURES].replace({column: {0: float("nan")} for column in ZERO_AS_MISSING})
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifiers = {
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra trees": ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=2,
            random_state=42,
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        ),
    }
    samplers = {
        "No resampling": "passthrough",
        "SMOTE": SMOTE(k_neighbors=5, random_state=42),
        "RandomOverSampler": RandomOverSampler(random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "Accuracy": "accuracy",
        "Balanced accuracy": "balanced_accuracy",
        "F1": "f1",
        "Recall": "recall",
    }
    results = []
    best_model = None
    best_combination = None
    best_cv_f1 = -1.0
    best_cv_balanced_accuracy = -1.0

    for sampler_name, sampler in samplers.items():
        for classifier_name, classifier in classifiers.items():
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("sampler", sampler),
                    ("classifier", classifier),
                ]
            )
            cv_result = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
            )
            mean_cv_f1 = cv_result["test_F1"].mean()
            mean_cv_balanced_accuracy = cv_result["test_Balanced accuracy"].mean()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            results.append(
                {
                    "Classifier": classifier_name,
                    "Resampling": sampler_name,
                    "CV F1": mean_cv_f1,
                    "CV balanced accuracy": mean_cv_balanced_accuracy,
                    "CV accuracy": cv_result["test_Accuracy"].mean(),
                    "Test accuracy": accuracy_score(y_test, predictions),
                    "Test balanced accuracy": balanced_accuracy_score(y_test, predictions),
                    "Test precision": precision_score(y_test, predictions, zero_division=0),
                    "Test recall": recall_score(y_test, predictions, zero_division=0),
                    "Test F1": f1_score(y_test, predictions, zero_division=0),
                    "Test ROC-AUC": roc_auc_score(y_test, probabilities),
                }
            )
            if (mean_cv_f1, mean_cv_balanced_accuracy) > (
                best_cv_f1,
                best_cv_balanced_accuracy,
            ):
                best_cv_f1 = mean_cv_f1
                best_cv_balanced_accuracy = mean_cv_balanced_accuracy
                best_combination = f"{sampler_name} + {classifier_name}"
                best_model = model
                best_predictions = predictions
                best_probabilities = probabilities

    comparison = pd.DataFrame(results).sort_values(
        ["CV F1", "CV balanced accuracy"], ascending=False
    ).reset_index(drop=True)
    metrics = {
        "Accuracy": accuracy_score(y_test, best_predictions),
        "Balanced accuracy": balanced_accuracy_score(y_test, best_predictions),
        "Precision": precision_score(y_test, best_predictions, zero_division=0),
        "Recall": recall_score(y_test, best_predictions, zero_division=0),
        "F1 score": f1_score(y_test, best_predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, best_probabilities),
    }
    matrix = confusion_matrix(y_test, best_predictions, labels=[0, 1])
    confusion = pd.DataFrame(
        matrix,
        index=["Actual: no diabetes", "Actual: diabetes"],
        columns=["Predicted: no diabetes", "Predicted: diabetes"],
    )
    return data, best_combination, best_model, metrics, comparison, confusion, len(y_test)


st.title("Pima Diabetes Classification")
st.caption(
    "An educational machine-learning demo trained on the Pima Indians Diabetes dataset. "
    "It is not a medical diagnostic tool."
)

try:
    data, best_name, model, metrics, comparison, confusion, test_size = train_and_evaluate()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Could not load the training data: {error}")
    st.stop()

with st.container(horizontal=True):
    st.metric("Best combination", best_name, border=True)
    st.metric("Holdout accuracy", f"{metrics['Accuracy']:.1%}", border=True)
    st.metric("Holdout F1 score", f"{metrics['F1 score']:.3f}", border=True)

st.caption(
    f"Evaluation uses a stratified 80/20 split ({test_size} test records). "
    "The combination is selected by five-fold training CV F1, with balanced accuracy as a tie-breaker."
)

predict_tab, performance_tab, data_tab = st.tabs(
    ["Try a prediction", "Model performance", "Dataset"]
)

with predict_tab:
    st.subheader("Enter the measurements")
    st.info(
        "For glucose, blood pressure, skin-fold thickness, insulin, and BMI, enter 0 "
        "when the measurement is unavailable; the model treats those values as missing."
    )
    with st.form("patient_measurements"):
        columns = st.columns(2)
        values = {}
        for index, feature in enumerate(FEATURES):
            column = columns[index % 2]
            source = data[feature].replace(0, pd.NA) if feature in ZERO_AS_MISSING else data[feature]
            source = pd.to_numeric(source, errors="coerce").dropna()
            default = float(source.median())
            minimum = 0.0 if feature in ZERO_AS_MISSING else float(data[feature].min())
            maximum = float(data[feature].max())
            integer = feature in {"preg_count", "age"}
            with column:
                values[feature] = st.number_input(
                    FEATURE_LABELS[feature],
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(round(default)) if integer else default,
                    step=1.0 if integer else 0.1,
                    key=f"input_{feature}",
                )
        submitted = st.form_submit_button("Estimate class", type="primary")

    if submitted:
        patient = pd.DataFrame([values], columns=FEATURES).replace(
            {feature: {0: float("nan")} for feature in ZERO_AS_MISSING}
        )
        predicted_class = int(model.predict(patient)[0])
        probability = float(model.predict_proba(patient)[0, 1])
        st.subheader("Model result")
        st.metric("Estimated probability for the positive class", f"{probability:.1%}")
        st.progress(probability)
        if predicted_class == 1:
            st.warning("The model predicts the positive diabetes class for these inputs.")
        else:
            st.success("The model predicts the negative diabetes class for these inputs.")
        st.caption(
            "This estimate reflects patterns in a small historical dataset. It cannot "
            "confirm or rule out diabetes; consult a qualified healthcare professional."
        )

with performance_tab:
    st.subheader("Held-out test results")
    metric_columns = st.columns(3)
    for index, metric_name in enumerate(
        ["Balanced accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"]
    ):
        metric_columns[index % 3].metric(metric_name, f"{metrics[metric_name]:.3f}")
    st.caption(
        "Only the selected model's results on the untouched test split are shown here. "
        "Cross-validation scores below are for model selection and should not be read as test scores."
    )
    st.subheader("Classifier and resampling comparison")
    st.caption(
        "Each row is one combination. Imputation, scaling, and resampling are fit "
        "inside each training fold; the held-out test split is never resampled. "
        "Rows are ranked by training CV F1, then balanced accuracy."
    )
    st.dataframe(
        comparison.style.format(
            {
                "CV F1": "{:.3f}",
                "CV balanced accuracy": "{:.3f}",
                "CV accuracy": "{:.1%}",
                "Test accuracy": "{:.1%}",
                "Test balanced accuracy": "{:.3f}",
                "Test precision": "{:.3f}",
                "Test recall": "{:.3f}",
                "Test F1": "{:.3f}",
                "Test ROC-AUC": "{:.3f}",
            }
        ),
        hide_index=True,
    )
    st.subheader("Confusion matrix: held-out test split")
    st.dataframe(confusion, hide_index=False)

with data_tab:
    positive_rate = float(data[TARGET].mean())
    left, right = st.columns(2)
    left.metric("Dataset records", f"{len(data):,}")
    right.metric("Positive class in dataset", f"{positive_rate:.1%}")
    st.caption(
        "Zero values in glucose, blood pressure, skin-fold thickness, insulin, and BMI "
        "are treated as missing and imputed within each training fold."
    )
    st.bar_chart(
        data[TARGET]
        .value_counts()
        .sort_index()
        .rename(index={0: "No diabetes", 1: "Diabetes"})
        .rename("Records")
    )
    st.dataframe(data.head(10), hide_index=True)
