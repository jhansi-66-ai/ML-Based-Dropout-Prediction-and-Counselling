from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from django.shortcuts import render
from django.conf import settings
import pandas as pd
import seaborn as sns
import joblib
import pymysql
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def get_connection():
    """
    Create and return a connection to MySQL database
    """
    return pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        password='root',
        database='dropout',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# ==============================
# BASIC PAGES
# ==============================
def index(request):
    return render(request, 'index.html')

def Signup(request):
    """
    User signup view
    """
    if request.method == 'POST':
        username = request.POST.get('t1')
        password = request.POST.get('t2')
        contact = request.POST.get('t3')
        email = request.POST.get('t4')
        address = request.POST.get('t5')

        con = get_connection()
        cur = con.cursor()

        # Check if username exists
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cur.fetchone()

        if user:
            return render(request, 'signup.html', {'msg': "⚠️ Username already exists"})
        else:
            # Insert new user
            cur.execute("""
                INSERT INTO users (username, password, contact_no, email, address, role)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (username, password, contact, email, address, 'user'))
            con.commit()
            con.close()
            return render(request, 'login.html', {'msg': "Signup successful! Login now."})

    return render(request, 'signup.html')

def Login(request):
    """
    User login view
    """
    if request.method == 'POST':
        username = request.POST.get('t1')
        password = request.POST.get('t2')

        con = get_connection()
        cur = con.cursor(pymysql.cursors.DictCursor)

        # Authenticate user
        cur.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (username, password)
        )
        data = cur.fetchone()
        con.close()

        if data:
            return render(request, 'user_home.html', {'user': username})
        else:
            return render(request, 'login.html', {'msg': "Invalid username or password"})

    return render(request, 'login.html')


# ==============================
# UPLOAD DATASET
# ==============================
def UploadDataset(request):
    """
    Upload dataset view
    """
    context = {}

    if request.method == 'POST' and request.FILES.get('dataset'):
        uploaded_file = request.FILES['dataset']

        dataset_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
        os.makedirs(dataset_dir, exist_ok=True)

        dataset_path = os.path.join(dataset_dir, 'uploaded_dataset.csv')
        with open(dataset_path, 'wb+') as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)

        try:
            df = pd.read_csv(dataset_path)
            context['msg'] = "Dataset uploaded successfully!"
            context['filename'] = uploaded_file.name
            context['rows'] = df.sample(min(10, len(df))).to_html(
                classes="table table-striped",
                index=False
            )
        except Exception as e:
            context['msg'] = f"Error reading CSV: {e}"

    return render(request, 'UploadDataset.html', context)


def PreprocessDataset(request):
    context = {}

    try:
        # Path of uploaded dataset
        dataset_path = os.path.join(
            settings.MEDIA_ROOT,
            'datasets',
            'uploaded_dataset.csv'
        )

        if not os.path.exists(dataset_path):
            context['msg'] = "Dataset not found. Please upload dataset first."
            return render(request, 'PreprocessDataset.html', context)

        # Load dataset
        df = pd.read_csv(dataset_path)

        # -----------------------------
        # 1. Remove duplicate rows
        # -----------------------------
        df.drop_duplicates(inplace=True)

        # -----------------------------
        # 2. Handle Missing Values
        # -----------------------------
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)

        # -----------------------------
        # 3. Encode Categorical Columns
        # -----------------------------
        le = LabelEncoder()
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])

        # -----------------------------
        # 4. Feature Scaling
        # -----------------------------
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # -----------------------------
        # 5. Save Preprocessed Dataset
        # -----------------------------
        preprocess_dir = os.path.join(settings.MEDIA_ROOT, 'preprocessed')
        os.makedirs(preprocess_dir, exist_ok=True)

        preprocess_path = os.path.join(
            preprocess_dir,
            'preprocessed_dataset.csv'
        )

        df.to_csv(preprocess_path, index=False)

        # -----------------------------
        # ✅ ADD THIS PART
        # -----------------------------
        context['row_count'] = df.shape[0]
        context['column_count'] = df.shape[1]

        # -----------------------------
        # 6. Send Preview to UI
        # -----------------------------
        context['msg'] = "Dataset preprocessed successfully!"
        context['rows'] = df.head(10).to_html(
            classes="table table-bordered",
            index=False
        )

    except Exception as e:
        context['msg'] = f"Preprocessing Error: {e}"

    return render(request, 'PreprocessDataset.html', context)


def FeatureExtraction(request):
    context = {}

    try:
        preprocess_path = os.path.join(
            settings.MEDIA_ROOT,
            'preprocessed',
            'preprocessed_dataset.csv'
        )

        if not os.path.exists(preprocess_path):
            context['msg'] = "Preprocessed dataset not found. Please preprocess first."
            return render(request, 'FeatureExtraction.html', context)

        # -----------------------------
        # Load dataset
        # -----------------------------
        df = pd.read_csv(preprocess_path)

        # -----------------------------
        # Separate features & target
        # -----------------------------
        X = df.drop('target', axis=1)
        y = df['target']

        # -----------------------------
        # Mutual Information
        # -----------------------------
        mi_scores = mutual_info_classif(X, y, random_state=42)

        mi_df = pd.DataFrame({
            'Feature': X.columns,
            'MI Score': mi_scores
        }).sort_values(by='MI Score', ascending=False)

        # -----------------------------
        # Select Top 15 Features
        # -----------------------------
        top_features = mi_df.head(15)

        df_selected = df[top_features['Feature'].tolist() + ['target']]

        # -----------------------------
        # Save Selected Dataset
        # -----------------------------
        feature_dir = os.path.join(settings.MEDIA_ROOT, 'features')
        os.makedirs(feature_dir, exist_ok=True)

        feature_path = os.path.join(
            feature_dir,
            'selected_features_dataset.csv'
        )
        df_selected.to_csv(feature_path, index=False)

        # =====================================================
        # ✅ ADDITION: SAVE FEATURE NAMES FOR PREDICTION
        # =====================================================
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(
            top_features['Feature'].tolist(),
            os.path.join(model_dir, 'lr_features.pkl')
        )

        # -----------------------------
        # FEATURE IMPORTANCE GRAPH
        # -----------------------------
        plt.figure(figsize=(10, 6))
        plt.barh(
            top_features['Feature'],
            top_features['MI Score']
        )
        plt.xlabel("Mutual Information Score")
        plt.ylabel("Features")
        plt.title("Top 15 Feature Importance (Mutual Information)")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        graph_path = os.path.join(feature_dir, 'feature_importance.png')
        plt.savefig(graph_path, dpi=300)
        plt.close()

        # -----------------------------
        # Send Data to UI
        # -----------------------------
        context['msg'] = "Feature extraction completed successfully!"
        context['total_features'] = X.shape[1]
        context['selected_features'] = len(top_features)
        context['feature_table'] = top_features.to_html(
            classes="table table-bordered",
            index=False
        )
        context['graph_url'] = settings.MEDIA_URL + 'features/feature_importance.png'

    except Exception as e:
        context['msg'] = f"Feature Extraction Error: {e}"

    return render(request, 'FeatureExtraction.html', context)


def TrainLogisticRegression(request):
    context = {}

    try:
        feature_path = os.path.join(
            settings.MEDIA_ROOT,
            'features',
            'selected_features_dataset.csv'
        )

        if not os.path.exists(feature_path):
            context['msg'] = "Feature dataset not found. Please perform feature extraction first."
            return render(request, 'TrainLogistic.html', context)

        df = pd.read_csv(feature_path)

        # -----------------------------
        # Separate Features & Target
        # -----------------------------
        X = df.drop('target', axis=1)
        y = df['target']

        # ==================================================
        # ✅ SAVE FEATURE NAMES (FOR PREDICTION)
        # ==================================================
        feature_save_dir = os.path.join(settings.MEDIA_ROOT, 'features')
        os.makedirs(feature_save_dir, exist_ok=True)

        joblib.dump(
            list(X.columns),
            os.path.join(feature_save_dir, 'lr_features.pkl')
        )

        # -----------------------------
        # Train-Test Split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # ==================================================
        # Logistic Regression WITHOUT SMOTE
        # ==================================================
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        acc_no = accuracy_score(y_test, y_pred)
        prec_no = precision_score(y_test, y_pred, average='weighted')
        rec_no = recall_score(y_test, y_pred, average='weighted')
        f1_no = f1_score(y_test, y_pred, average='weighted')
        cm_no = confusion_matrix(y_test, y_pred)

        joblib.dump(lr, os.path.join(model_dir, 'lr_no_smote.pkl'))

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_no, annot=True, fmt="d", cmap="Blues")
        plt.title("LR Without SMOTE")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(
            os.path.join(model_dir, 'lr_no_smote.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        # ==================================================
        # Logistic Regression WITH SMOTE
        # ==================================================
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)

        lr_sm = LogisticRegression(max_iter=1000)
        lr_sm.fit(X_smote, y_smote)
        y_pred_sm = lr_sm.predict(X_test)

        acc_sm = accuracy_score(y_test, y_pred_sm)
        prec_sm = precision_score(y_test, y_pred_sm, average='weighted')
        rec_sm = recall_score(y_test, y_pred_sm, average='weighted')
        f1_sm = f1_score(y_test, y_pred_sm, average='weighted')
        cm_sm = confusion_matrix(y_test, y_pred_sm)

        joblib.dump(lr_sm, os.path.join(model_dir, 'lr_smote.pkl'))

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_sm, annot=True, fmt="d", cmap="Greens")
        plt.title("LR With SMOTE")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(
            os.path.join(model_dir, 'lr_smote.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        # ==================================================
        # SEND RESULTS TO UI
        # ==================================================
        context = {
            'msg': "Logistic Regression Training Completed Successfully!",

            'acc_no': round(acc_no * 100, 2),
            'prec_no': round(prec_no * 100, 2),
            'rec_no': round(rec_no * 100, 2),
            'f1_no': round(f1_no * 100, 2),

            'acc_sm': round(acc_sm * 100, 2),
            'prec_sm': round(prec_sm * 100, 2),
            'rec_sm': round(rec_sm * 100, 2),
            'f1_sm': round(f1_sm * 100, 2),

            'cm_no': settings.MEDIA_URL + 'models/lr_no_smote.png',
            'cm_sm': settings.MEDIA_URL + 'models/lr_smote.png',
        }

    except Exception as e:
        context['msg'] = f"Training Error: {e}"

    return render(request, 'TrainLogistic.html', context)


def TrainXGBoost(request):
    context = {}

    try:
        feature_path = os.path.join(
            settings.MEDIA_ROOT,
            'features',
            'selected_features_dataset.csv'
        )

        if not os.path.exists(feature_path):
            context['msg'] = "Feature dataset not found. Please perform feature extraction first."
            return render(request, 'TrainXGBoost.html', context)

        df = pd.read_csv(feature_path)

        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # ==================================================
        # XGBOOST WITHOUT SMOTE
        # ==================================================
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='multi:softmax',
            eval_metric='mlogloss',
            random_state=42
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)

        acc_no = accuracy_score(y_test, y_pred)
        prec_no = precision_score(y_test, y_pred, average='weighted')
        rec_no = recall_score(y_test, y_pred, average='weighted')
        f1_no = f1_score(y_test, y_pred, average='weighted')
        cm_no = confusion_matrix(y_test, y_pred)

        joblib.dump(xgb, os.path.join(model_dir, 'xgb_no_smote.pkl'))

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_no, annot=True, fmt="d", cmap="Purples")
        plt.title("XGBoost Without SMOTE")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(model_dir, 'xgb_no_smote.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ==================================================
        # XGBOOST WITH SMOTE
        # ==================================================
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)

        xgb_sm = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='multi:softmax',
            eval_metric='mlogloss',
            random_state=42
        )
        xgb_sm.fit(X_smote, y_smote)
        y_pred_sm = xgb_sm.predict(X_test)

        acc_sm = accuracy_score(y_test, y_pred_sm)
        prec_sm = precision_score(y_test, y_pred_sm, average='weighted')
        rec_sm = recall_score(y_test, y_pred_sm, average='weighted')
        f1_sm = f1_score(y_test, y_pred_sm, average='weighted')
        cm_sm = confusion_matrix(y_test, y_pred_sm)

        joblib.dump(xgb_sm, os.path.join(model_dir, 'xgb_smote.pkl'))

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_sm, annot=True, fmt="d", cmap="Greens")
        plt.title("XGBoost With SMOTE")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(model_dir, 'xgb_smote.png'), dpi=300, bbox_inches='tight')
        plt.close()

        context = {
            'msg': "XGBoost Training Completed Successfully!",

            'acc_no': round(acc_no * 100, 2),
            'prec_no': round(prec_no * 100, 2),
            'rec_no': round(rec_no * 100, 2),
            'f1_no': round(f1_no * 100, 2),

            'acc_sm': round(acc_sm * 100, 2),
            'prec_sm': round(prec_sm * 100, 2),
            'rec_sm': round(rec_sm * 100, 2),
            'f1_sm': round(f1_sm * 100, 2),

            'cm_no': settings.MEDIA_URL + 'models/xgb_no_smote.png',
            'cm_sm': settings.MEDIA_URL + 'models/xgb_smote.png',
        }

    except Exception as e:
        context['msg'] = f"Training Error: {e}"

    return render(request, 'TrainXGBoost.html', context)



def ModelComparisonGraph(request):
    context = {}

    try:
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Load dataset
        feature_path = os.path.join(settings.MEDIA_ROOT, 'features', 'selected_features_dataset.csv')
        if not os.path.exists(feature_path):
            context['msg'] = "Feature dataset not found. Please perform feature extraction first."
            return render(request, 'ModelComparisonGraph.html', context)

        df = pd.read_csv(feature_path)
        X = df.drop('target', axis=1)
        y = df['target']

        # Train-test split (same as training)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Apply SMOTE for SMOTE models
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)

        # Models to load
        model_files = {
            'LR_NoSMOTE': 'lr_no_smote.pkl',
            'LR_SMOTE': 'lr_smote.pkl',
            'XGB_NoSMOTE': 'xgb_no_smote.pkl',
            'XGB_SMOTE': 'xgb_smote.pkl'
        }

        metrics = {}
        for name, file in model_files.items():
            model_path = os.path.join(model_dir, file)
            if not os.path.exists(model_path):
                context['msg'] = f"Model {file} not found. Train all models first."
                return render(request, 'ModelComparisonGraph.html', context)

            model = joblib.load(model_path)
            # Predict on original test set
            if 'SMOTE' in name:
                y_pred = model.predict(X_test)  # Always evaluate on original test set
            else:
                y_pred = model.predict(X_test)

            metrics[name] = {
                'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
                'Precision': round(precision_score(y_test, y_pred, average='weighted') * 100, 2),
                'Recall': round(recall_score(y_test, y_pred, average='weighted') * 100, 2),
                'F1': round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
            }

        # Plotting
        models = ['LR_NoSMOTE', 'LR_SMOTE', 'XGB_NoSMOTE', 'XGB_SMOTE']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1']:
            values = [metrics[m][metric_name] for m in models]

            plt.figure(figsize=(8,5))
            sns.barplot(x=models, y=values, palette=colors)
            plt.title(f"{metric_name} Comparison")
            plt.ylabel(f"{metric_name} (%)")
            plt.ylim(0,100)
            for i, v in enumerate(values):
                plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f'comparison_{metric_name.lower()}.png'), dpi=300)
            plt.close()

        # Pass image URLs to template
        context['acc_img'] = '/media/models/comparison_accuracy.png'
        context['prec_img'] = '/media/models/comparison_precision.png'
        context['recall_img'] = '/media/models/comparison_recall.png'
        context['f1_img'] = '/media/models/comparison_f1.png'
        context['msg'] = "Model comparison graphs generated successfully!"

    except Exception as e:
        context['msg'] = f"Error generating comparison graphs: {e}"

    return render(request, 'ModelComparisonGraph.html', context)

def PredictStudent(request):
    import os
    import pandas as pd
    import joblib
    from django.shortcuts import render
    from django.conf import settings

    context = {}

    try:
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        feature_path = os.path.join(settings.MEDIA_ROOT, 'features', 'lr_features.pkl')
        lr_model_path = os.path.join(model_dir, 'lr_no_smote.pkl')
        xgb_model_path = os.path.join(model_dir, 'xgb_no_smote.pkl')

        # ----------------- Load feature list -----------------
        if not os.path.exists(feature_path):
            context['msg'] = "Feature information missing. Train model first."
            return render(request, 'Predict.html', context)

        feature_columns = joblib.load(feature_path)

        # ----------------- Example UI guidance -----------------
        example_values = [
            2, 6.5, 6, 5, 6, 6.2, 6, 6, 1, 1,
            1, 1, 6.5, 1, 0
        ]
        context['feature_data'] = [
            (feature_columns[i], example_values[i])
            for i in range(len(feature_columns))
        ]

        # ----------------- POST: Prediction -----------------
        if request.method == "POST":
            model_type = request.POST.get('model_type')

            # Load the selected model
            if model_type == "lr":
                model = joblib.load(lr_model_path)
                model_name = "Logistic Regression"
            elif model_type == "xgb":
                model = joblib.load(xgb_model_path)
                model_name = "XGBoost"
            else:
                raise Exception("Invalid model selected")

            # ----------------- Map inputs -----------------
            input_values = []
            for i, feature in enumerate(feature_columns):
                val = request.POST.get(f'f{i+1}')
                if val is None or val.strip() == '':
                    raise Exception(f"Missing value for feature: {feature}")

                # --------- Categorical Mapping Example ---------
                # Map dropdowns to numeric codes exactly like your training data
                if feature == "Application mode":
                    val = 1 if val.lower() == "offline" else 2
                elif feature == "Course":
                    val = 1 if val.lower() == "engineering" else 2
                elif feature == "Tuition fees up to date":
                    val = 1 if val.lower() in ["yes", "up to date", "1"] else 0
                elif feature == "Scholarship holder":
                    val = 1 if val.lower() in ["yes", "1"] else 0
                elif feature == "Mother's occupation":
                    # Example encoding; adjust to match your training data
                    occupations = {"teacher":1, "farmer":2, "business":3, "other":4}
                    val = occupations.get(val.lower(), 0)
                elif feature == "Previous qualification":
                    # Map to numeric level used during training
                    qualifications = {"highschool":1, "bachelor":2, "master":3, "other":4}
                    val = qualifications.get(val.lower(), 0)
                else:
                    val = float(val)  # numeric input

                input_values.append(val)

            # ----------------- Create DataFrame -----------------
            input_df = pd.DataFrame([input_values], columns=feature_columns)

            # ----------------- Predict probability -----------------
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]
            else:
                prob = model.predict(input_df)[0]

            risk_percent = round(prob * 100, 2)

            # ----------------- Risk Level & Advice -----------------
            if risk_percent >= 25:
                risk_level = "🔴 High Risk"
                prediction_msg = "⚠️ Student is at High Risk of Dropout"
                counselling = ("Immediate academic counselling, financial review, "
                               "and close faculty monitoring are required.")
            elif risk_percent >= 15:
                risk_level = "🟡 Moderate Risk"
                prediction_msg = "⚠️ Student Needs Attention"
                counselling = ("Periodic academic monitoring and counselling sessions "
                               "are recommended.")
            else:
                risk_level = "🟢 Low Risk"
                prediction_msg = "✅ Student is Likely to Continue"
                counselling = ("Maintain current academic engagement and performance.")

            # ----------------- Update Context -----------------
            context.update({
                "model_used": model_name,
                "probability": risk_percent,
                "risk_level": risk_level,
                "prediction": prediction_msg,
                "counselling": counselling
            })

    except Exception as e:
        context['msg'] = f"Prediction Error: {e}"

    return render(request, 'Predict.html', context)


def BulkPredict(request):
    import os
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    from django.core.files.storage import FileSystemStorage
    from django.shortcuts import render
    from django.conf import settings

    context = {}

    try:
        # ================= PATHS =================
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        feature_path = os.path.join(settings.MEDIA_ROOT, 'features', 'lr_features.pkl')

        if not os.path.exists(feature_path):
            context['msg'] = "Feature information missing. Train model first."
            return render(request, 'BulkPredict.html', context)

        model_features = joblib.load(feature_path)

        # ================= COLUMN MAPPING =================
        column_mapping = {
            "Subjects Approved (Semester 2)": "Curricular units 2nd sem (approved)",
            "Average Grade (Semester 2)": "Curricular units 2nd sem (grade)",
            "Subjects Enrolled (Semester 2)": "Curricular units 2nd sem (enrolled)",
            "Evaluation Attempts (Semester 2)": "Curricular units 2nd sem (evaluations)",

            "Subjects Approved (Semester 1)": "Curricular units 1st sem (approved)",
            "Average Grade (Semester 1)": "Curricular units 1st sem (grade)",
            "Subjects Enrolled (Semester 1)": "Curricular units 1st sem (enrolled)",
            "Evaluation Attempts (Semester 1)": "Curricular units 1st sem (evaluations)",

            "Admission Mode": "Application mode",
            "Tuition Fees Status": "Tuition fees up to date",
            "Scholarship Status": "Scholarship holder",
            "Course Type": "Course",
            "Age at Enrollment": "Age at enrollment",
            "Highest Previous Qualification": "Previous qualification",
            "Mother's Occupation": "Mother's occupation"
        }

        if request.method == "POST":
            csv_file = request.FILES.get("csv_file")
            model_type = request.POST.get("model_type", "lr")

            # ================= VALIDATE CSV =================
            if not csv_file or not csv_file.name.endswith(".csv"):
                context['msg'] = "Upload a valid CSV file (.csv only)."
                return render(request, 'BulkPredict.html', context)

            fs = FileSystemStorage()
            filename = fs.save(csv_file.name, csv_file)
            path = fs.path(filename)

            # ================= READ CSV =================
            df_original = pd.read_csv(path)
            df_original.columns = df_original.columns.str.strip()
            df_original.reset_index(drop=True, inplace=True)

            # ================= RENAME COLUMNS =================
            df_model = df_original.rename(columns=column_mapping)

            # ================= CHECK REQUIRED FEATURES =================
            missing = set(model_features) - set(df_model.columns)
            if missing:
                context['msg'] = f"Missing columns in CSV: {', '.join(missing)}"
                return render(request, 'BulkPredict.html', context)

            # ================= CONVERT TO NUMERIC =================
            X = df_model[model_features].apply(pd.to_numeric, errors='coerce')
            if X.isnull().any().any():
                context['msg'] = "CSV contains invalid or missing numeric values."
                return render(request, 'BulkPredict.html', context)

            # ================= LOAD MODEL =================
            if model_type == "xgb":
                model_file = os.path.join(model_dir, 'xgb_no_smote.pkl')
                model_name = "XGBoost"
            else:
                model_file = os.path.join(model_dir, 'lr_no_smote.pkl')
                model_name = "Logistic Regression"

            if not os.path.exists(model_file):
                context['msg'] = f"Model file {model_file} not found."
                return render(request, 'BulkPredict.html', context)

            model = joblib.load(model_file)

            # ================= PREDICT =================
            probs = (model.predict_proba(X)[:, 1] * 100).round(2)

            # ================= BALANCED RISK LOGIC =================
            def risk_level(p):
                if p >= 18:      # High Risk
                    return "High Risk"
                elif p >= 8:    # Moderate Risk
                    return "Moderate Risk"
                else:            # Low Risk
                    return "Low Risk"

            df_original['Dropout Probability (%)'] = probs
            df_original['Risk Level'] = df_original['Dropout Probability (%)'].apply(risk_level)

            # ================= COUNSELLING ADVICE =================
            counselling_advice = {
                "High Risk": (
                    "🔴 High Risk: Immediate intervention recommended. "
                    "Schedule one-on-one academic counselling, review financial or personal support needs, "
                    "and assign close faculty mentorship to prevent dropout."
                ),
                "Moderate Risk": (
                    "🟡 Moderate Risk: Periodic monitoring suggested. "
                    "Arrange regular check-ins with academic advisors and provide guidance on time management "
                    "and study strategies."
                ),
                "Low Risk": (
                    "🟢 Low Risk: Continue current academic engagement. "
                    "Encourage student to maintain consistent performance and participation in courses."
                )
            }

            df_original['Counselling Advice'] = df_original['Risk Level'].map(counselling_advice)

            # ================= SUMMARY =================
            total = len(df_original)
            high = (df_original['Risk Level'] == "High Risk").sum()
            moderate = (df_original['Risk Level'] == "Moderate Risk").sum()
            low = (df_original['Risk Level'] == "Low Risk").sum()

            # ================= GRAPH =================
            plt.figure(figsize=(6, 4))
            plt.bar(
                ["High Risk", "Moderate Risk", "Low Risk"],
                [high, moderate, low],
                color=['#FF4C4C', '#FFC300', '#4CAF50']
            )
            plt.title("Student Dropout Risk Distribution")
            plt.xlabel("Risk Level")
            plt.ylabel("Number of Students")
            graph_name = f"bulk_risk_graph_{model_type}.png"
            plt.savefig(os.path.join(settings.MEDIA_ROOT, graph_name), dpi=300, bbox_inches='tight')
            plt.close()

            # ================= CONTEXT =================
            context.update({
                "model_used": model_name,
                "total": total,
                "high": high,
                "moderate": moderate,
                "low": low,
                "bulk_table": df_original.to_html(
                    classes="table table-bordered table-striped",
                    index=False,
                    escape=False
                ),
                "graph": settings.MEDIA_URL + graph_name
            })

    except Exception as e:
        context['msg'] = f"Prediction Error: {str(e)}"

    return render(request, 'BulkPredict.html', context)










