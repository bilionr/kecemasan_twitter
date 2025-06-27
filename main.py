import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import logging
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
nltk.download('punkt')

# Initialize Sastrawi stopwords
stop_factory = StopWordRemoverFactory()
stop_words = set(stop_factory.get_stop_words()) | {
    'yang', 'dan', 'atau', 'di', 'ke', 'dari', 'untuk', 'pada', 'dengan', 'oleh', 'dalam', 'tentang',
    'ini', 'itu', 'tersebut', 'sebuah', 'adalah', 'akan', 'sudah', 'belum', 'masih', 'lagi', 'saja',
    'hanya', 'bisa', 'harus', 'boleh', 'juga', 'kembali', 'sekali', 'kalau', 'kalo', 'apa', 'siapa', 'kenapa',
    'bagaimana', 'kapan', 'dimana', 'mengapa', 'sih', 'deh', 'dong', 'lah', 'kok', 'ya', 'yaa', 'yah', 'gak', 'ga',
    'nggak', 'tidak', 'tak', 'enggak', 'bukan', 'gk', 'gitu', 'aja', 'nih', 'tu', 'tuh', 'kayak', 'kyk', 'soalnya',
    'gw', 'gue', 'gua', 'lu', 'lo', 'loe', 'aku', 'saya', 'kamu', 'kau', 'dia', 'ie', 'lg', 'lgi', 'sedang', 'sdh',
    'udh', 'udah', 'sm', 'sama', 'bs', 'bgt', 'banget', 'tp', 'tapi', 'ttg', 'krn', 'karena', 'dr', 'dgn', 'utk',
    'blm', 'jgn', 'jangan', 'dlu', 'duluan', 'ato', 'mau', 'mw', 'msh', 'cm', 'cuma', 'koq', 'si', 'emg', 'emang',
    'bener', 'bnr', 'btw', 'oh', 'ok', 'oke', 'jd', 'jdi', 'jadi', 'k', 'km', 'knp', 'knpa', 'smpe', 'sampe', 'dg',
    'ama', 'hrs', 'kudu', 'lgsg', 'langsung', 'bla', 'blh', 'brp', 'berapa', 'kl', 'kalo', 'klu', 'klo', 'nah', 'ni'
}

# Load InSet Lexicon from positive.csv and negative.csv
def load_inset_lexicon():
    try:
        pos_df = pd.read_csv('positive.csv')
        neg_df = pd.read_csv('negative.csv')
        inset_lexicon = {}
        for _, row in pos_df.iterrows():
            inset_lexicon[row['word']] = {'label': 'positif', 'weight': row['weight']}
        for _, row in neg_df.iterrows():
            inset_lexicon[row['word']] = {'label': 'negatif', 'weight': row['weight']}
        logging.info(f"Loaded InSet Lexicon: {len(inset_lexicon)} words")
        return inset_lexicon
    except FileNotFoundError as e:
        logging.error(f"Lexicon file not found: {e}")
        raise
    except KeyError as e:
        logging.error(f"Missing column in lexicon file: {e}")
        raise

inset_lexicon = load_inset_lexicon()

# Function to calculate InSet Lexicon sentiment features
def calculate_inset_sentiment(text):
    if not isinstance(text, str) or text.strip() == '':
        return 0, 0, 0
    tokens = word_tokenize(text.lower())
    pos_count = sum(1 for t in tokens if inset_lexicon.get(t, {}).get('label') == 'positif')
    neg_count = sum(1 for t in tokens if inset_lexicon.get(t, {}).get('label') == 'negatif')
    score = sum(inset_lexicon.get(t, {}).get('weight', 0) for t in tokens)
    text_length = len(tokens) if len(tokens) > 0 else 1
    normalized_score = score / text_length
    return pos_count, neg_count, normalized_score

# Function to load training data
def load_training_data(file_path='final_training_dataset.csv'):
    logging.info("Loading training data...")
    df = pd.read_csv(file_path)
    df = df[['full_text', 'created_at', 'user_id_str', 'username', 'label']].copy()
    df = df.dropna(subset=['full_text', 'label'])
    df = df[df['full_text'].apply(lambda x: isinstance(x, str))]
    df = df.groupby(['username', 'user_id_str']).agg({
        'full_text': list,
        'created_at': list,
        'label': lambda x: x.mode()[0]
    }).reset_index()
    df = df.explode(['full_text', 'created_at']).reset_index(drop=True)
    logging.info(f"Loaded {len(df)} training rows after ensuring one label per username")
    logging.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df

# Function to load testing data
def load_testing_data(file_path='final_testing_dataset.csv'):
    logging.info("Loading testing data...")
    df = pd.read_csv(file_path)
    df = df[['full_text', 'created_at', 'user_id_str', 'username']].copy()
    df = df.dropna(subset=['full_text'])
    df = df[df['full_text'].apply(lambda x: isinstance(x, str))]
    logging.info(f"Loaded {len(df)} testing rows")
    return df

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'#\w+|@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    if not tokens:
        return ''
    return ' '.join(tokens)

# Extract behavioral and sentiment features
def extract_features(df):
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', format='%a %b %d %H:%M:%S %z %Y')
    df['cleaned_text'] = df['full_text'].apply(preprocess_text)
    df = df[df['cleaned_text'].str.strip() != '']
    df['TextLength'] = df['cleaned_text'].apply(lambda x: len(x.split()) if x else 0)
    df['HourOfDay'] = df['created_at'].dt.hour.fillna(0)
    df['DayOfWeek'] = df['created_at'].dt.dayofweek.fillna(0)
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df[['inset_pos', 'inset_neg', 'inset_score']] = df['cleaned_text'].apply(
        lambda x: pd.Series(calculate_inset_sentiment(x))
    )
    return df

# Latent Profile Analysis with descriptive profile names
def perform_lpa(df, features, max_profiles=6):
    scaler = StandardScaler()
    X_lpa = scaler.fit_transform(df[features])
    aic, bic, entropy, silhouette = [], [], [], []
    models = {}

    for n in range(2, max_profiles + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(X_lpa)
        aic.append(gmm.aic(X_lpa))
        bic.append(gmm.bic(X_lpa))
        labels = gmm.predict(X_lpa)
        probas = gmm.predict_proba(X_lpa)
        entropy_val = -np.sum(probas * np.log(probas + 1e-10)) / len(X_lpa)
        entropy.append(entropy_val)
        if len(set(labels)) > 1:
            silhouette.append(silhouette_score(X_lpa, labels))
        else:
            silhouette.append(0)
        models[n] = gmm

    optimal_n = 4
    logging.info(f"Optimal number of profiles set to: {optimal_n}")

    final_gmm = models[optimal_n]
    df['LatentProfile'] = final_gmm.predict(X_lpa)

    profile_means = df.groupby('LatentProfile')[features].mean()
    profile_means_scaled = scaler.transform(profile_means)
    profile_means_df = pd.DataFrame(profile_means_scaled, columns=features, index=profile_means.index)

    # Blok logika penamaan Anda tetap dipertahankan
    profile_labels = {}
    unassigned_indices = list(profile_means_df.index)
    sd_idx = profile_means_df['TextLength'].idxmax()
    profile_labels[sd_idx] = 'Self-Distancing'
    unassigned_indices.remove(sd_idx)
    ls_idx = profile_means_df.loc[unassigned_indices, 'inset_score'].idxmin()
    profile_labels[ls_idx] = 'Low Sentiment'
    unassigned_indices.remove(ls_idx)
    hs_idx = profile_means_df.loc[unassigned_indices, 'inset_score'].idxmax()
    profile_labels[hs_idx] = 'High Sentiment'
    unassigned_indices.remove(hs_idx)
    if len(unassigned_indices) == 1:
        neutral_idx = unassigned_indices[0]
        profile_labels[neutral_idx] = 'Neutral'

    df['ProfileName'] = df['LatentProfile'].map(profile_labels)

    print("\nProfile Summary (Standardized Mean Scores):")
    profile_summary = profile_means_df.copy()
    profile_summary.index = [profile_labels.get(i, f"Unnamed_{i}") for i in profile_summary.index]
    print(tabulate(profile_summary, headers='keys', tablefmt='pretty', floatfmt='.6f'))

    print("\nProfile Indicators:")
    print("- High Sentiment: Karakteristik skor sentimen positif tertinggi.")
    print("- Low Sentiment: Karakteristik skor sentimen negatif tertinggi.")
    print("- Self-Distancing: Karakteristik panjang teks tertinggi.")
    print("- Neutral: Tidak menunjukkan nilai ekstrim pada indikator utama.")

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Model Fit (Tidak diubah)
    ax1.plot(range(2, max_profiles + 1), aic, label='AIC', marker='o')
    ax1.plot(range(2, max_profiles + 1), bic, label='BIC', marker='o')
    ax1.set_xlabel('Number of Profiles')
    ax1.set_ylabel('AIC / BIC')
    ax1.legend(loc='upper left')
    ax3 = ax1.twinx()
    ax3.plot(range(2, max_profiles + 1), entropy, label='Entropy', color='green', marker='o')
    ax3.set_ylabel('Entropy')
    ax3.legend(loc='upper right')
    ax1.set_title('Latent Profile Analysis: Model Fit')

    # ==============================================================================
    # <<< BLOK VISUALISASI PLOT KEDUA (ax2) YANG DIREVISI >>>
    # ==============================================================================
    
    # Menentukan urutan plot yang logis agar legenda lebih rapi
    desired_order = ['Self-Distancing', 'Low Sentiment', 'High Sentiment', 'Neutral']

    # Melakukan iterasi sesuai urutan yang telah ditentukan
    for profile_name in desired_order:
        # Mencari indeks asli (0,1,2,3) yang sesuai dengan nama profil saat ini
        # Ini untuk memastikan plot tidak error jika ada nama profil yang tidak ditemukan
        original_idx = next((idx for idx, name in profile_labels.items() if name == profile_name), None)
        
        # Hanya menggambar plot jika profil dengan nama tersebut memang ada
        if original_idx is not None:
             ax2.plot(features, profile_means_df.loc[original_idx], label=profile_name, marker='o')

    ax2.set_xlabel('Indicators')
    ax2.set_ylabel('Standardized Mean Score')
    ax2.legend()
    ax2.set_title('Profile Characteristics')
    # Mengubah rotasi label sumbu-x agar tidak tumpang tindih
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    # ==============================================================================
    # <<< AKHIR DARI BLOK YANG DIREVISI >>>
    # ==============================================================================

    plt.tight_layout()
    plt.savefig('lpa_results.png')
    plt.close()
    logging.info("LPA results plot saved to 'lpa_results.png'")

    return df, optimal_n

# Visualize model AUC
def plot_model_auc(models, X_val, y_val, title="Model ROC Curves"):
    plt.figure(figsize=(8, 5))
    for name, model in models.items():
        model = model.best_estimator_ if isinstance(model, GridSearchCV) else model
        if name == 'Naive Bayes':
            y_scores = model.predict_proba(X_val[:, :-7])[:, 1]
        else:
            y_scores = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('model_roc_curves.png')
    plt.close()
    logging.info("Model ROC curves plot saved to 'model_roc_curves.png'")

# Visualize model accuracy comparison
def plot_model_accuracy_comparison(accuracies, model_names, title="Model Accuracy Comparison"):
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Model')
    plt.ylabel('Validation Accuracy')
    plt.title(title)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png')
    plt.close()
    logging.info("Model accuracy comparison plot saved to 'model_accuracy_comparison.png'")

# Visualize feature importance (sorted)
def plot_feature_importance(model, feature_names, title="Behavioral and Sentiment Feature Importance"):
    selected_features = ['TextLength', 'HourOfDay', 'DayOfWeek', 'IsWeekend', 'inset_pos', 'inset_neg', 'inset_score']
    selected_indices = [feature_names.index(f) for f in selected_features]
    importance = model.feature_importances_[selected_indices]
    sorted_idx = np.argsort(importance)[::-1]
    sorted_features = [selected_features[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]
    
    plt.figure(figsize=(8, 5))
    plt.barh(sorted_features, sorted_importance, align='center')
    plt.title(title)
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('behavioral_sentiment_feature_importance.png')
    plt.close()
    logging.info("Behavioral and sentiment feature importance plot saved to 'behavioral_sentiment_feature_importance.png'")

# Main function
def label_tweets():
    # Load and preprocess training data
    df_train = load_training_data()
    df_train = extract_features(df_train)
    logging.info(f"Training data after preprocessing: {len(df_train)} rows")

    # Load and preprocess testing data
    df_test = load_testing_data()
    initial_ids = df_test['user_id_str'].unique()
    df_test = extract_features(df_test)
    df_test_valid = df_test[df_test['cleaned_text'].str.strip() != '']
    logging.info(f"Testing data after preprocessing: {len(df_test_valid)} rows")

    # Log failed preprocessing
    failed_ids = set(df_test['user_id_str'].unique()) - set(df_test_valid['user_id_str'].unique())
    if failed_ids:
        logging.warning(f"{len(failed_ids)} accounts failed preprocessing")
        df_failed = df_test[df_test['user_id_str'].isin(failed_ids)][['user_id_str', 'full_text', 'cleaned_text']]
        df_failed.to_csv('failed_preprocessing.csv', index=False)

    if len(df_test_valid) == 0:
        logging.error("No valid testing data after preprocessing")
        raise ValueError("No valid testing data")

    # Perform LPA on training data
    lpa_features = ['TextLength', 'HourOfDay', 'DayOfWeek', 'IsWeekend', 'inset_pos', 'inset_neg', 'inset_score']
    df_train, optimal_n = perform_lpa(df_train, lpa_features)
    logging.info(f"Training data with {optimal_n} latent profiles assigned")

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(df_train['cleaned_text'])
    X_test_tfidf = vectorizer.transform(df_test_valid['cleaned_text'])
    feature_names = vectorizer.get_feature_names_out().tolist() + lpa_features

    # Scale behavioral and sentiment features
    scaler = StandardScaler()
    X_train_additional = scaler.fit_transform(df_train[lpa_features])
    X_test_additional = scaler.transform(df_test_valid[lpa_features])

    # Combine features
    X_train = hstack([X_train_tfidf, X_train_additional])
    X_test = hstack([X_test_tfidf, X_test_additional])
    y_train = df_train['label']

    # Balance training data
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    logging.info(f"Training data after undersampling: {X_train_resampled.shape[0]} rows")
    logging.info(f"Balanced label distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")

    # Split training data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.2, random_state=42, stratify=y_train_resampled
    )

    # Initialize models with hyperparameter tuning
    models = {
        'Random Forest': GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid={'n_estimators': [50, 100], 'max_depth': [10, 20, None]},
            cv=3, scoring='accuracy'
        ),
        'Naive Bayes': GridSearchCV(
            MultinomialNB(),
            param_grid={'alpha': [0.1, 0.5, 1.0, 2.0]},
            cv=3, scoring='accuracy'
        ),
        'LASSO': GridSearchCV(
            LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
            param_grid={'C': [0.01, 0.1, 1.0, 10.0]},
            cv=3, scoring='accuracy'
        )
    }

    # Train and evaluate models
    accuracies = []
    model_names = list(models.keys())
    for name, model in models.items():
        logging.info(f"Training {name}...")
        if name == 'Naive Bayes':
            model.fit(X_train_split[:, :-7], y_train_split)
            y_val_pred = model.predict(X_val_split[:, :-7])
        else:
            model.fit(X_train_split, y_train_split)
            y_val_pred = model.predict(X_val_split)
        acc = accuracy_score(y_val_split, y_val_pred)
        accuracies.append(acc)
        best_params = model.best_params_ if isinstance(model, GridSearchCV) else None
        print(f"\n{name} Validation Accuracy: {acc:.4f}")
        print(f"{name} Best Parameters: {best_params}")
        print(f"\n{name} Classification Report:")
        print(classification_report(y_val_split, y_val_pred, target_names=['Non-Anxious', 'Anxious']))

    # Plot model accuracy comparison
    plot_model_accuracy_comparison(accuracies, model_names)

    # Plot model AUC
    plot_model_auc(models, X_val_split, y_val_split)

    # Use Random Forest for feature importance and prediction
    rf_model = models['Random Forest'].best_estimator_ if isinstance(models['Random Forest'], GridSearchCV) else models['Random Forest']
    plot_feature_importance(rf_model, feature_names)

    # Predict testing data per tweet
    y_test_pred = rf_model.predict(X_test)
    df_test_valid.loc[:, 'Condition'] = y_test_pred

    # Aggregate predictions per account with threshold and minimum tweet count
    min_tweet_count = 3
    anxious_threshold = 0.7
    df_test_valid_grouped = df_test_valid.groupby(['user_id_str', 'username']).agg({
        'full_text': list,
        'cleaned_text': list,
        'created_at': list,
        'Condition': lambda x: (sum(x) / len(x) >= anxious_threshold) and len(x) >= min_tweet_count
    }).reset_index()
    df_test_valid_grouped['Condition'] = df_test_valid_grouped['Condition'].astype(int)
    logging.info(f"Aggregated predictions for {len(df_test_valid_grouped)} accounts with threshold {anxious_threshold} and min tweets {min_tweet_count}")

    # Merge back to original test data
    df_test = df_test.merge(
        df_test_valid_grouped[['user_id_str', 'username', 'Condition']],
        on=['user_id_str', 'username'], how='left'
    )
    df_test['Condition'] = df_test['Condition'].fillna(0)

    # Save results
    output_columns = ['full_text', 'user_id_str', 'username', 'created_at', 'Condition']
    df_test[output_columns].to_csv('labeled_tweets.csv', index=False)
    print("\nLabeled results saved to 'labeled_tweets.csv'")

    # Save preprocessed data
    df_test[['user_id_str', 'full_text', 'cleaned_text']].to_csv('preprocessed_tweets.csv', index=False)
    print("Preprocessed data saved to 'preprocessed_tweets.csv'")

    # Profile accounts
    account_predictions = df_test_valid_grouped.groupby(['user_id_str', 'username']).agg({
        'Condition': 'first',
        'full_text': 'count'
    }).rename(columns={'full_text': 'TweetCount'}).reset_index()
    logging.info(f"Number of unique labeled accounts: {len(account_predictions)}")

    # Handle unlabeled accounts
    unlabeled_ids = set(initial_ids) - set(account_predictions['user_id_str'])
    if unlabeled_ids:
        logging.warning(f"{len(unlabeled_ids)} accounts not labeled")
        df_unlabeled = df_test[df_test['user_id_str'].isin(unlabeled_ids)][['user_id_str', 'username', 'full_text', 'cleaned_text', 'Condition']]
        df_unlabeled.to_csv('unlabeled_accounts.csv', index=False)

    # Display profiling
    anxious_accounts = account_predictions[account_predictions['Condition'] == 1].head(5)
    non_anxious_accounts = account_predictions[account_predictions['Condition'] == 0].head(5)
    total_anxious = len(account_predictions[account_predictions['Condition'] == 1])
    total_non_anxious = len(account_predictions[account_predictions['Condition'] == 0])

    print("\nTabel Profil 5 Akun Anxious dan 5 Akun Non-Anxious:")
    combined_accounts = pd.concat([anxious_accounts, non_anxious_accounts])
    print(tabulate(combined_accounts[['user_id_str', 'username', 'Condition', 'TweetCount']],
                   headers=['UserID', 'Username', 'Condition', 'TweetCount'], tablefmt='pretty', showindex=False))
    print(f"\nTotal Akun Anxious: {total_anxious}")
    print(f"Total Akun Non-Anxious: {total_non_anxious}")

    return df_test, account_predictions

# Run
if _name_ == "_main_":
    try:
        df_labeled, account_predictions = label_tweets()
    except Exception as e:
        logging.error(f"Script failed: {e}")
        raise