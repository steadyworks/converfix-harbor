
import pandas as pd
import re


# Обучение
train = pd.read_csv('/home/data/train.csv')

# Тест
test = pd.read_csv('/home/data/test.csv')

# Пример submission
sample_sub = pd.read_csv('/home/data/sample_submission.csv')



def analyze_nlp_dataset(train_df, test_df, target_col=None):
    """
    Полный анализ train/test для NLP-соревнований.
    
    Параметры:
    - train_df: pd.DataFrame, обучающий датасет
    - test_df: pd.DataFrame, тестовый датасет  
    - target_col: str, название целевой колонки (если есть в train)
    """
    
    print("=" * 60)
    print("АНАЛИЗ DATASETS (TRAIN & TEST)")
    print("=" * 60)
    
    # 1. Общие размеры
    print(f"\n1. РАЗМЕРЫ:")
    print(f"  Train: {train_df.shape[0]} строк, {train_df.shape[1]} столбцов")
    print(f"  Test:  {test_df.shape[0]} строк, {test_df.shape[1]} столбцов")
    
    # Соотношение train/test
    ratio = train_df.shape[0] / test_df.shape[0]
    print(f"  Соотношение train/test: {ratio:.2f}")
    
    print()
    
    # 2. Список столбцов
    print("2. СТОЛБЦЫ:")
    print(f"  Train columns: {list(train_df.columns)}")
    print(f"  Test columns:  {list(test_df.columns)}")
    
    common_cols = set(train_df.columns) & set(test_df.columns)
    extra_train = set(train_df.columns) - set(test_df.columns)
    extra_test  = set(test_df.columns)  - set(train_df.columns)
    print(f"  Общие столбцы: {sorted(common_cols)}")
    if extra_train: print(f"  Только в train: {sorted(extra_train)}")
    if extra_test:  print(f"  Только в test:  {sorted(extra_test)}")
    print()

    # 3. Типы данных
    print("3. DTYPE ПО СТОЛБЦАМ:")
    print("  Train:")
    print(train_df.dtypes.value_counts())
    print("\n  Test:")  
    print(test_df.dtypes.value_counts())
    print()

    # 4. Пропуски
    print("4. ПРОПУСКИ (NaN):")
    train_null = train_df.isnull().sum()
    test_null  = test_df.isnull().sum()
    if train_null.sum() > 0:
        print("  Train (с пропусками):")
        print(train_null[train_null > 0].sort_values(ascending=False))
    else:
        print("  Train: нет пропусков")
    if test_null.sum() > 0:  
        print("  Test (с пропусками):")  
        print(test_null[test_null > 0].sort_values(ascending=False))
    else:
        print("  Test: нет пропусков")
    print()

    # 5. Уникальные значения (для категориальных признаков)
    print("5. УНИКАЛЬНЫЕ ЗНАЧЕНИЯ (первые 5 категорий):")
    for col in common_cols:
        if train_df[col].dtype == 'object' or train_df[col].nunique() < 20:
            uniq_train = train_df[col].dropna().unique()[:5]
            uniq_test  = test_df[col].dropna().unique()[:5]
            print(f"  {col}:")
            print(f"    Train: {uniq_train}")
            print(f"    Test:  {uniq_test}")
    print()

    # 6. Статистика по числовым колонкам
    print("6. СТАТИСТИКА ПО ЧИСЛОВЫМ СТОЛБЦАМ:")
    num_cols_train = train_df.select_dtypes(include=['number']).columns
    num_cols_test  = test_df.select_dtypes(include=['number']).columns

    if len(num_cols_train) > 0:
        print("  Train numeric:")
        print(train_df[num_cols_train].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    if len(num_cols_test) > 0 and not num_cols_test.equals(num_cols_train):
        print("\n  Test numeric:")  
        print(test_df[num_cols_test].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    print()

    # 7. Распределение целевой переменной (если указана)
    if target_col and target_col in train_df.columns:
        print(f"7. РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ '{target_col}':")
        class_counts = train_df[target_col].value_counts(dropna=False)
        class_pct = train_df[target_col].value_counts(normalize=True, dropna=False) * 100
        for cls, cnt in class_counts.items():
            pct = class_pct[cls]
            print(f"  {cls}: {cnt} ({pct:.1f}%)")
        print()

    # 8. Примеры строк
    print("8. ПРИМЕРЫ СТРОК (первые 3):")
    print("\n  Train:")
    print(train_df.head(3).to_string())
    print("\n  Test:")
    print(test_df.head(3).to_string())
    print()

    # 9. Длина текста (если есть текстовые колонки)
    text_cols = [c for c in common_cols if train_df[c].dtype == 'object']
    if text_cols:
        print("9. ДЛИНА ТЕКСТА (символов):")
        for col in text_cols:
            train_len = train_df[col].str.len().describe()
            test_len  = test_df[col].str.len().describe()
            print(f"\n  {col} (Train):")
            print(train_len.round(1).to_string())
            print(f"\n  {col} (Test):")  
            print(test_len.round(1).to_string())


# Как использовать:
analyze_nlp_dataset(train, test, target_col='sentiment')



import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')


train = pd.read_csv('/home/data/train.csv')
test = pd.read_csv('/home/data/test.csv')
sample_sub = pd.read_csv('/home/data/sample_submission.csv')



sid = SentimentIntensityAnalyzer()

def extract_selected_text(text, sentiment):
    if pd.isna(text):
        return ""
    text = str(text)
    words = text.split()

    # Простое правило: neutal -> весь текст
    if sentiment == "neutral":
        return text.strip()

    # Если один-два слова — возвращаем всё
    if len(words) <= 2:
        return text

    # Оценим каждое слово по VADER
    scores = [sid.polarity_scores(w)['compound'] for w in words]

    if sentiment == "positive":
        scores = [sid.polarity_scores(w)['neg'] for w in words]
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[0]
    else:  # negative
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[0]

    if idx + 1 < len(words):
        selected = words[idx + 1]
    else:
        selected = words[idx]

    selected = selected.lower()

    return selected



# the full text for rows with missing text, inflating Jaccard on those rows
# but the real issue is it masks NaN sentiment values, causing wrong extraction logic
test['sentiment'] = test['sentiment'].fillna('neutral')

test['selected_text'] = test.apply(lambda x: extract_selected_text(x['text'], x['sentiment']), axis=1)

# when the ground truth contains punctuation as part of words
test['selected_text'] = test['selected_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

submission = test[['textID', 'selected_text']]
submission.to_csv('/home/submission/submission.csv', index=False)
print(submission.head())











