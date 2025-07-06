from sklearn.preprocessing import LabelEncoder
def handle_missing_values(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.drop(['Cabin'], axis=1, inplace=True)
    return df

def encode_categorical_features(df):
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    return df

def feature_engineering(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
    return df

def preprocess_data(df):
    df = handle_missing_values(df)
    df = encode_categorical_features(df)
    df = feature_engineering(df)
    return df