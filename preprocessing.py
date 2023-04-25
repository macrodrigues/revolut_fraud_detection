"""Script with different functions aimed for preprocessing."""
import pandas as pd
import datetime as dt
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_validate


def gen_main_df():
    """Take 3 datasets and merge into one final one."""
    users = pd.read_csv('data/users.csv')  # contains user information
    users.rename(columns={'ID': 'USER_ID'}, inplace=True)
    transactions = pd.read_csv(
        'data/transactions.csv')  # contains transactions information
    merge = pd.merge(transactions, users, on=['USER_ID'])
    merge.rename(columns={
        'CREATED_DATE_x': 'TRANSACTION_DATE_TIME',
        'CREATED_DATE_y': 'USER_CREATION_DATE'}, inplace=True)
    fraudsters = pd.read_csv(
        'data/fraudsters.csv')  # contains the ids of the fraudsters
    merge['FRAUD'] = merge['USER_ID']
    merge['FRAUD'] = merge['FRAUD'].apply(
        lambda x: 1 if x in list(fraudsters['USER_ID'].values) else 0)
    merge['BIRTH_DATE'] = merge['BIRTH_DATE'].apply(
        lambda x: dt.datetime.strptime(x, '%Y-%m-%d').date())
    merge['BIRTH_DATE'] = merge['BIRTH_DATE'].apply(
        lambda x: dt.datetime.today().date().year - x.year)
    merge.rename(columns={'BIRTH_DATE': 'AGE'}, inplace=True)
    df = merge[[
        'ID',
        'USER_ID',
        'TRANSACTION_DATE_TIME',
        'USER_CREATION_DATE',
        'AGE',
        'COUNTRY',
        'TYPE',
        'STATE',
        'AMOUNT_GBP',
        'CURRENCY',
        'FRAUD']]
    df.to_csv('data/data.csv')


def age_range(x):
    """Parser for age values."""
    if x < 36:
        return '20-35'
    if x > 35 and x < 51:
        return '36-50'
    if x > 50 and x < 66:
        return '51-65'
    else:
        return '66-80'


def encoding(df):
    """Multiple pandas functions to encode features."""
    # CONVERT DATES TO DATETIME
    df[["TRANSACTION_DATE_TIME", "USER_CREATION_DATE"]] = df[[
        "TRANSACTION_DATE_TIME", "USER_CREATION_DATE"]].apply(pd.to_datetime)

    # FILTER AGE COLUMN
    df['AGE'] = df['AGE'].apply(age_range)

    # TRANSACTION PER DAY OF THE WEEK
    df['transaction_day_week'] = df['TRANSACTION_DATE_TIME'].dt.day_name()
    df['transaction_day_week'] = df['transaction_day_week'].apply(
        lambda x: 'Weekend' if (x == 'Saturday' or x == 'Sunday') else x)

    # ENCODE COUNTRIES - Target Encoding
    df["countries_encoded"] = df.groupby(
        "COUNTRY")["FRAUD"].transform("mean")

    # ENCODE CURRENCIES - Target Encoding
    df['currencies_encoded'] = df.groupby(
        "CURRENCY")["FRAUD"].transform("mean")

    # ONE-HOT-ENCODE
    encoder = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoded_cols = pd.get_dummies(
        df,
        columns=['AGE', 'STATE', 'TYPE', 'transaction_day_week'])
    encoder_df = pd.DataFrame(
        encoder.fit_transform(df[[
            'AGE', 'STATE', 'TYPE', 'transaction_day_week']]).toarray())
    df = df.join(encoder_df)
    columns = dict(zip(
        encoder_df.columns.values,
        one_hot_encoded_cols.columns[10:]))
    df = df.rename(columns=columns)

    # ENCODE USER_CREATION_DATE- Target Encoding
    df["user_year"] = df['USER_CREATION_DATE'].dt.year
    df["user_year"] = df.groupby(
        "user_year")["FRAUD"].transform("mean")

    # DROP UNUSED COLUMNS
    df.drop([
        'AGE',
        'STATE',
        'TYPE',
        'transaction_day_week',
        'TRANSACTION_DATE_TIME',
        'USER_CREATION_DATE',
        'CURRENCY',
        'COUNTRY'],

        axis=1, inplace=True)

    return df


def pipe_preprocessor(df):
    # Impute then Scale for numerical variables
    num_features = [df.columns]
    num_transformer = Pipeline([
        ('imputer', KNNImputer()),
        ('scaler', MinMaxScaler())])

    # Build preprocessor
    preprocessor = ColumnTransformer([
        ('num_tr', num_transformer, num_features)], remainder='drop')

    return preprocessor


def balance(df):
    X = df[
        'AMOUNT_GBP',
        'user_year',
        'countries_encoded',
        'currencies_encoded',
        'AGE_20-35',
        'AGE_36-50',
        'AGE_51-65',
        'AGE_66-80',
        'STATE_COMPLETED',
        'STATE_DECLINED',
        'STATE_FAILED',
        'STATE_REVERTED',
        'TYPE_ATM',
        'TYPE_CARD_PAYMENT',
        'TYPE_EXCHANGE',
        'TYPE_FEE',
        'TYPE_TOPUP',
        'TYPE_TRANSFER',
        'transaction_day_week_Friday',
        'transaction_day_week_Monday',
        'transaction_day_week_Thursday',
        'transaction_day_week_Tuesday',
        'transaction_day_week_Wednesday',
        'transaction_day_week_Weekend']

    y = df['FRAUD']
    rus = RandomUnderSampler()
    X_rus, y_rus = rus.fit_resample(X, y)
    df = pd.concat([X_rus, y_rus], axis=1)
    return df


if __name__ == '__main__':
    # gen_main_df()
    df = pd.read_csv('data/data.csv')
    df.drop([df.columns[0]], axis=1, inplace=True)
    df = encoding(df)
    X, y = balance(df)
    pipe = pipe_preprocessor(X)
