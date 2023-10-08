import pandas as pd
from datetime import date, timedelta
import joblib


def forecast(sales: dict, item_info: dict, store_info: dict) -> list:
    """
    Функция для предсказания продажЖ
    :params sales: исторические данные по продажам
    :params item_info: характеристики товара
    :params store_info: характеристики магазина
    """
    # Загрузка датафреймов
    sales_df = pd.DataFrame(sales).drop(
        columns=['sales_units_promo', 'sales_rub_promo']
    )
    holiday_df = pd.read_csv('./holidays_covid_calendar.csv').drop(
        columns=['year', 'day', 'weekday', 'calday', 'covid']
    )
    holiday_df['date'] = pd.to_datetime(holiday_df['date'])
    mapping = pd.read_csv('./mapping.csv')
    model_input = pd.DataFrame({
        'store': [store_info['store']] * 14,
        'sku': [item_info['sku']] * 14,
        'date': [date.today() + timedelta(days=d) for d in range(1, 15)],
        'sales_units': [0] * 14,
        'sales_rub': [0] * 14
    })
    # Создание признаков
    sales_df.insert(0, 'sku', [item_info['sku']] * len(sales_df))
    sales_df.insert(0, 'store', [store_info['store']] * len(sales_df))
    model_input.insert(3, 'sales_type', [0] * 14)
    model_input = pd.concat([sales_df, model_input]).reset_index(drop=True)
    model_input['date'] = pd.to_datetime(model_input['date'])
    model_input = model_input.merge(holiday_df, on='date', how='left')
    model_input['uom'] = [item_info['uom']] * len(model_input)
    model_input.index = model_input['date']
    model_input = create_date_features(model_input)
    model_input = create_lag_features(model_input)
    model_input = create_rolling_features(model_input)
    model_input = create_expanding_features(model_input)
    model_input = fillna_lag_rolling_features(model_input)
    product_mapping = mapping[mapping['pr_sku_id'] == item_info['sku']]
    store_mapping = mapping[mapping['st_id'] == store_info['store']]
    if (len(product_mapping) != 0):
        model_input['pr_cluster_label'] = list(
            product_mapping['pr_cluster_label'].unique()
        )[0]
    else:
        model_input['pr_cluster_label'] = mapping['pr_cluster_label'].max() + 1
    if (len(store_mapping) != 0):
        model_input['st_cluster_label'] = list(
            store_mapping['st_cluster_label'].unique()
        )[0]
        model_input['st_id_encoded'] = list(
            store_mapping['st_id_encoded'].unique()
        )[0]
    else:
        model_input['st_cluster_label'] = mapping['st_cluster_label'].max() + 1
        model_input['st_id_encoded'] = mapping['st_id_encoded'].max() + 1
    if(len(product_mapping) != 0):
        model_input['pr_sku_id_encoded'] = list(
            product_mapping['pr_sku_id_encoded'].unique()
        )[0]
    else:
        model_input['pr_sku_id_encoded'] = mapping['pr_sku_id_encoded']
    if (len(store_mapping) != 0):
        model_input['st_sku_interaction'] = list(
            store_mapping['st_sku_interaction'].unique()
        )[0]
    else:
        model_input['st_sku_interaction'] = mapping['st_sku_interaction'].max() + 1
    # Загрузка модели и итеративное предсказание
    model = joblib.load('./lgbm.sav')
    for i in range(14, 0, -1):
        predicted = model.predict([model_input.drop(
            columns=['store', 'sku', 'date', 'sales_units', 'sales_rub']
        ).iloc[-i, :]])[0]
        sales_units = list(model_input['sales_units'])
        sales_units[-i] = round(predicted)
        model_input['sales_units'] = sales_units
        model_input = create_date_features(model_input)
        model_input = create_lag_features(model_input)
        model_input = create_rolling_features(model_input)
        model_input = create_expanding_features(model_input)
        model_input = fillna_lag_rolling_features(model_input)
    return list(model_input.tail(14)['sales_units'])


def create_date_features(df):
    # признаки по дате
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofyear'] = df.index.dayofyear
    df['dayofweek'] = df.index.dayofweek
    df['weekofyear'] = df.index.isocalendar().week
    df['weekofyear'] = df['weekofyear'].astype('int32')

    # Добавление столбца для выходных дней
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Добавление столбца для рабочих дней
    df['is_workday'] = (
        (df['dayofweek'] >= 0) & (df['dayofweek'] <= 4) & (df['holiday'] == 0)
    ).astype(int)

    return df


def create_lag_features(df, lag_days=range(1, 15)):
    for lag in lag_days:
        df[f'lag_{lag}'] = df.groupby(
            ['store', 'sku', 'sales_type']
        )['sales_units'].transform(lambda x: x.shift(lag))
    return df


def create_rolling_features(df, window_sizes=[7, 14, 28]):
    for window in window_sizes:
        df[f'rolling_mean_t{window}'] = df.groupby(
            ['store', 'sku', 'sales_type']
        )['sales_units'].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )

        df[f'rolling_std_t{window}'] = df.groupby(
            ['store', 'sku', 'sales_type']
        )['sales_units'].transform(
            lambda x: x.shift(1).rolling(window).std()
        )
    return df


def create_expanding_features(df):
    df['expanding_median'] = df.groupby(
        ['store', 'sku', 'sales_type']
    )['sales_units'].transform(
        lambda x: x.expanding().median()
    )
    return df


def fillna_lag_rolling_features(df):
    lag_cols = [col for col in df.columns if 'lag' in col]

    for col in lag_cols:
        df[col].fillna(0, inplace=True)

    rolling_cols = [col for col in df.columns if 'rolling' in col]
    for col in rolling_cols:
        df[col].fillna(0, inplace=True)

    return df
