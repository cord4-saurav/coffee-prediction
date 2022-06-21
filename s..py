import pandas as pd


def get_means(df):
    df = df.copy()
    countries = df[df.columns[0]]
    means = df.mean(axis=1)
    df = pd.concat([countries, means], axis=1)
    df.columns = ['country', countries.name]
    return df

def make_df(dfs):
    processesed_dfs = []

    for df in dfs:
        processesed_dfs.append(get_means(df))

    df = processesed_dfs[0]

    for i in range(1, len(processesed_dfs)):
        df = df.merge(processesed_dfs[1], on='country')

    return df


def preprocess_input(df):
    df = df.copy()

    y = df['total_production']
    x = df.drop('total_production', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=1)
    return  x_train, x_test, y_train, y_test