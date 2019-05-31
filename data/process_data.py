                ### imports ###

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


                ### Functions ###

def load_data(messages_filepath, categories_filepath):
    """Load and merge datasets

    Args:
        messages_filepath: Filepath for messages dataset.
        categories_filepath: Filepath for categories dataset.

    Returns:
        df: merged dataframe of messages and categories datasets.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how='left', on=['id'])

    return df

def clean_data(df):
    """Clean dataframe and convert categories

    Args:
        df: merged dataframe of messages and categories datasets.
    Returns:
        df: clean dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = list(row.transform(lambda x: x[:-2]))

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # dropping duplicates
    df = df.drop_duplicates()

    # dropping messages in different languages than English
    df = df[df.related != 2]

    return df

def save_data(df, database_filename):
    """Save cleaned dataframe into SQLite database.

    Args:
        df: clean dataframe
        database_filename: Filename of databased with clean dataframe

    Returns:
        None
    """
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
