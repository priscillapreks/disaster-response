import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Loads the messages and categories datasets
    and merges the two datasets'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on='id')
    return df

def clean_data(df):
    '''cleaning pipeline'''
    
    # create a df of the individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # use first row to extract a list of column names for categories
    row = categories.iloc[0]
    category_colnames = [w[:-2] for w in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = [i.split('-')[1] for i in categories[column]]
        categories[column] = categories[column].astype(int)
        
    # Replace categories column in df with new category columns
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1, sort=False)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # drop unnecessary columns
    df = df.loc[df['related'] != 2]
    df.drop(columns=['id', 'original', 'genre'], inplace=True)
   
    return df


def save_data(df, database_filepath):
    '''Store data in a SQLite database'''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('messages_clean', engine, index=False)  


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