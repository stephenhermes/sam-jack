"""Get plot data for films."""

import datetime
import glob
import os
import json
import pathlib

import pandas as pd
import requests

from dotenv import load_dotenv, find_dotenv
#find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)

API_KEY = os.environ.get('OMDB_API_KEY')


def _instantiate_dir(prefix='omdb-movies-'):
    """Create directory for pulled data.

    Parameters
    ----------
    prefix : string, optional (default='omdb-movies-')
        The the portion of the folder name before the auto-generated
        date suffix.

    Returns
    -------
    data_dir : pathlib directory
        The directory to where pulled data should be written.

    """

    run_date = datetime.datetime.now().strftime('%Y%m%d')
    # [2] since using `cookiecutter` project layout
    data_dir = pathlib.Path(__file__).resolve().parents[2]
    data_dir = data_dir / 'data' / 'raw' / (prefix + run_date)
    try:
        data_dir.mkdir()
    except FileExistsError as e:
        print(
            f"Directory `{data_dir}` already exists. Any files already " 
            "present will be skipped."
        )
    
    return data_dir


def get_movie_data(name, year=None, full_plot=False, api_key=API_KEY):
    """Returns json from OMDb API for specified movie.
    
    Parameters
    ----------
    name : string
        The movie name

    year : int, string or None, optional (default=None)
        The year of the movie, if provided.

    full_plot : bool, optional (default=False)
        If the fill plot of the movie should be returned or not.

    """
    api_url = f'http://private.omdbapi.com/?apikey={api_key}'
    # There are actually utilities that can automatically escape invalid characters
    # but here we do the manual dumb solution
    name = name.lower().replace(' ', '+')
    
    body = {'t': name}
    if year is not None:
        body['y'] = year
    if full_plot:
        body['plot'] = 'full'
    response = requests.get(api_url, params=body)
    
    # Throw error if API call has an error
    if response.status_code != 200:
        raise requests.HTTPError(
            f'Couldn\'t call API. Error {response.status_code}.'
        )
     
    # Throw error if movie not found
    # if response.json()['Response'] == 'False':
    #     raise ValueError(response.json()['Error'])
    
    return response.json()


if __name__ == '__main__':
    data_dir = _instantiate_dir()
    # Get wikipedia movie names
    files = data_dir.parent.glob('wikipedia-*/*.csv')

    for f in files:
        file_prefix = f.stem
        year = f.stem.split('-')[1]

        print(f'Fetching data for {year}...')

        output_path = data_dir / f'plots-{year}.csv'
        if output_path.exists():
            print(f'Already have data for {year}.')
            continue

        df = pd.read_csv(f)
        # Just in case there is inconcistency
        df.columns = [c.lower() for c in df.columns]

        titles = df['title']
        raw_data = {
            'imdbID': [],
            'Title': [],
            'Year': [],
            'Released': [], 
            'Genre': [],
            'Actors': [],
            'Plot': [],
            'Response': []
        }
        for t in titles:
            # For some reason, 2014 has a float title?
            response = get_movie_data(str(t), year=year, full_plot=True)
            # If can't get for year, search without year
            if response['Response'] == 'False':
                response = get_movie_data(t, full_plot=True)

            # If still no data, record name and flag
            if response['Response'] == 'False':
                response['Title'] = t

            for key in raw_data.keys():
                val = response.pop(key, None)
                if val == 'N/A':
                    val = None
                raw_data[key].append(val)

        data = pd.DataFrame(raw_data)
        data.columns = [c.lower() for c in data.columns]
        data['year_mismatch'] = data['year'] == str(year)
        
        data.to_csv(output_path, index=False)

    print('Finished fetching data.')
