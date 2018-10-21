"""Get movie names from Wikipedia.

There are a few inconsistencies with the way the Wikipedia pages
scraped by this script are set up:

- For 1952 the first table has a 'Notes/Studio' column while all
subsequent tables only have a 'Notes' column. The script returns a data
frame with a column for each. The should be combined into a single
column.

- Starting in 2014 there are multi_col spans for relase month. These
multicolumns should ideally be collapsed into a single field per row, 
or dropped.

- There used to be an error where a page had a random extra cell which
threw everything off.

"""

import datetime
import pathlib
import time

from bs4 import BeautifulSoup
import pandas as pd 
import requests


# Number of seconds to pause between subsequent wikipedia requests.
SLEEPTIME = 2


def _instantiate_dir(prefix='wikipedia-movies-'):
    """Create directory for pulled data.

    Parameters
    ----------
    prefix : string, optional (default='wikipedia-movies-')
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


def _parse_header_row(row, **options):
    """Parse the header row of a table.

    If a column spans multiple cells, then duplicate values are returned
    for each. Duplicated columns are appended with ``suffix`` and index
    for each repeat.
    
    Parameters
    ----------
    row : BeautifulSoup Tag object
        A <tr> tag from the html, all of whose cells are <th> tags

    suffix : string, optional (default='__')
        The seperator between column name and index for multi-cols.

    Returns
    -------
    columns : list
        The headers as a list.

    """ 

    suffix = options.pop('suffix', '__')

    # If exists <td> tags then not a header row
    if row.find_all('td'):
        raise ValueError("`row` is not a table header.")
    
    columns = []
    for x in row.find_all('th'):
        colspan = int(x.attrs.pop('colspan', 1))
        if colspan > 1:
            columns += \
                [x.text.strip() + suffix + str(i) for i in range(colspan)]
        else:
            columns += [x.text.strip()]

    return columns
            

def _parse_data_row(row, columns, counters, **options):
    """Parse table data row.
    
    If a cell has multiple tags within it then each will be seperated
    by `sep` character.

    Parameters
    ----------
    row : BeautifulSoup Tag object
        A <tr> tag from the html, with data in at least one cell.

    columns : list
        The list of column headers for the table.

    counters : dict
        Counters used for propogating multirow data.

    sep : string, optional (default='')
        Seperator between multiple tags in a cell.

    Returns
    -------
    row_processed : list
        The processed row.

    """
    sep = options.pop('sep', '')

    cells = row.find_all(['th', 'td'])
    cell_cursor = 0
    row_processed = []

    for col in columns:
        # Check if values to propagate
        if counters[col][0] > 0:
            cell_value = counters[col][1]
            counters[col][0] -= 1   
        # If not propagate, get from cell    
        elif cell_cursor < len(cells):
            cell = cells[cell_cursor]
            rowspan = int(cell.attrs.pop('rowspan', 1))
            cell_value = sep.join(cell.stripped_strings)

            if rowspan > 1:
                counters[col] = [rowspan - 1, cell_value]
                
            cell_cursor += 1
        # If cursor out of index range, assume cells missing from
        else:
            cell_value = None

        row_processed.append(cell_value)     
        
    return row_processed


def parse_table(table, **options):
    """Parse table from html.
    
    Parameters
    table : BeautifulSoup Tag object
        The html <table> to parse.
    
    Returns
    -------
    table_parsed : list of lists
        An array of the table data.

    columns : list
        The column headers for the table.
    
    """
    rows = table.find_all('tr')

    header_row = rows.pop(0)
    columns = _parse_header_row(header_row, **options) 
    # For entries with rowspan, save value and amount left to fill        
    counters = dict((key, [0, None]) for key in columns)
    
    table_parsed = []
    for row in rows:
        row_parsed = _parse_data_row(row, columns, counters, **options)
        table_parsed.append(row_parsed)
    
    return table_parsed, columns


def fetch_movie_data(year, **options):
    """Get all movie data for specified year.

    Crawls wikipedia pages for `List_of_American_films_of_year_YYYY` to
    get names (and some additional data) of movies made. If no wikipedia
    page is found then returns `None`.

    Note this script was written on 2018-10-20 so the wikipedia pages
    may have changed since the script was tested and functional.

    Parameters
    ----------
    year : int
        The year for which to pull data. The Wikipedia pages are 
        moderately consistent in layout starting in 1900.

    first_as_master : bool, optional (default=True)
        If True, forces all tables collected to have the same columns
        and layout as the first table on the page.
    
    Returns
    -------
    movie_info : data frame
        The raw data in a pandas data frame.

    """

    first_as_master = options.pop('first_as_master', True)

    url = f'https://en.wikipedia.org/wiki/List_of_American_films_of_{year}'
    
    response = requests.get(url)  
    if response.status_code != 200:
        if response.status_code == 404:
            print(
                f"No Wikipedia page for year {year}. "
                f"Error {response.status_code}"
            )
            # Assume no more future data
            return None
        else:
            raise requests.HTTPError(
                "Failed GET Request to Wikipedia. "
                f"Error {response.status_code}"
            )
  
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get all tables on the site where the class is wikitable
    # This prevents it from including layout tabes, like the wikipedia 
    # footer etc.
    tables = soup.find_all('table', {'class': 'wikitable'})

    first_table = tables.pop(0)
    data, master_columns = parse_table(first_table, **options)
    fetched = pd.DataFrame(data, columns=master_columns)

    for table in tables:
        data, columns = parse_table(table)
        if first_as_master:
            columns = master_columns
        # Make sure we got data. 
        if data and columns:
            try:
                df = pd.DataFrame(data, columns=columns)
                fetched = fetched.append(df)
            # If for whatever reason there is an error, don't include
            except Exception as e:
                print(e)
                continue

    return fetched


def fetch_all(start_year=1900, **options):
    """Fetch data from `start_year` until no results.

    Parameters
    ----------
    start_year : int, optional (default=1900)
        The year to start the crawl from.

    verbose : boolean, optional (default=True)
        Whether or not to print runtime information.

    """

    verbose = options.pop('verbose', True)

    data_dir = _instantiate_dir()

    if verbose:
        print('Begin fetching data...')

    year, df = start_year, pd.DataFrame([])
    while df is not None:
        if verbose:
            print(f'Fetching data for {year}...')

        output_path = data_dir / f'movies-{year}.csv'
        
        if output_path.exists():
            print(f'Already have data for {year}.')
        else:
            df = fetch_movie_data(year, **options)
            if df is not None:
                df.to_csv(output_path, index=False)
            # Sleep for 2 seconds to not overwhelm Wikipedia
            time.sleep(SLEEPTIME)
        year += 1
        
    if verbose:
        print('Finished fetching data.')


if __name__ == '__main__':
    fetch_all(sep='|')
    