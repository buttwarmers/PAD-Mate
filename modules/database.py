# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import sqlite3
import os
from atexit import register
from pathlib import Path
import time
import re

try:
    from .utils import timeit
except:
    os.chdir(os.path.dirname(__file__))
    from utils import timeit

# =============================================================================
# DATABASE MANAGER
# =============================================================================
class Database:
    def __init__(self, 
            src: str = None, 
            cache_limit: int = 16,
            *args, 
            **kwargs
            ):
        
        # exit handling
        register(self._exit)
        
        # get source folder for database files
        self.source_folder = self.get_source_folder() if src is None else src
        
        # make sure source folder is a directory and not a file
        if os.path.isfile(self.source_folder):
            self.source_folder = os.path.dirname(self.source_folder)
            
        if not os.path.exists(self.source_folder):
            raise OSError(f'Source folder {self.source_folder} does not exist')
            
        # make sure source folder has trailing slash
        self.source_folder = os.path.join(self.source_folder, '')
        
        # database file extensions
        self.valid_extensions = ('.sqlite', '.feather', '.pkl')
        
        # get list of database files
        self.source_files = self.get_source_files()
        
        # load sqlite stuff
        self._setup_sqlite()
        
        # set up caching
        self.cache_limit = cache_limit
        self.reset_cache()
        
        # class variables
        self.dataframes = None
        
# =============================================================================
# CHECK FOR REQUIRED FILES
# =============================================================================
    def get_source_folder(self):
        parent = Path(os.path.dirname(__file__)).parent
        return os.path.join(parent, 'databases', '')
    
    def get_source_files(self):
        folder = self.get_source_folder()
        files = [os.path.join(folder, f) for f in os.listdir(folder) 
                 if f.endswith(self.valid_extensions)]
        if not files:
            raise ValueError('No database files found in {folder}')
        return files
        
# =============================================================================
# SQLITE
# =============================================================================
    @timeit
    def load_sqlite(self, sqlite_file: str = None) -> sqlite3.Connection:
        if sqlite_file is None:
            sqlite_file = os.path.join(self.source_folder, 'pad.sqlite')
        if not os.path.exists(sqlite_file):
            raise OSError(f'sqlite database file {sqlite_file} not found')
        return sqlite3.connect(sqlite_file)
    
    def close_sqlite(self) -> None:
        try:
            self.sqlite.close()
            print('Closed sqlite database file')
        except:
            pass
    
    def sqlite_cmd(self, cmd) -> list:
        self.sqlite = self.load_sqlite() if self.sqlite is None else self.sqlite
        cur = self.sqlite.cursor()
        try:
            cur.execute(cmd)
            res = cur.fetchall()
            cur.close()
            return [r[0] for r in res] if res else res
        except Exception as ex:
            cur.close()
            print(f'Error executing {cmd}: {ex}')
        
    def get_sqlite_tables(self) -> list:
        cmd = 'select name from sqlite_master where type = "table"'
        tables = self.sqlite_cmd(cmd)
        return tables
    
    def _setup_sqlite(self) -> None:
        self.sqlite = self.load_sqlite()
        self.sqlite_tables = self.get_sqlite_tables()
        
# =============================================================================
# SAVING AND LOADING DATAFRAMES
# =============================================================================
    def dataframe_from_table(self, table: str) -> pd.DataFrame:
        if table in self.sqlite_tables:
            return pd.read_sql_query(f'select * from {table}', self.sqlite)
        else:
            print(f'{table} not found in sqlite tables:')
            self._print_items(self.sqlite_tables)
            
    def dataframe_from_file(self, name: str) -> pd.DataFrame:
        filename = name.split('.')[0] + '.feather'
        filepath = os.path.join(self.source_folder, filename)
        if not os.path.exists(filepath):
            return print(f'{filepath} does not exist')
        return pd.read_feather(filepath)
    
    @timeit
    def load_dataframes(self) -> dict:
        dfs = {t: self.dataframe_from_file(t) for t in self.sqlite_tables}
        self.dataframes = dfs
        return self.dataframes
        
    @timeit
    def update_dataframes(self) -> dict:
        sql_dfs = {t: self.dataframe_from_table(t) for t in self.sqlite_tables}
        if self.dataframes is None:
            self.dataframes = self.load_dataframes()
        
        # update any new info
        for name, df in self.dataframes.items():
            old = sql_dfs[name]
            self.dataframes[name] = old if df is None else df.merge(old, 
                                                                    how='outer')
            
        return self.dataframes
        
    @timeit
    def save_dataframes(self) -> None:
        for k, v in self.dataframes.items():
            filepath = os.path.join(self.source_folder, f'{k}.feather')
            # cannot feather non-default indices so reset index to default
            v.reset_index(inplace=True, drop=True) 
            v.to_feather(filepath)
        print('Saved dataframes')
    
# =============================================================================
# SEARCHING DATAFRAMES
# =============================================================================
    def df_by_name(self, df_name: str) -> pd.DataFrame:
        if df_name not in self.dataframes.keys():
            print(f'{df_name} not in dataframes:')
            self._print_items(self.dataframes.keys())
            return
        return self.dataframes[df_name]

    def sort_df(self, 
            df: pd.DataFrame, 
            columns: list, 
            ascending: bool = False
            ) -> pd.DataFrame:
        columns = [columns] if type(columns) is not list else columns
        for col in columns:
            if col not in df.columns:
                self._column_not_found(df, col)
                return df
        try:
            return df.sort_values(by=columns, axis=0, ascending=ascending)
        except Exception as ex:
            print(f'Unable to sort {columns}: {ex}')
            return df

    def _filter_dataframe(self, df_name, use_regex, *args, **kwargs):
        df = self.df_by_name(df_name)
        if df is None:
            return
        
        # only use terms that are in dataframe columns
        terms = {k: v for k, v in kwargs.items() if k in df.columns}
        
        # only use tuple values of (comparison, value)
        terms = {k: v for k, v in terms.items() if type(v) is tuple}
        terms = {k: v for k, v in terms.items() if len(v) == 2}
        
        valid_comparisons = ['<', '>', '<=', '>=', '==', 
                             'contains', 'does not contain']
        for column, (comparison, value) in terms.items():
            if comparison not in valid_comparisons:
                print(f'Comparison {comparison} is not valid:')
                self._print_items(valid_comparisons)
                continue
            df = self.apply_mask(df, column, comparison, value, use_regex)
            
        return df
            
    @timeit
    def filter_dataframe(self, df_name, use_regex: bool = True, **kwargs):
        return self.cache(self._filter_dataframe, df_name, use_regex, **kwargs)
            
    def apply_mask(self, 
           df: pd.DataFrame, 
           column: str, 
           comparison: str, 
           value,
           use_regex: bool = True
           ):
        if column not in df.columns:
            self._column_not_found(df, column)
            return df
        
        try:        
            if comparison == '<':
                return df[df[column] < value]
            elif comparison == '>':
                return df[df[column] > value]
            elif comparison == '<=':
                return df[df[column] <= value]
            elif comparison == '>=':
                return df[df[column] >= value]
            elif comparison == '==':
                return df[df[column] == value]
            else:
                value = str(value)
                mask = df[column].apply(str).str.contains(value, regex=True, 
                                                          flags=re.I)
                if comparison == 'contains':
                    return df[mask]
                elif comparison == 'does not contain':
                    return df[~mask]
            return df
        except Exception as ex:
            print(f'Could not filter dataframe: {ex}')
            return df
       
# =============================================================================
# CACHING
# =============================================================================
    def reset_cache(self) -> None:
        self.cached = {}
        
    def cache(self, func, *args, **kwargs):
        name = func.__name__
        
        # store args in kwargs
        kwargs['__args__'] = args
        
        # turn kwargs into hashable object so it can be used as key
        uuid = self._freeze_dict(kwargs)
        
        # check for cached results and return them if they exist
        cached_results = None
        if name in self.cached.keys():
            cached_results = self.cached[name]
            if uuid in cached_results.keys():
                print('Using cached results')
                return self.cached[name][uuid]
            
        # make sure number of results doesn't exceed cache size limit
        if cached_results is None:
            self.cached[name] = {}
        elif len(cached_results.keys() > self.cache_limit):
            oldest_entry = list(cached_results.keys())[0]
            self.cached[name].pop(oldest_entry)
            
        # store cached result
        result = func(*args, **kwargs)
        self.cached[name].update({uuid: result})
        
        return result
        
# =============================================================================
# UTILS    
# =============================================================================
    def _exit(self) -> None:
        self.close_sqlite()
        self.save_dataframes()
        
    def _print_items(self, items: list) -> None:
        [print(f'\t{i}') for i in items]
        
    def _column_not_found(self, df: pd.DataFrame, column: str) -> None:
        print(f'Column {column} not in dataframe columns:')
        self._print_items(df.columns)
        
    def _freeze_dict(self, dictionary: dict) -> tuple:
        return frozenset(dictionary.items())
        
# =============================================================================
# TESTING
# =============================================================================
    def _test(self) -> None:
        self.load_dataframes()
        # kwargs = {
        #     'name_na': ('==', 'Tyrra')
        #     }
        # res = self.filter_dataframe('monsters', **kwargs)
        self._print_items(self.dataframes['monsters'].columns)
        
# =============================================================================
# LOCAL RUNNING
# =============================================================================
if __name__ == '__main__':
    @timeit
    def main():
        db = Database()
        db._test()
        db._exit()
        
    main()

