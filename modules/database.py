# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import sqlite3
import os
from atexit import register
from pathlib import Path
import re
import numpy as np

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    from utils import timeit
else:
    if os.getcwd() == os.path.dirname(__file__):
        from utils import timeit
    else:
        try:
            from modules.utils import timeit
        except:
            from .utils import timeit
            
# =============================================================================
# GLOBALS
# =============================================================================
DATABASE_FOLDER = os.path.join(Path(__file__).parent.parent, 'databases', '')
os.makedirs(DATABASE_FOLDER, exist_ok=True)

# =============================================================================
# DATABASE MANAGER
# =============================================================================
class Database:
    def __init__(self, 
            cache_limit: int = 16,
            *args, 
            **kwargs
            ):
        
        # exit handling
        register(self._exit)
        
        # database file extensions
        self.valid_extensions = ('.sqlite', '.feather', '.pkl')
        
        # get list of database files
        self.source_files = self.get_source_files()
        
        # load sqlite stuff
        self._setup_sqlite()
        
        # set up caching
        self._cache_limit = cache_limit
        self._reset_cache()
        
        # class variables
        self.dataframes = self.load_dataframes()
        
    # =============================================================================
    # CHECK FOR REQUIRED FILES
    # =============================================================================
    def get_source_folder(self):
        parent = Path(os.path.dirname(__file__)).parent
        return os.path.join(parent, 'databases', '')
    
    def get_source_files(self):
        files = [os.path.join(DATABASE_FOLDER, f) for f in os.listdir(DATABASE_FOLDER) 
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
            sqlite_file = os.path.join(DATABASE_FOLDER, 'pad.sqlite')
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
            self.print_items(self.sqlite_tables)
            
    @staticmethod
    def dataframe_from_file(name: str) -> pd.DataFrame:
        filename = name.split('.')[0] + '.feather'
        filepath = os.path.join(DATABASE_FOLDER, filename)
        if not os.path.exists(filepath):
            return print(f'{filepath} does not exist')
        return pd.read_feather(filepath)
    
    @timeit
    def load_dataframes(self) -> dict:
        dfs = {t: self.dataframe_from_file(t) for t in self.sqlite_tables}
        self.dataframes = dfs
        self.fix_na_names()
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
            filepath = os.path.join(DATABASE_FOLDER, f'{k}.feather')
            # cannot feather non-default indices so reset index to default
            v.reset_index(inplace=True, drop=True) 
            v.to_feather(filepath)
        print('Saved dataframes')
    
    # =============================================================================
    # GENERIC DATAFRAME SEARCHING
    # =============================================================================
    def df_by_name(self, df_name: str) -> pd.DataFrame:
        if df_name not in self.dataframes.keys():
            print(f'{df_name} not in dataframes:')
            self.print_items(self.dataframes.keys())
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
                self.print_items(valid_comparisons)
                continue
            df = self.apply_mask(df, column, comparison, value, use_regex)
            
        return df
            
    @timeit
    def filter_dataframe(self, df_name, use_regex: bool = True, **kwargs):
        return self._cache(self._filter_dataframe, df_name, use_regex, **kwargs)
            
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
       
    def fix_na_names(self):
        df = self.dataframes['monsters']
        df['name_na'] = np.where(df['name_na_override'].isna(), df['name_na'],
                                 df['name_na_override'])
        self.dataframes['monsters'] = df
        return df
       
    # =============================================================================
    # SPECIALIZED SEARCH FUNCTIONS
    # =============================================================================
    def name_from_id(self, monster_id: int, region: str='na') -> str:
        df = self.dataframes['monsters']
        res = df[df['monster_id'] == monster_id][f'name_{region}'].values
        return res[0] if res else print(f'Name not found for ID {monster_id}')
    
    def names_from_ids(self, monster_ids: list) -> list:
        df = self.dataframes['monsters']
        return list(df[df['']])
       
    # =============================================================================
    # CACHE
    # =============================================================================
    def _reset_cache(self) -> None:
        self._cached = {}
        
    def _cache(self, func, *args, **kwargs):
        name = func.__name__
        
        # store args in kwargs
        kwargs['__args__'] = args
        
        # turn kwargs into hashable object so it can be used as key
        uuid = self._freeze_dict(kwargs)
        
        # check for cached results and return them if they exist
        cached_results = None
        if name in self._cached.keys():
            cached_results = self._cached[name]
            if uuid in cached_results.keys():
                print('Using cached results')
                return self._cached[name][uuid]
            
        # make sure number of results doesn't exceed cache size limit
        if cached_results is None:
            self._cached[name] = {}
        elif len(cached_results.keys() > self._cache_limit):
            oldest_entry = list(cached_results.keys())[0]
            self._cached[name].pop(oldest_entry)
            
        # store cached result
        result = func(*args, **kwargs)
        self._cached[name].update({uuid: result})
        
        return result
        
    # =============================================================================
    # UTILS    
    # =============================================================================
    def _exit(self) -> None:
        self.close_sqlite()
        self.save_dataframes()
        
    @staticmethod
    def print_items(items: list) -> None:
        [print(f'\t{i}') for i in items]
        
    def _column_not_found(self, df: pd.DataFrame, column: str) -> None:
        print(f'Column {column} not in dataframe columns:')
        self.print_items(df.columns)
        
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
        self.print_items(self.dataframes['monsters'].columns)
        fixed = self.fix_na_names()
        return fixed
        
# =============================================================================
# LOCAL RUNNING
# =============================================================================
if __name__ == '__main__':
    db = Database()
    fixed = db._test()
    db._exit()
    