# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import os
from pathlib import Path
import configparser
import ast

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    from boxmanager import BoxManager
    from utils import eval_dtypes
    
else:
    try:
        from modules.boxmanager import BoxManager
        from modules.utils import eval_dtypes
    except ImportError:
        from boxmanager import BoxManager
        from utils import eval_dtypes

# =============================================================================
# GLOBALS
# =============================================================================
USER_FOLDER = os.path.join(Path(__file__).parent.parent, 'users', '')
os.makedirs(USER_FOLDER, exist_ok=True)

DEFAULT_USER = 'test'
MATCH_COLUMNS = ['source_filepath', 'source_frame', 'source_id', 'bbox', 'name']

# =============================================================================
# USER MANAGEMENT
# =============================================================================
class User:
    def __init__(self, username: str = DEFAULT_USER):
        # store input variables
        self.username = username
        self._username = self.username
        
        # TODO: validate username
        
        # get folder where user files are located
        self.folder = os.path.join(USER_FOLDER, username, '')
        self.config_filepath = os.path.join(self.folder, 'config.cfg')
        self.matches_filepath = os.path.join(self.folder, 'matches.csv')
        self.box_filepath = os.path.join(self.folder, 'box.csv')
        os.makedirs(self.folder, exist_ok=True)
        
        # load files
        self.load_config()
        self.load_matches()
        
    # =============================================================================
    # LOADING AND SAVING FILES
    # =============================================================================
    @staticmethod
    def get_box_filepath(username: str) -> str:
        return os.path.join(USER_FOLDER, username, 'box.csv')
    
    def load_config(self) -> None:
        # get the filepath
        cfg = configparser.ConfigParser()
        if not os.path.exists(self.config_filepath):
            self.make_default_config()
            
        # read the config and convert to dictionary
        cfg.read(self.config_filepath)
        self.config = {s: dict(cfg.items(s)) for s in cfg.sections()}
        
        # convert values from strings
        for section in self.config.keys():
            for key, val in self.config[section].items():
                try: self.config[section][key] = ast.literal_eval(val)
                except: pass
        print(f'Loaded config for {self.username}')
            
    def save_config(self) -> None:
        # convert config dictionary back to right format
        cfg = configparser.ConfigParser()
        for section, info in self.config.items():
            cfg.add_section(section)
            for key, val in info.items():
                cfg[section][key] = str(val)
                
        # save config
        with open(self.config_filepath, 'w') as f:
            cfg.write(f)
        print('Saved config for {self.username}')
            
    def load_matches(self) -> None:
        if not os.path.exists(self.matches_filepath):
            self.matches = pd.DataFrame(columns=MATCH_COLUMNS)
            self.save_matches()
            return
        self.matches = eval_dtypes(pd.read_csv(self.matches_filepath))
        print(f'Loaded matches for {self.username}')
        
    def save_matches(self) -> None:
        self.matches.to_csv(self.matches_filepath, index=False)
        print(f'Saved matches for {self.username}')
        
    # =============================================================================
    # DEFAULTS
    # =============================================================================
    def make_default_config(self) -> None:
        cfg = configparser.ConfigParser()
        cfg['General'] = {
            'username': f'{self.username}',
            'account_name': 'Default',
            }
        cfg['Matching'] = {
            'input_w': 'None',
            'input_h': 'None',
            'box_top_boundary': '0',
            'box_bottom_boundary': '1',
            'rows_per_page': 'None',
            'scale': 'None',
            'scroll_bbox': 'None',
            'orb_crop_factor': '0.15',
            'icon_w': 'None',
            'icon_h': 'None',
            }
        with open(self.config_filepath, 'w') as file:
            cfg.write(file)
            
    # =============================================================================
    # PROPERTIES
    # =============================================================================
    @property
    def username(self) -> str:
        return self._username
    
    @username.setter
    def username(self, username: str) -> None:
        if isinstance(username, str):
            self._username = username
            
    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, config) -> None:
        self._config = config
            
    @property
    def matches(self) -> pd.DataFrame:
        return self._matches
    
    @matches.setter
    def matches(self, matches: pd.DataFrame) -> None:
        if isinstance(matches, pd.DataFrame):
            self._matches = matches
    
    @property
    def box(self):
        return self._box
    
    @box.setter
    def box(self, box: pd.DataFrame) -> pd.DataFrame:
        if isinstance(box, pd.DataFrame):
            self._box = box
           
    @property
    def images(self) -> list:
        images = []
        img_exts = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        for root, dirs, files in os.walk(self.folder):
            for filename in files:
                filepath = os.path.join(root, filename)
                if filepath.lower().endswith(img_exts):
                    images.append(filepath)
        return images
        
    @property
    def videos(self) -> list:
        videos = []
        vid_exts = ('.mp4', 'mp3', '.gif', '.mkv', '.mov', '.webm', '.avi', '.ogg')
        for root, dirs, files in os.walk(self.folder):
            for filename in files:
                filepath = os.path.join(root, filename)
                if filepath.lower().endswith(vid_exts):
                    videos.append(filepath)
        return videos
    
    @property 
    def urls(self) -> list:
        return []
    
# =============================================================================
# LOCAL TESTING
# =============================================================================
if __name__ == '__main__':
    u = User('test')
    