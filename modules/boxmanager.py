# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import os
from pathlib import Path
import numpy as np
import itertools
import math
import ast

if __name__ == '__main__':
    os.chdir((Path(__file__).parent))
    from database import Database as db
    
else:
    try:
        from modules.database import Database as db
    except ImportError:
        from database import Database as db

# =============================================================================
# GLOBALS
# =============================================================================
BOX_FOLDER = os.path.join(Path(__file__).parent.parent, 'box', '')
os.makedirs(BOX_FOLDER, exist_ok=True)

EMPTY_BOX = np.empty(0, dtype = np.dtype([
                                    ('monster_id', 'uint8'),
                                    ('source_id', str),
                                    ('date_added', 'datetime64[ns]')
                                    ]))

# =============================================================================
# BOX MANAGER CLASS
# =============================================================================
class BoxManager:
    def __init__(self, box_name: str = 'box'):
        self.box_name = box_name
        self.box_file = os.path.join(BOX_FOLDER, f'{self.box_name}.csv')
        self.box = self.load_box()
        self._box = self.box
        self.db = db()

    # =============================================================================
    # LOADING / SAVING BOX AND MATCHES
    # =============================================================================
    def load_box(self) -> pd.DataFrame:
        if not os.path.exists(self.box_file):
            print(f'Box file {self.box_file} does not exist: creating empty box...')
            return self.new_box()
        try:
            self.box = pd.read_csv(self.box_file)
            self.fix_dtypes()
            return self.box
        except:
            print(f'Unable to load box from {self.box_file}: making new box...')
            return self.new_box()
        
    def import_box(self, box: pd.DataFrame) -> None:
        self.box = box
        for c in ['box_row', 'box_col', 'duplicate', 'grid_row', 'grid_col',
                  'grid_rows', 'grid_cols']:
            if c not in self.box.columns:
                self.box[c] = np.nan
        self.process_frames()
        self.fix_dtypes()
        
    def save_box(self) -> None:
        self.box.reset_index(drop=True).to_csv(self.box_file, index=False)
        print(f'Saved box to {self.box_file}')
        
    def new_box(self) -> pd.DataFrame:
        self.box = pd.DataFrame(EMPTY_BOX)
        self.save_box()
        return self.box
    
    # =============================================================================
    # UPDATING BOX
    # =============================================================================
    def update_box(self, new: pd.DataFrame) -> None:
        pass
    
    def identify_dupes(self) -> pd.DataFrame:
        # remove identical matches (same bounding box and source frame)
        df = self.box.drop_duplicates(subset=['source_id', 'bbox'])
        
        # function for calculating overlap between two frames
        def overlap(f1: int, f2: int) -> int:
            cards_1, cards_2 = list(f1['monster_id']), list(f2['monster_id'])
            return [m for m in cards_1 if m in cards_2]
        
        # get card positions
        self.process_frames()
        
        # sort by box position
        df.sort_values(by='box_row', ascending=True, inplace=True)
        
        # compare frames and remove duplicates based on overlap with other frames
        df.loc[:, 'duplicate'] = False
        frames = list(self.box.groupby(by=['source_id']))
        for (fid1, f1), (fid2, f2) in itertools.combinations(frames, r=2):
            common_cards = overlap(f1, f2)
            
            # ignore small overlap - likely just spillover
            if len(common_cards) <= 2: # 3
                continue
            
            for card in set(common_cards):
                n1 = len(f1[f1['monster_id'] == card])
                n2 = len(f2[f2['monster_id'] == card])
                if n1 == n2 == 1:
                    row1 = f1[f1['monster_id'] == card]['box_row'].values[0]
                    row2 = f2[f2['monster_id'] == card]['box_row'].values[0]
                    col1 = f1[f1['monster_id'] == card]['box_col'].values[0]
                    col2 = f2[f2['monster_id'] == card]['box_col'].values[0]
                    if abs(row1 - row2) > 1 or col1 != col2:
                        continue
                    else:
                        # print(f'Dropping {self.db.name_from_id(card)}')
                        row1 = f1[f1['monster_id'] == card]['box_row'].values[0]
                        row2 = f2[f2['monster_id'] == card]['box_row'].values[0]
                        drop_id = sorted([(row1, fid1), (row2, fid2)])[0][-1]
                        df.loc[((df['source_id'] == drop_id) & 
                                (df['monster_id'] == card)), 'duplicate'] = True
                elif n1 != n2:
                    row1 = list(f1[f1['monster_id'] == card]['box_row'])
                    row2 = list(f2[f2['monster_id'] == card]['box_row'])
                    col1 = list(f1[f1['monster_id'] == card]['box_col'])
                    col2 = list(f2[f2['monster_id'] == card]['box_col'])
                    _fid1 = [fid1 for f in range(len(row1))]
                    _fid2 = [fid2 for f in range(len(row2))]
                    keep = []
                    drop = []
                    for (r, c, fid) in zip(row1+row2, col1+col2, _fid1+_fid2):
                        if not any([abs(r-p[0]) <= 1 and c == p[1] for p in keep]):
                            keep.append((r, c))
                        else:
                            drop.append((r, c))
                            df.loc[((df['source_id'] == fid) &
                                    (df['monster_id'] == card) &
                                    (df['box_row'] == r) & 
                                    (df['box_col'] == c)), 'duplicate'] = True
                else:
                    row1 = f1[f1['monster_id'] == card]['box_row'].values[0]
                    row2 = f2[f2['monster_id'] == card]['box_row'].values[0]
                    drop_id = sorted([(row1, fid1), (row2, fid2)])[0][-1]
                    df.loc[((df['source_id'] == drop_id) & 
                            (df['monster_id'] == card)), 'duplicate'] = True

        self.box = df
    
    # =============================================================================
    # PROPERTIES
    # =============================================================================
    @property
    def box(self) -> pd.DataFrame:
        return self._box
    
    @box.setter
    def box(self, box: pd.DataFrame) -> None:
        if isinstance(box, pd.DataFrame):
            self._box = box
    
    @property
    def size(self) -> int:
        return len(self.box[self.box['duplicate'] == False])
    
    @property
    def columns(self) -> list:
        return list(self.box.columns)
    
    @property
    def unique_cards(self) -> list:
        return list(self.box['monster_id'].unique())
    
    @property
    def num_unique_cards(self) -> list:
        return len(self.unique_cards)
    
    # =============================================================================
    # GETTING MATCH GRID POSITION
    # =============================================================================
    def get_frame(self, source_id: str) -> pd.DataFrame:
        return self.box[self.box['source_id'] == source_id]
    
    def get_grid_size(self, source_id: str) -> tuple:
        num_matches = self.get_frame(source_id)['num_matches'].values[0]
        return (math.ceil(num_matches / 5), 5)
    
    def get_card_center(self, bbox: tuple) -> tuple:
        ((tl_x, tl_y), (br_x, br_y)) = bbox
        return ((tl_x + br_x) / 2, (tl_y + br_y) / 2)
    
    def get_card_positions(self, source_id: str) -> None:
        df = self.get_frame(source_id)
        if len(df['grid_row'].dropna()) != 0:
            return
        nrows, ncols = self.get_grid_size(source_id)
        centers = [self.get_card_center(bbox) for bbox in df['bbox'].values]
        xsort, ysort = sorted(centers), sorted(centers, key=lambda pt: pt[1])
        tl_x, br_x = xsort[0][0], xsort[-1][0]
        tl_y, br_y = ysort[0][1], ysort[-1][1]
        row_positions = list(np.linspace(tl_y, br_y, nrows))
        col_positions = list(np.linspace(tl_x, br_x, ncols))
        for i, card in df.iterrows():
            (x, y) = self.get_card_center(card['bbox'])
            row = row_positions.index(min(row_positions, key=lambda r: abs(r-y)))
            col = col_positions.index(min(col_positions, key=lambda c: abs(c-x)))
            self.box.loc[i, 'grid_row'] = row
            self.box.loc[i, 'grid_col'] = col
            self.box.loc[i, 'box_row'] = card['top_full_row'] + row
            self.box.loc[i, 'box_col'] = col
            
    def get_all_card_positions(self) -> None:
        [self.get_card_positions(src) for src in self.box['source_id'].unique()]
    
    def process_frame(self, source_id: str) -> None:
        mask = (self.box['source_id'] == source_id)
        grid_rows, grid_cols = self.get_grid_size(source_id)
        self.box.loc[mask, 'grid_rows'] = grid_rows
        self.box.loc[mask, 'grid_cols'] = grid_cols
    
    def process_frames(self) -> None:
        for source_id in self.box['source_id'].unique():
            mask = (self.box['source_id'] == source_id)
            grid_size = self.get_grid_size(source_id)
            self.box.loc[mask, 'grid_size'] = self.to_series(grid_size, mask)
            self.get_card_positions(source_id)
            
    # =============================================================================
    # UTILITIES
    # =============================================================================
    @staticmethod
    def to_series(item, mask: pd.Series) -> pd.Series:
        mask = mask[mask==True]
        return pd.Series([item for x in range(mask.sum())], index=mask.index)
    
    def fix_dtypes(self):
        self.box = self.box.convert_dtypes()
        for col in self.box.columns:
            try:
                self.box.loc[:, col] = self.box[col].map(ast.literal_eval)
            except:
                pass
    
    # =============================================================================
    # BUILT-INS
    # =============================================================================
    def __len__(self):
        return self.size
    
    def __iter__(self):
        for col in self.box.columns:
            yield (col, self.box[col].tolist())
            
    def __getitem__(self, i):
        if i in self.box.columns:
            return self.box[i]
    
# =============================================================================
# LOCAL TESTING
# =============================================================================
if __name__ == '__main__':
    def main():
        box = BoxManager()

    main()

