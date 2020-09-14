# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================
import padtools
import requests
import os
from pathlib import Path
import pandas as pd
import re
from PIL import Image
import numpy as np
import json
import cv2

try:
    from .texturetool import TextureWriter, TextureReader
    from .utils import timeit, imread_rgba
except:
    os.chdir(os.path.dirname(__file__))
    from texturetool import TextureWriter, TextureReader
    from utils import timeit, imread_rgba

# =============================================================================
# GLOBALS
# =============================================================================
READ_MODE = cv2.IMREAD_UNCHANGED

# =============================================================================
# ASSET MANAGER
# =============================================================================
class AssetManager():
    def __init__(self, default_server: str = 'NA'):
        self.valid_server_names = ['NA', 'JP', 'KR', 'HK']
        if default_server.upper() not in self.valid_server_names:
            print(f'Server {default_server} not a valid server: using NA')
            default_server = 'NA'
        self.default_server_name = default_server.upper()
        self.server_name = self.default_server_name
        self.startup()
        
    @timeit
    def startup(self):
        # get save locations
        self._get_folders()
        
        # get list of attributes and attribute combinations
        self.attributes = self.attributes()
        self.attribute_combinations = self.attribute_combinations()
        
        # attempt to load existing load asset information
        self.assets = self.load_asset_info()
        
        # load card info
        self.cards = self.load_card_info()
        
        # re-check for existing downloaded assets if asset file isn't found
        if self.assets is None:
            self.assets = self.get_asset_info(update=True)
            self.download_assets(redownload=False, icons_only=False)
            self.convert_assets(reconvert=False, icons_only=False)
        
        print(f'Asset manager startup complete for {self.server_name} server')
        
# =============================================================================
# FETCHING ASSETS
# =============================================================================
    @timeit
    def get_asset_info(self, update: bool = True):
        if self.assets is not None and not update:
            return self.assets
        rows_list = []
        
        # iterate over multiple servers to pull all info
        for server_name in ['NA', 'JP']:
            self.server_name = server_name
            print(f'Getting asset info from {server_name}...')
            server = self.get_server(server_name)
            if server is None:
                continue
            for asset in server.assets:
                asset_dict = {
                    'AssetUrl': asset.url,
                    'AssetId': asset.id_number,
                    'MonsterId': self.get_monster_id(asset.id_number,
                                                     server_name),
                    'RawFilename': asset.file_name,
                    'CompressedSize': asset.compressed_size,
                    'UncompressedSize': asset.uncompressed_size,
                    'AssetType': self.get_asset_type(asset.file_name),
                    'Server': self.server_name,
                    }
                rows_list.append(asset_dict)
            assets = pd.DataFrame(rows_list)
            if self.assets is not None:
                self.assets = pd.concat([self.assets, assets])
            else:
                self.assets = assets
            self.assets.drop_duplicates(subset=['MonsterId', 'Server'], 
                                        keep='last', inplace=True)
        print(f'Found {len(self.assets)} assets')
        self.save_asset_info()
        return self.assets
    
    @timeit
    def download_assets(self, redownload = False, icons_only: bool = False):
        tot = len(self.assets)
        for i, row in self.assets.iterrows():
            url = row['AssetUrl']
            server = row['Server']
            filename = self.get_correct_filename(row['RawFilename'], server)
            asset_type = row['AssetType']
            server = row['Server'].lower()
            
            if icons_only and asset_type != 'icons':
                continue
            
            # check if file has already been downloaded
            if asset_type != 'icons':
                save_path = os.path.join(self.dirs['raw'], filename)
            else:
                save_folder = self.dirs[f'raw_cards_{server}']
                save_path = os.path.join(save_folder, filename)
            self.assets.loc[i, 'RawFilepath'] = save_path
            if os.path.exists(save_path) and not redownload:
                print(f'{filename} already downloaded: skipping...')
                continue
            
            # download asset from url
            print(f'Downloading {filename}: {tot-i} remaining...')
            r = requests.get(url)
            with open(save_path, 'wb') as f:
                f.write(r.content)
            print(f'Saved asset to {save_path}')
            
        self.save_asset_info()

# =============================================================================
# ASSET CONVERSION
# =============================================================================
    def bc_to_png(self, filepath: str, server: str, update: bool = False):
        if not str(filepath).endswith('.bc'):
            return print(f'Filepath \"{filepath}\" does not end with \".bc\"')
            
        filename = os.path.basename(filepath)
        print(f'Converting {filename}...')
        
        tw = TextureWriter()
        tr = TextureReader()
        
        # do not trim card images otherwise portraits get messed up
        trimming = False if 'card' in filename.lower() else True
        tw.enableTrimming(trimming)
        
        # TODO add support for reading .apk zips
        
        with open(filepath, 'rb') as bc:
            contents = bc.read()
        
        textures = list(tr.extractTexturesFromBinaryBlob(contents, filepath))
        num_textures = len(textures)
        
        output_filepaths = []
        
        for texture in textures:
            w, h = texture.width, texture.height
            suggested_name = texture.name
            num = ''.join(n for n in suggested_name if n.isdigit())
            if num and 'card' not in suggested_name.lower():
                monster_id = self.get_monster_id(int(num), server)
                suggested_name = suggested_name.replace(num, str(monster_id))
            output_filepath = self.get_output_filename(suggested_name, server)
            # output_folder = os.path.dirname(output_filepath)
            output_filename = os.path.basename(output_filepath)
            if num_textures > 1:
                print(f'Found {num_textures} textures')
                # input_filename = filename.split('.')[0]
                # output_filename = f'{input_filename}_{output_filename}'
                # output_filepath = os.path.join(output_folder, output_filename)
            print(f'Writing {output_filename} ({w} x {h})')
            output_filepaths.append(output_filepath)
            if os.path.exists(output_filepath) and not update:
                print(f'{output_filepath} already exists')
                continue
            try:
                tw.exportToImageFile(texture, output_filepath)
            except Exception as ex:
                print(f'Could not export to {output_filename}')
                print(f'Error: {ex}')
            
        return output_filepaths
    
    @timeit
    def convert_assets(self, reconvert: bool = False, icons_only: bool = False):
        if 'ConvertedFilepaths' not in self.assets.columns:
            self.assets['ConvertedFilepaths'] = None
            
        if icons_only:
            mask = (self.assets['AssetType'] == 'icons')
            df = self.assets[mask]
        else:
            df = self.assets

        tot = len(df)
        
        for i, row in df.iterrows():
            # check if asset has already been converted
            try:
                already_converted = row['ConvertedFilepaths'] is not None
            except Exception as ex:
                print('Could not check existing conversion:', ex)
                already_converted = False
            if already_converted and not reconvert:
                continue
            
            # convert raw .bc file to .png
            print(f'{tot-i} assets remaining...')
            path = row['RawFilepath']
            server = row['Server']
            output_filepaths = self.bc_to_png(path, server, update=reconvert)
            
            # check if card is animated
            if output_filepaths:
                if len(output_filepaths) > 1 or '000.PNG' in output_filepaths[0]:
                    self.assets.loc[i, 'AnimatedCard'] = True
                else:
                    self.assets.loc[i, 'AnimatedCard'] = False
                
            # store portrait filepaths associated with card
            output_filepaths = ', '.join([p for p in output_filepaths])
            self.assets.loc[i, 'ConvertedFilepaths'] = output_filepaths
            
            # save every 100 files processed
            if (tot-i) % 100 == 0:
                self.save_asset_info()
                
        self.save_asset_info()
        
# =============================================================================
# IMAGE PROCESSING
# =============================================================================
    def fix_portrait_size(self, img = None, img_path = None):
        if img is None and img_path is None:
            return print('Must provide either an image or path to an image')
        
        if img is None:
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGBA')
            except Exception as ex:
                return print(f'Unable to load {img_path}:\n(Error: {ex})')
            
        size = (640, 388)
        size_no_pad = (640 - 70 * 2, 388 - 35 * 2)

        max_size = size_no_pad[0] if img.size[0] > img.size[1] else size_no_pad[1]
            
        img.thumbnail((max_size, max_size), Image.ANTIALIAS)
        
        old_size = img.size
        
        new_img = Image.new('RGBA', size)
        box = (int((size[0] - old_size[0]) / 2), int((size[1] - old_size[1]) / 2))
        new_img.paste(img, box)
        img.close()
        return new_img
    
    @timeit
    def fix_portrait_sizes(self, redo: bool = False):
        # only convert portraits (not animated & not icons)
        mask = ((self.assets['AnimatedCard'] == False) & 
                (self.assets['AssetType'] == 'monster'))
        df = self.assets[mask]
        
        # df = self.assets[self.assets['AnimatedCard'] == False]
        
        tot = len(df)
        for i, row in df.iterrows():
            filepath = row['ConvertedFilepaths']
            if str(filepath) in ['None', 'nan']:
                print('Converted filepaths not found: skipping...')
                continue
            filename = os.path.basename(filepath)
            
            # check if it has already been fixed
            save_path = os.path.join(self.dirs['portraits'], filename)
            self.assets.loc[i, 'PortraitFilepath'] = save_path
            if os.path.exists(save_path) and not redo:
                print(f'{filename} has already been fixed: skipping...')
            
            # fix image dimensions and save to portraits folder
            print(f'Fixing {filename}: {tot-i} remaining...')
            fixed_img = self.fix_portrait_size(img_path=filepath)
            if fixed_img is None:
                print(f'Unable to fix {filename}')
                continue
            fixed_img.save(save_path)
            
        self.save_asset_info()
    
    def generate_icon_borders(self):
        path = os.path.join(self.dirs['card_borders'], 'all_borders.png')
        if not os.path.exists(path):
            return print(f'Icon border file {path} does not exist')
        attr_imgs = {}
        sattr_imgs = {}
        with Image.open(path) as borders_img:
            for i, t in enumerate(self.attributes):
                pw = ph = 100
                xstart, ystart = i * (pw + 2), 0
                xend, yend = xstart + pw, ystart + ph
                attr_imgs[t] = borders_img.crop(box=(xstart, ystart, xend, yend))
                ystart += (ph + 5)
                yend = ystart + ph - 2
                sattr_imgs[t] = borders_img.crop(box=(xstart, ystart, xend, yend))
        # save border images
        for (attr, aimg), (sattr, simg) in zip(attr_imgs.items(), sattr_imgs.items()):
            aimg.save(os.path.join(self.dirs['card_borders'], f'{attr}_main.png'))
            simg.save(os.path.join(self.dirs['card_borders'], f'{sattr}_sub.png'))
            print(f'Saved borders for {attr}')
        return attr_imgs, sattr_imgs
    
    def load_icon_borders(self):
        source_dir = self.dirs['card_borders']
        attr_borders, sattr_borders = {}, {}
        for attr in self.attributes:
            attr_path = os.path.join(source_dir, f'{attr}_main.png')
            sattr_path = os.path.join(source_dir, f'{attr}_sub.png')
            try:
                attr_borders[attr] = Image.open(attr_path)
                sattr_borders[attr] = Image.open(sattr_path)
            except:
                return self.generate_card_icons()
        return attr_borders, sattr_borders
    
    @timeit
    def generate_card_icons(self, update: bool = False):
        # make sure column exists
        if 'CardIconFilepath' not in self.assets.columns:
            self.assets['CardIconFilepath'] = None
            
        # load icon borders
        self.attr_borders, self.sattr_borders = self.load_icon_borders()
            
        tot = len(self.assets)
        
        for i, row in self.assets.iterrows():
            monster_id = row['MonsterId']
            
            # check if card icon already exists
            saved_path = row['CardIconFilepath']
            if saved_path and not update:
                # print(f'Monster ID {monster_id} already saved to {saved_path}')
                continue
            
            # generate card icon
            print(f'Generating icon for {monster_id}: {tot-i} remaining...')
            save_path = self.generate_card_icon(monster_id, update=update)
            self.assets.loc[i, 'CardIconFilepath'] = save_path
            
        self.compile_all_icons()
        self.save_asset_info()
        
    def get_icon_image(self, monster_id: int):
        # different for monster ID > 10000 (NA exclusives)
        if monster_id > 9999:
            monster_id -= 10000
            source_dir = self.dirs['converted_cards_na']
        else:
            source_dir = self.dirs['converted_cards_jp']
        
        # get monster icon
        monster_id -= 1
        
        card_file_idx = int(monster_id / 100) + 1
        sub_idx = monster_id % 100
        col = sub_idx % 10
        row = int(sub_idx / 10)
        card_filename = f'CARDS_{str(card_file_idx).zfill(3)}.PNG'
        card_filepath = os.path.join(source_dir, card_filename)
        
        if not os.path.exists(card_filepath):
            print(card_filepath)
            return print(f'Portraits not found for monster ID {monster_id}')
        
        try:
            with Image.open(card_filepath) as portraits:
                card_dim = 96
                spacer = 6
                xstart = (card_dim + spacer) * col
                ystart = (card_dim + spacer) * row
                xend = xstart + card_dim
                yend = ystart + card_dim
            
                portrait = portraits.crop(box=(xstart, ystart, xend, yend))
                
            if self.is_entirely_transparent(portrait):
                return print(f'Image for {monster_id} is entirely transparent')
                
            return portrait
                
        except OSError:
            return print(f'Unable to load {card_filepath}')
        
    def generate_card_icon(self, monster_id: int, update: bool = False) -> str:
        # check if icon already exists
        test_path = os.path.join(self.dirs['card_icons'], f'{monster_id}.png')
        if os.path.exists(test_path) and not update:
            # print(f'Monster ID {monster_id} icon already exists')
            return test_path
        
        # get card image
        card_img = self.get_icon_image(monster_id)
        if card_img is None:
            return print(f'Could not get icon image for {monster_id}')
        
        # get card attributes
        attr, sattr = self.get_card_attributes(monster_id)
        
        # get main attribute border
        try:
            attr_img = self.attr_borders[attr]
        except:
            self.attr_borders, self.sattr_borders = self.load_icon_borders()
            attr_img = self.attr_borders[attr]
        # else:
        #     return print('Unable to load icon borders')
        
        # adjust to fit portrait
        new_card_img = Image.new('RGBA', attr_img.size)
        new_card_img.paste(card_img, (2, 2))
        
        # add grey background to fill in transparency
        grey_img = Image.new('RGBA', attr_img.size, color=(68, 68, 68, 255))
        card_img = Image.alpha_composite(grey_img, new_card_img)
        
        # merge the attribute border onto the portrait
        merged_img = Image.alpha_composite(new_card_img, attr_img)
        
        # add subattribute border
        if sattr:
            # get subattribute border
            sattr_img = self.sattr_borders[sattr]
            
            # adjust to correct size
            new_sattr_img = Image.new('RGBA', attr_img.size)
            
            # slight offset for subattribute border
            new_sattr_img.paste(sattr_img, (0, 1))
            
            # combine
            merged_img = Image.alpha_composite(merged_img, new_sattr_img)
            
        # save the image
        filename = f'{int(monster_id)}.png'
        save_path = os.path.join(self.dirs['card_icons'], filename)
        merged_img.save(save_path)
        print(f'Saved {filename}')
        
        return save_path
    
    def generate_empty_icon(self, attr: str, sattr: str = None):
        # get single-attribute border
        if sattr is None:
            return self.attr_imgs[attr]
        attr_img = self.attr_imgs[attr]
        sattr_img = self.sattr_imgs[sattr]
        new_sattr_img = Image.new('RGBA', attr_img.size)
        new_sattr_img.paste(sattr_img, (0, 1))
        merged = Image.alpha_composite(attr_img, new_sattr_img)
        name = f'{attr}_main_{sattr}_sub.png'
        save_path = os.path.join(self.dirs['card_borders'], name)
        merged.save(save_path)
        return merged
    
    def generate_empty_icons(self):
        for attr, attr_img in self.attr_imgs.items():
            for sattr, sattr_img in self.sattr_imgs.items():
                self.generate_empty_icon(attr, sattr)
                
    def get_empty_icon(self, attr: str, sattr: str = None):
        name = f'{attr}_main_{sattr}_sub.png' if sattr else f'{attr}_main.png'
        filepath = os.path.join(self.dirs['card_borders'], name)
        if os.path.exists(filepath):
            return imread_rgba(filepath)
        return self.generate_empty_icon(attr, sattr)
    
    @staticmethod
    def load_template(name: str) -> np.ndarray:
        dirs = AssetManager.get_folders()
        filepath = os.path.join(dirs['feature_templates'], f'{name}.png')
        if not os.path.exists(filepath):
            return print(f'{filepath} does not exist')
        try:
            return imread_rgba(filepath)
        except Exception as ex:
            return print(f'Could not load {name} from {filepath}\n{ex}')
    
    @staticmethod
    def load_standard_template():
        dirs = AssetManager.get_folders()
        filepath = os.path.join(dirs['feature_templates'], 'standard_template.png')
        if not os.path.exists(filepath):
            return print(f'{filepath} does not exist')
        try:
            return imread_rgba(filepath)
        except Exception as ex:
            return print(f'Could not load standard template from {filepath}\n{ex}')
    
    @timeit
    def compile_card_icons(self):
        print('\nCompiling all icons...\n')
        icons = {}
        df = self.assets[~self.assets['CardIconFilepath'].isna()]
        
        for i, row in df.iterrows():
            print(f'{len(df) - i} remaining...')
            monster_id = row['MonsterId']
            filepath = row['CardIconFilepath']
            img = imread_rgba(filepath)
            
            # get card name
            name = self.name_from_id(monster_id)
            icons[name] = img
            
        # save icons to npz archive (more secure than pickle)
        save_path = os.path.join(self.dirs['icons'], 'card_icons.npz')
        np.savez_compressed(save_path, **icons)
        print(f'Saved all card icons to {save_path}')
        
        return icons
    
    @timeit
    def load_card_icons(self):
        load_path = os.path.join(self.dirs['icons'], 'card_icons.npz')
        if not os.path.exists(load_path):
            return self.compile_card_icons()
        try:
            icons_npz = np.load(load_path, allow_pickle=False)
            icons = {name: icons_npz[name] for name in icons_npz.files}
            return icons
        except Exception as ex:
            return print(f'Unable to load icons: {ex}')
        
    def load_orb_icons(self):
        folder = self.dirs['card_borders']
        self.orb_icons = {attr: imread_rgba(os.path.join(folder, f'{attr}_orb.png'))
                          for attr in self.attributes}
        return self.orb_icons
    
    @timeit
    def compile_icons_by_attributes(self):
        print('\nCompiling icons by attributes...\n')
        cards = self.get_cards_by_attributes(require_icon=True)
        icons = self.load_card_icons()
        icon_h, icon_w = np.array(list(icons.values())[0]).shape[0:2]
        
        def stitch_icons(attributes: str):
            card_list = cards[attributes]
            card_icons = [(name, icons[name]) for name in card_list]
            
            # get size of template to use
            num_icons = len(card_icons)
            num_cols = int(np.sqrt(num_icons))
            num_rows = -(-num_icons // num_cols)
            canv_w = icon_w * num_cols
            canv_h = icon_h * num_rows
            canv_size = (canv_w, canv_h)
            
            # create template with stored position identifiers
            icons = Image.new('RGB', canv_size, color=(255, 255, 255))
            bboxes = {}
            for col in range(num_cols):
                for row in range(num_rows):
                    # get bounding box
                    x1 = col * icon_w
                    x2 = x1 + icon_w
                    y1 = row * icon_h
                    y2 = y1 + icon_h
                    bbox = (x1, y1, x2, y2)
                    
                    # get icon and name
                    if len(card_icons) == 0:
                        break
                    name, icon = card_icons.pop(0)
                    bboxes[name] = bbox
                    
                    # paste the icon WITHOUT the alpha channel
                    icons.paste(icon, bbox, mask=icon.split()[3])
                   
            # save the canvas
            save_dir = os.path.join(self.dirs['icon_templates'])
            save_path = os.path.join(save_dir, f'{attributes}_icons.png')
            icons.save(save_path)
            print(f'Saved compiled icons for {attributes}')
            
            # save the labels
            with open(os.path.join(save_dir, f'{attributes}_labels.json'), 'w') as f:
                json.dump(bboxes, f)
            print(f'Saved labels for {attributes}')
                   
            return {'icons': icons, 'labels': bboxes}
        
        icon_info = {}
        for i, attributes in enumerate(cards.keys()):
            print(f'{len(cards) - i} remaining...')
            icon_info[attributes] = stitch_icons(attributes)
            
        return icon_info
    
    @timeit
    def get_icons_by_attributes(self):
        src = self.dirs['icon_templates']
        attrs = self.attribute_combinations
                    
        icon_info = {}
        for attr in attrs:
            if attr not in icon_info.keys():
                icon_info[attr] = {}
                
            image_path = os.path.join(src, f'{attr}_icons.png')
            labels_path = os.path.join(src, f'{attr}_labels.json')
            
            # load the image
            try:
                icon_info[attr]['icons'] = imread_rgba(image_path)
            except:
                icon_info[attr]['icons'] = None
                
            # load the labels
            try:
                with open(labels_path, 'r') as f:
                    icon_info[attr]['labels'] = json.load(f)
            except:
                icon_info[attr]['labels'] = None
                
        # make sure everything was loaded
        for attr, data in icon_info.items():
            if any(v is None for v in data.values()):
                print(f'Attribute {attr} is missing data')
                print('Attempting to compile icons by attribute...')
                try:
                    return self.compile_icons_by_attributes()
                except Exception as ex:
                    return print(f'Unable to compile icons: {ex}')
                
        return icon_info
        
# =============================================================================
# UTILITIES
# =============================================================================
    def get_server(self, server_name: str):
        if server_name.upper() == 'NA':
            return padtools.regions.north_america.server
        elif server_name.upper() == 'JP':
            return padtools.regions.japan.server
        elif server_name.upper() == 'KR':
            return padtools.regions.korea.server
        elif server_name.upper() == 'HK':
            return padtools.regions.hong_kong.server
        else:
            return print(f'Server {server_name} not a valid server')
        
    @staticmethod
    def get_folders():
        # set working directory and its parent
        cur = os.path.dirname(__file__)
        os.chdir(cur)
        par = Path(cur).parent
        
        # make the save directories
        dirs = {
            'assets': os.path.join(par, 'assets', ''),
            'matching': os.path.join(par, 'assets', 'matching', ''),
            'icon_templates': os.path.join(par, 'assets', 'matching', 
                                           'icon_templates', ''),
            'feature_templates': os.path.join(par, 'assets', 'matching',
                                              'feature_templates', ''),
            'screenshots': os.path.join(par, 'assets', 'screenshots', ''),
            'raw': os.path.join(par, 'assets', 'raw', ''),
            'raw_cards_na': os.path.join(par, 'assets', 'raw', 'cards', 'na', ''),
            'raw_cards_jp': os.path.join(par, 'assets', 'raw', 'cards', 'jp', ''),
            'converted': os.path.join(par, 'assets', 'converted', ''),
            'converted_cards_na': os.path.join(par, 'assets', 'converted', 
                                               'cards', 'na', ''),
            'converted_cards_jp': os.path.join(par, 'assets', 'converted', 
                                                'cards', 'jp', ''),
            'portraits': os.path.join(par, 'assets', 'portraits', ''),
            'icons': os.path.join(par, 'assets', 'icons', ''),
            'card_icons': os.path.join(par, 'assets', 'icons', 'cards', ''),
            'card_borders': os.path.join(par, 'assets', 'icons', 'borders', ''),
            'databases': os.path.join(par, 'databases', ''),
            }
        [os.makedirs(d) for d in dirs.values() if not os.path.exists(d)]
        
        return dirs
        
    def _get_folders(self):
        self.dirs = self.get_folders()
        return self.dirs

    def get_asset_type(self, filename: str):
        if 'cards' in filename.lower():
            return 'icons'
        elif 'mons' in filename.lower():
            return 'monster'
            
    def get_output_filename(self, suggested_name: str, server: str):
        output_filename = suggested_name
        try:
            prefix, monster_id, suffix = re.match(r'^(MONS_)(\d+)(\..+)$', 
                                                suggested_name, flags=re.I).groups()
            output_filename = f'{prefix}{monster_id.zfill(5)}{suffix}'
        except AttributeError:
            pass
        output_filename = output_filename.replace('.bc', '.png')
        
        output_folder = self.dirs['converted']
        if 'card' in suggested_name.lower():
            server = server.lower()
            if server == 'na':
                output_folder = self.dirs[f'converted_cards_{server}']
            elif server == 'jp':
                output_folder = self.dirs[f'converted_cards_{server}']
            
        output_filepath = os.path.join(output_folder, output_filename)
        if os.path.exists(output_filename):
            ext = output_filepath.split('.')[-1]
            output_filepath = output_filepath.split('.')[0] + f'Copy.{ext}'
        return output_filepath
    
    def is_entirely_transparent(self, image):
        return image.getextrema() == ((0, 0), (0, 0), (0, 0), (0, 0))
    
    @staticmethod
    def attr_id_to_str(attr):
        attrs = {
            0: 'r',
            1: 'b',
            2: 'g',
            3: 'l',
            4: 'd',
            }
        return attrs.get(attr, '')
    
    def get_card_attributes(self, monster_id: int):
        if monster_id not in self.cards['monster_id'].values:
            print(f'Monster ID {monster_id} not found in card database')
            return None, None
        
        info = self.cards[self.cards['monster_id'] == monster_id]
        main = info['attribute_1_id'].values[0]
        sub = info['attribute_2_id'].values[0]
        
        return (self.attr_id_to_str(main), self.attr_id_to_str(sub))
    
    @timeit
    def get_cards_by_attributes(self, require_icon: bool = True, icons: dict = {}):
        # return the cards if this function has alreaby been run
        try:
            if require_icon:
                return self.cards_by_attributes_icons
            else:
                return self.cards_by_attributes_no_icons
        except:
            pass
        
        if require_icon and not icons:
            icons = self.load_card_icons()
            if not icons:
                print('No icons found')
                return icons
        
        # get cards grouped by main and sub-attribute
        df = self.cards.copy()
        groups = df.groupby(by=['attribute_1_id', 'attribute_2_id'], dropna=False)
        cards = {}
        for (attr, sattr), group in groups:
            main, sub = self.attr_id_to_str(attr), self.attr_id_to_str(sattr)
            cards[f'{main}{sub}'] = []
            for i, row in group.iterrows():
                monster_id = row['monster_id']
                card_name = self.name_from_id(monster_id)
                if require_icon and card_name not in icons.keys():
                    continue
                cards[f'{main}{sub}'].append(card_name)
                
        if require_icon:
            self.cards_by_attributes_icons = cards
            # save the result so it can be loaded later
            filepath = os.path.join(self.dirs['icons'], 'cards_by_attributes_icons.json')
            with open(filepath, 'w') as f:
                json.dump(cards, f)
            
        else:
            self.cards_by_attributes_no_icons = cards
            filepath = os.path.join(self.dirs['icons'], 'cards_by_attributes_all.json')
            with open(filepath, 'w') as f:
                json.dump(cards, f)
            
        return cards
    
    def load_cards_by_attributes(self, require_icon: bool = True):
        load_dir = self.dirs['icons']
        if require_icon:
            filename = 'cards_by_attributes_icons.json'
        else:
            filename = 'cards_by_attributes_all.json'
        filepath = os.path.join(load_dir, filename)
        if not os.path.exists(filepath):
            print('Saved card list does not exist: re-creating...')
            return self.get_cards_by_attributes()
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data            
    
    def name_from_id(self, monster_id: int):
        if monster_id not in self.cards['monster_id'].values:
            print(f'Monster ID {monster_id} not found in database')
            return monster_id
        mask = (self.cards['monster_id'] == monster_id)
        name_na = self.cards[mask]['name_na'].values[0]
        override_name = self.cards[mask]['name_na_override'].values[0]
        if str(override_name) not in ['nan', 'None']:
            name_na = override_name
        return name_na
    
    def get_asset_id(self, monster_id: int):
        if monster_id not in self.cards['monster_id']:
            return print(f'Monster ID {monster_id} does not exist')
        
        # cards released in NA but not JP (e.g. Voltron) have a 1 in front of ID
        if self.default_server_name == 'NA':
            mask = ((self.cards['monster_id'] == monster_id) &
                    (self.cards['on_na'] == 1) &
                    (self.cards['on_jp'] == 0))
        else:
            mask = (self.cards['monster_id'] == monster_id)
            
        return self.cards[mask]['monster_no_jp'].values[0]
    
    def get_monster_id(self, asset_id: int, server: str = 'NA'):
        if asset_id not in self.cards['monster_no_jp']:
            return print(f'Asset ID {asset_id} does not exist')
        
        # cards released in NA but not JP (e.g. Voltron) have a 1 in front of ID
        mask = (self.cards['monster_no_jp'] == asset_id)
        results = self.cards[mask]
        
        # if there are multiple matches, NA will have the higher value
        # (e.g. Zordon, asset_id = 4982 -> monster_id = 14982)
        if server == 'NA':
            return int(results['monster_id'].max())
        else:
            return int(results['monster_id'].min())
    
    def get_correct_filename(self, filename: str, server: str):
        match = re.search(r'(\d+)', filename)
        num_str = match.groups()[0] if match is not None else None
        if num_str is None:
            return filename
        monster_id = self.get_monster_id(int(num_str), server)
        return filename.replace(num_str, str(monster_id).zfill(len(num_str)))
    
    @staticmethod
    def attributes():
        return ['r', 'b', 'g', 'l', 'd']
    
    @staticmethod
    def attribute_combinations():
        attrs = AssetManager.attributes()
        return attrs + list(set([m + s for m in attrs for s in attrs]))
    
# =============================================================================
# SAVING AND LOADING DATAFRAMES
# =============================================================================
    def load_asset_info(self):
        load_path = os.path.join(self.dirs['databases'], 'asset_info.feather')
        try:
            return pd.read_feather(load_path)
        except OSError:
            return print('Asset info file not found')

    def save_asset_info(self):
        save_path = os.path.join(self.dirs['databases'], 'asset_info.feather')
        # reset index since feather can only save default indices
        self.assets.reset_index(inplace=True, drop=True)
        self.assets.to_feather(save_path)
        print('Saved asset info')
        
    def load_card_info(self):
        load_path = os.path.join(self.dirs['databases'], 'monsters.feather')
        try:
            return pd.read_feather(load_path)
        except OSError:
            return print('Card info file not found')
        
# =============================================================================
# BATCH OPERATIONS
# =============================================================================
    def refresh_all_data(self):
        self.get_asset_info(update=True)
        self.download_assets(redownload=True, icons_only=False)
        self.convert_assets(reconvert=True, icons_only=False)
        self.generate_icon_borders()
        self.fix_portrait_sizes(redo=True)
        self.generate_card_icons(update=True)
        self.compile_icons_by_attributes()
        self.save_asset_info()
        
# =============================================================================
# TESTING    
# =============================================================================
    def test(self):
        # self.get_asset_info(update=True)
        # self.download_assets(redownload=True, icons_only=True)
        # self.convert_assets(reconvert=True, icons_only=False)
        # self.generate_card_icons(update=True)
        # self.fix_portrait_sizes(redo=True)
        # self.compile_icons_by_attributes()
        # self.generate_card_icons(update=True)
        self.compile_card_icons()
        # icons = self.load_card_icons()
        # self.load_cards_by_attributes(require_icon=True)
        # self.generate_icon_borders()
        
        # self.refresh_all_data()
        
# =============================================================================
# RUNNING DIRECTLY
# =============================================================================
if __name__ == '__main__':
    am = AssetManager()
    am.test()
    am.save_asset_info()
    