# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from scipy.ndimage import gaussian_filter
import concurrent.futures
import math
# from threading import Thread
# from queue import Queue
# import asyncio
# import nest_asyncio
# nest_asyncio.apply()

try:
    from .assetmanager import AssetManager
    from .utils import timeit, pt_sep, imread_rgba, imread_rgb
except:
    os.chdir(os.path.dirname(__file__))
    from assetmanager import AssetManager
    from utils import timeit, pt_sep, imread_rgba, imread_rgb

# =============================================================================
# LOGGING
# =============================================================================
# import logging
# logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))

# =============================================================================
# GLOBALS
# =============================================================================
METHODS = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
# Notes on methods:
# - TM_SQDIFF requires the minimum value instead of maximum
# - TM_CCORR is fast but gives more false identifications
# - TM_CCOEFF is overall the best but is slow

# based on tests, cv2.TM_CCOEFF_NORMED is indeed the most accurate
PREFERRED_METHOD = cv2.TM_CCOEFF_NORMED

# cv2.INTER_LANCZOS4 is the most accurate interpolation
INTERPOLATION = cv2.INTER_LANCZOS4 # cv2.INTER_CUBIC

LINEWIDTH = 5
OFFSET_LINEWIDTH = True
LINECOLOR = (0, 255, 255)
DEFAULT_SCALE = 1.0
READ_MODE = cv2.IMREAD_UNCHANGED
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 2.5
SLEEP = 0.0001
DEFAULT_SCREENSHOT_WIDTH = 800 # min. size to detect orbs consistently // 750
DEFAULT_SCALE_WIDTH = DEFAULT_SCREENSHOT_WIDTH
NO_PEEK = False
DEBUG = False
# DEBUG = True

# =============================================================================
# MAIN MATCHING CLASS
# =============================================================================
class Matcher():
    def __init__(
            self, 
            input_images: list,
            default_scale: float = None,
            orb_crop_factor: float = 0.15, # 0.15 default
            **kwargs
            ) -> dict:
        
        # store original inputs so we know when they've been identified
        self.input_dict = {idx: i for idx, i in enumerate(input_images)}

        # store input variables
        self.default_scale = default_scale
        self.orb_crop_factor = orb_crop_factor
        
        # store empty variables
        self.rgb_input_images = []
        self.matched_images = []
        self.matched_cards = []
        self.input_videos = {}
        self.box_bottom_boundary = 1
        self.box_top_boundary = 0
        
        # parse the input images
        self.rgb_input_images = self.add_inputs(input_images)
        self.num_input_images = len(self.rgb_input_images)
        if self.num_input_images == 0 and len(self.input_videos) == 0:
            return print('No input images provided')
        print(f'Got {self.num_input_images} input images',
              f'and {len(self.input_videos)} videos')
        
        # initialize the asset manager and load required assets
        self._load_required_assets()
        
    def add_inputs(self, inputs: list):
        inputs = [inputs] if not isinstance(inputs, list) else inputs
        vid_exts = ('.mp4', 'mp3', '.gif', '.mkv', '.mov', '.webm', '.avi', '.ogg')
        for item in inputs:
            if isinstance(item, str):
                if not os.path.exists(item):
                    print(f'{item} does not exist')
                    continue
                elif item.lower().endswith(vid_exts):
                    self.input_videos[item] = None
                    continue
                img = imread_rgb(item)
                # TODO: handle URLs
            img = to_numpy(img)
            self.rgb_input_images.append(img) if img is not None else None
        return self.rgb_input_images
    
    # =============================================================================
    # STATIC METHODS
    # =============================================================================
    @staticmethod
    def standardize_input_image(image: np.ndarray, rgb: bool = True) -> np.ndarray:
        image = to_numpy(image)
        rgb = to_rgb(image) if not rgb else image
        h, w = image.shape[0:2]
        scale = min(1.0, DEFAULT_SCREENSHOT_WIDTH / w)
        return rescale(rgb, scale)
    
    @staticmethod
    def crop_orb(orb: np.ndarray, crop_factor: float = 0.15) -> np.ndarray:
        if crop_factor == 0.0:
            return orb
        h, w = orb.shape[0:2]
        dh, dw = int(crop_factor * h), int(crop_factor * w)
        return orb[dh:min(-dh, -1), dw:min(-dw, -1)]
    
    @staticmethod
    def crop_icon(icon: np.ndarray, offsets: tuple) -> np.ndarray:
        (dw0, dw1), (dh0, dh1) = offsets
        return icon[dh0:-dh1, dw0:-dw1]
    
    @staticmethod
    def standardize_template(template: np.ndarray, scale: float) -> np.ndarray:
        template = to_gray(template)
        temp_w, temp_h = template.shape[1], template.shape[0]
        dsize = (int(scale * temp_w), int(scale * temp_h))
        if scale != 1.0:
            template = cv2.resize(template, dsize=dsize, interpolation=INTERPOLATION)
        template = gauss_sharpen(template)    
        return template
    
    @staticmethod
    def get_matched_cards(match_list: list):
        if isinstance(match_list, dict):
            match_list = [match_list]
        matches = []
        for match_info in match_list:
            if not isinstance(match_info, dict):
                continue
            for match in match_info.get('matched_cards', []):
                matches.append(match.get('name', None))
        print(f'Total number of matches: {len(matches)}')
        return matches
    
    @staticmethod
    def get_unmatched_regions(match_list: list):
        if isinstance(match_list, dict):
            match_list = [match_list]
        unmatched = []
        for match_info in match_list:
            if not isinstance(match_info, dict):
                continue
            for attr, region in match_info.get('unmatched_cards', {}).items():
                # for bbox, region in regions.items():
                unmatched.append((attr, region))
        print(f'Number of unmatched regions: {len(unmatched)}')
        return unmatched
    
    @staticmethod
    def is_cutoff(icon: np.ndarray):
        short, long = sorted(list(icon.shape[0:2]))
        try: return abs(short / long) < 0.9
        except: return False
    
    # =============================================================================
    # CLASS METHODS
    # =============================================================================
    @classmethod
    def trim_image(
            cls,
            image: np.ndarray,
            top: int = 0,
            bottom: int = 1,
            left: int = 0,
            right: int = 1
            ) -> np.ndarray:
        try:
            return image[top:-bottom, left:-right]
        except:
            print(f'Could not trim image using {(top, bottom, left, right)}')
            return image
        
    # =============================================================================
    # IMAGE UTILITIES
    # =============================================================================
    def trim_to_box(self, image: np.ndarray) -> np.ndarray:
        return self.trim_image(image, top = self.box_top_boundary, 
                               bottom = self.box_bottom_boundary)
        
    # =============================================================================
    # PROTECTED FUNCTIONS
    # =============================================================================
    @timeit
    def _load_required_assets(self):
        self._load_asset_manager()
        fs = (self._load_card_icons, self._load_orb_icons, self._load_bottom_template,
              self._load_standard_template, self._load_scrollbar_template, 
              self._load_cards_by_attributes)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            [pool.submit(f) for f in fs]
            
    def _load_asset_manager(self):
        self.asset_manager = AssetManager()
        
    def _load_card_icons(self):
        self.card_icons = self.asset_manager.load_card_icons()
        
    def _load_orb_icons(self):
        self.orb_icons = self.asset_manager.load_orb_icons()
        
    def _load_standard_template(self):
        template = self.asset_manager.load_standard_template()
        self.standard_template = self.standardize_template(template, 1.0)
        
    def _load_bottom_template(self):
        template = self.asset_manager.load_template('cards_text_template')
        self.cards_text_template = self.standardize_template(template, 1.0)
        
    def _load_scrollbar_template(self):
        template = self.asset_manager.load_template('scrollbar_template')
        # load scrollbar as grayscale instead of standard template processing
        self.scrollbar_template = to_gray(template)
        
    def _load_cards_by_attributes(self):
        self.cards_by_attributes = self.asset_manager.load_cards_by_attributes()
        
    # =============================================================================
    # SUB-ROUTINES
    # =============================================================================
    @timeit
    def find_bottom_boundary(
            self,
            input_image: np.ndarray,
            scale: float
            ) -> int:
        
        # get input dimensions
        im_h, im_w = input_image.shape[0:2]
        
        template = self.cards_text_template
        if template is None:
            return print('Unable to get bottom boundary: missing template')
        
        # resize the template based on the known scaling factor
        template = rescale(template, (100/164)*scale)
        temp_h, temp_w = template.shape[0:2]
        
        # get match position
        match = find_matches(input_image, template)
        if not match['matched']:
            print('Unable to get bottom boundary: no match found')
            return 1
        
        # get position of bottom boundary
        boundary = im_h - (match['boxes'][0][1] + temp_h)
        match['boundary'] = boundary
        match['temp_w'] = temp_w
        match['temp_h'] = temp_h
        print(f'Bottom boundary: {boundary}')
        return match
    
    @timeit
    def get_scrollbar_info(
            self,
            input_image: np.ndarray,
            scale: float,
            scroll_bbox: tuple = None,
            ) -> tuple:
        
        # skip matching if the region is already known
        if scroll_bbox is not None:
            (tl_x, tl_y), (br_x, br_y) = scroll_bbox
        
        else:
            ctxt_h, ctxt_w = self.cards_text_template.shape[0:2]
            ctxt_h, ctxt_w = ctxt_h*scale*(100/164), ctxt_w*scale*(100/164)
            
            # get scrollbar boundary using some hacky relative position bullshit
            _info = self.bottom_boundary_info
            _tl_x = _info['boxes'][0][0]
            _tl_y = _info['boxes'][0][1]
            br_x = int(_tl_x + (298/140)*ctxt_w)
            br_y = int(_tl_y - (10/52)*ctxt_h)
            tl_x = int(br_x - (41/140)*ctxt_w)
            tl_y = int(self.box_top_boundary + (136/52)*ctxt_h)
        
        # # get the scrollbar template
        # template = self.scrollbar_template
        # if template is None:
        #     return print('Unable to get scrollbar position: missing template')
        
        # # convert inputs to grayscale
        # template = to_gray(template)
        # input_image = to_gray(input_image)
        
        # # resize template based on known scaling factor
        # template = rescale(template, (100/164)*scale) # 100 / 164
        # temp_h, temp_w = template.shape[0:2]
        
        # # skip matching if the region is already known
        # if scroll_bbox is not None:
        #     (tl_x, tl_y), (br_x, br_y) = scroll_bbox

        # else:        
        #     # get match position
        #     match = find_matches(input_image, template)
        #     match['name'] = 'scrollbar'
        #     if not match['matched']:
        #         return print('Unable to get scrollbar position: no match found')
            
        #     # show the bounding region
        #     box = match['boxes'][0]
        #     tl_x, tl_y = box
        #     br_x, br_y = tl_x + temp_w, tl_y + temp_h
        #     match['bbox'] = (tl_x, tl_y), (br_x, br_y)
        
        # show the region
        scroll_region = input_image[tl_y:br_y, tl_x:br_x]
        # self.show_matches(input_image, template, match, show_full_box=False, save_heatmap=True)
        
        # analyze the scrollbar region
        scroll_info = self.analyze_scrollbar(scroll_region)
        scroll_info['bbox'] = (tl_x, tl_y), (br_x, br_y)
        # scroll_info.update(match)
        return scroll_info
        
    def analyze_scrollbar(
            self,
            scrollbar: np.ndarray,
            ) -> int:
        
        # isolate scrollbar from background via gaussian blur subtraction
        no_bg = scrollbar - gaussian_filter(scrollbar, 25)
        
        # get top/bottom of whole scrollbar
        coords = np.where(no_bg < 175)
        scroll_top, scroll_bot = max(coords[0]), min(coords[0])
        
        # trim input image to just the scrollbar
        trimmed = scrollbar[scroll_bot:scroll_top, :]
        
        # peek(trimmed)
        
        # get scrollbar dimensions
        scroll_h, scroll_w = scrollbar.shape[0:2]
        
        # get coordinates of handle pixels (white pixels > 175)
        # this value is based on a histogram of the standard template
        active = np.where(trimmed > 175)
        
        # get position of the middle of the handle relative to scrollbar
        h_vals = active[0]
        
        # skip if scrollbar is not found
        if len(h_vals) == 0:
            print('Scrollbar handle not found')
            # peek(no_bg)
            # peek(scrollbar)
            # peek(trimmed)
            # peek(np.where(trimmed > 175, 255, 0))
            return {'scroll_h': scroll_h, 'scroll_w': scroll_w, 'trimmed': trimmed,
                    'matched': False}
        
        bot, top = max(h_vals), min(h_vals)
            
        mid = (bot + top) / 2
        handle_height = abs(top - bot)
        
        # get approximate page that the current screen is on
        # rows per page: 46 rows in a box -> 164px handle, 1141px scroll
        # this is true for the Galaxy S10+, will not be exact for others
        rows_per_page = (scroll_h / self.icon_h) * 0.8575
        total_pages = scroll_h / handle_height
        total_rows = int(total_pages * rows_per_page)
        
        # get approximate page number
        page_num = bot / handle_height
        
        # get index of top and bottom card row (one-indexed)
        top_partial_row = (rows_per_page * page_num) - (rows_per_page - 1)
        bot_partial_row = (rows_per_page * page_num)
        
        # get the top / bottom FULL row
        top_full_row = math.ceil(top_partial_row - 0.25)
        bot_full_row = math.floor(bot_partial_row - 0.25)
        num_full_rows = abs(top_full_row - bot_full_row) + 1
        
        # calculate minimum number of cards in box: 
        min_cards = 4 + 5*(total_rows - 2) + 1
        max_cards = min_cards + 5
        num_cards = (min_cards, max_cards)
        
        print(f'Scrollbar height: {scroll_h}')
        print(f'Handle height: {handle_height}')
        print(f'Estimated total pages: {total_pages}')
        print(f'Estimated page #: {page_num}')
        print(f'Estimated number of rows: {total_rows}')
        print(f'Estimated cards in box: {num_cards}')
        print(f'Top partial row: {top_partial_row}')
        print(f'Bottom partial row: {bot_partial_row}')
        print(f'Top full row: {top_full_row}')
        print(f'Bottom full row: {bot_full_row}')
        
        scrollbar_info = {
            'matched': True,
            # 'trimmed': trimmed,
            'scroll_w': scroll_w,
            'scroll_h': scroll_h,
            'top': top,
            'mid': mid,
            'bot': bot,
            'handle_height': handle_height,
            'rows_per_page': rows_per_page,
            'total_pages': total_pages,
            'total_rows': total_rows,
            'page_num': page_num,
            'top_partial_row': top_partial_row,
            'bot_partial_row': bot_partial_row,
            'top_full_row': top_full_row,
            'bot_full_row': bot_full_row,
            'row_span': (top_full_row, bot_full_row),
            'num_full_rows': abs(top_full_row - bot_full_row) + 1,
            'included_rows': list(range(top_full_row, bot_full_row + 1)),
            'min_cards': min_cards,
            'max_cards': max_cards,
            'max_possible_rows': math.floor(rows_per_page) == num_full_rows,
            'actual_page': (bot_full_row) / math.floor(rows_per_page)
            }
        
        return scrollbar_info
    
    def find_orbs(
            self,
            input_image: np.ndarray, 
            orb: np.ndarray, 
            color: str, 
            ) -> tuple:
        
        # crop the orb
        orb = self.crop_orb(orb, self.orb_crop_factor)
        dh = self.orb_crop_factor * orb.shape[0]
        dw = self.orb_crop_factor * orb.shape[1]
        offsets = (dw, dw), (dh, dh)
        
        # find matches for the orb in the input image
        matches = find_matches(input_image, orb)
        matches['name'] = color
        matches['offests'] = offsets
        return (color, matches)
    
    def count_orbs(
            self, 
            region: np.ndarray, 
            orbs: dict = None,
            scale: float = None,
            standardize: bool = True
            ) -> bool:
        
        # check if the region contains an orb
        scale = self.default_scale if scale is None else scale
        if orbs is None:
            orbs = {k: self.standardize_template(v, scale) 
                    for k, v in self.orb_icons.items()}
        if standardize:
            region = self.standardize_template(region, scale)
        orb_count = 0
        for _, orb in orbs.items():
            cropped = self.crop_orb(orb, self.orb_crop_factor)
            matches = find_matches(cropped, orb)
            matched = matches.get('matched', False)
            orb_count += int(matched)
        return orb_count
    
    @staticmethod
    def show_matches(
            input_image: np.ndarray, 
            template: np.ndarray, 
            matches: dict,
            show_cropped: bool = True,
            save_heatmap: bool = False
            ):
        
        # make a copy of the image to avoid overwriting anything else
        img = input_image.copy()
        
        # get values
        name = matches.get('name', 'matched boxes')
        offsets = matches.get('offsets', ((0, 0), (0, 0)))
        
        def label_img(img, start, end):
            cv2.rectangle(img, start, end, LINECOLOR, LINEWIDTH)
            st = start[0] - 2*LINEWIDTH, start[1] - 2*LINEWIDTH
            cv2.putText(img, name, st, FONT, FONT_SCALE, 
                        (0, 255, 255), 3, cv2.LINE_AA)
        
        # check if bounding box already exists
        (dw0, dw1), (dh0, dh1) = offsets
        lw = LINEWIDTH if OFFSET_LINEWIDTH else 0
        h, w = template.shape[0:2]
        match_boxes = []
        if 'bbox' in matches.keys():
            start, end = matches['bbox']
            start = (start[0] - lw), (start[1] + lw)
            end = (end[0] - lw), (end[1] + lw)
            match_boxes.append((start, end))
        
        # otherwise, create bounding boxes for display
        else:
            boxes = matches['boxes']
            for box in boxes:
                # expand the box to account for line width and initial crop
                start = (box[0] - lw, box[1] - lw)
                end = (box[0] + w + lw, box[1] + h + lw)
                match_boxes.append((start, end))
        
        # expand box
        for (start, end) in match_boxes:
            if show_cropped:
                start = (start[0] + dw0, start[1] + dh0)
                end = (end[0] - dw1, end[1] - dh1)
            label_img(img, start, end)
        
        # save result image
        match_path = os.path.abspath(f'./tests/{name}.png')
        save_rgb(img, match_path)
            
        # save heatmap
        confidence_map = matches.get('confidence_map', None)
        if save_heatmap and confidence_map is not None:
            heatmap_path = os.path.abspath(f'./tests/{name}_map.png')
            normalized = confidence_map*(255*np.amax(confidence_map))
            save_rgb(normalized, heatmap_path)

    def identify_region(
            self,
            input_image: np.ndarray,
            orig_img: np.ndarray,
            attributes: str,
            bboxes: tuple,
            icons: dict, 
            scale: float,
            offsets: tuple,
            crop: bool = True
            ):
        
        if len(bboxes) == 0:
            print(f'\nNo {attributes} regions identified...\n')
            return [], {}, {}
        
        print(f'\nIdentifying {attributes} region...\n')
        
        # get list of cards to search over
        card_list = self.cards_by_attributes[attributes]
        
        # extract all the card regions
        regions = {bbox: extract_region(input_image, bbox)
                   for bbox in bboxes}
        
        # get RGB icons for reviewing later
        regions_rgb = {bbox: extract_region(orig_img, bbox)
                       for bbox in bboxes}
        
        # match all the regions
        matched_cards = []
        unmatched_regions = regions.copy()
        # compare provided region against all icons with the same attributes
        for card_name in card_list:
            # prepare card icon
            card_icon = icons.get(card_name, None)
            if card_icon is None:
                continue
            card_icon = self.standardize_template(card_icon, scale)
            card_icon = self.crop_icon(card_icon, offsets) if crop else card_icon
            
            # check for a match against all regions
            for (bbox, region) in regions.items():
                # return if all regions have been matched
                if len(unmatched_regions) == 0:
                    return matched_cards, unmatched_regions, {}
                
                # make sure region hasn't already been matched
                if bbox not in unmatched_regions.keys():
                    continue
                
                # check for match
                result = find_matches(region, card_icon)
                
                # don't store info for non-matches
                if not result['matched']:
                    continue
                
                # if attributes == 'gr':
                #     Image.fromarray(region).show()
                #     Image.fromarray(card_icon).show()
                
                # store match information
                # concurrent.futures has trouble with special characters
                result['name'] = card_name
                result['bbox'] = bbox
                result['offsets'] = offsets
                matched_cards.append(result)
                unmatched_regions.pop(bbox)
                print(f'Matched {card_name}')
                
                # show the matched region
                self.show_matches(orig_img, card_icon, result, 
                                  show_cropped=False, save_heatmap=False)
        
        if not matched_cards:
            print(f'No match found for predicted {attributes} card')
            
        # fixed unmatched regions
        unmatched_regions_rgb = {bbox: region for bbox, region in regions_rgb.items()
                                 if bbox in unmatched_regions.keys()}
            
        return matched_cards, unmatched_regions, unmatched_regions_rgb

    # =============================================================================
    # GETTING BEST VIDEO FRAMES
    # =============================================================================
    @timeit
    def identify_video(
            self,
            video_filepath: str,
            ) -> list:
        
        # load video info
        frames = self.input_videos.get(video_filepath, None)
        if frames is None:
            frames = load_video_frames(video_filepath)
            self.input_videos[video_filepath] = frames
            if frames is None:
                return
        
        num_frames = frames.shape[0]
        print(f'Number of frames: {num_frames}')
        
        # identify the first frame to get info
        frame_info = {}
        first_match = None
        for i in range(num_frames):
            first_match = self.identify_image(frames[i])
            if first_match is not None:
                frame_info[i] = first_match
                break
        
        # get scrollbar info for each frame
        scale = first_match.get('scale', self.default_scale)
        scroll_bbox = first_match.get('bbox', None)
        scroll_info = {}
        
        for i in range(num_frames):
            print(f'Getting scrollbar info: {num_frames-i} remaining...')
            frame = self.standardize_input_image(frames[i])
            scroll_info[i] = self.get_scrollbar_info(frame, scale, scroll_bbox)
            
        # remove rows with no data
        df = pd.DataFrame.from_dict(scroll_info, orient='index')
        df = df[df['matched'] == True]
        
        # remove outliers
        med, std = df['handle_height'].median(), df['handle_height'].std()
        mask = ((abs(df['handle_height'] - med) < 5) | 
                (df['handle_height'] > med - 0.3*std) & 
                (df['handle_height'] < med + 0.3*std))
        df = df[mask]
        
        # sort by top full row
        df.sort_values(by=['top_full_row'], ascending=True, inplace=True)
        
        # drop duplicates
        df.drop_duplicates(subset='row_span', inplace=True, keep='first')
        
        # remove more duplicates
        df.sort_values(by=['actual_page', 'num_full_rows'], inplace=True, ascending=True)
        df.drop_duplicates(subset='actual_page', inplace=True, keep='last')
        
        # find all frames to use
        selected = df.head(1)
        unselected = df.copy()
        while True:
            # break when selection is complete
            if selected['actual_page'].max() == df['actual_page'].max():
                break
            
            # look for page 1 after the previous
            last_page = selected['actual_page'].max()
            next_page = unselected[(unselected['actual_page'] == last_page + 1) &
                                   (unselected['max_possible_rows'] == True)]
            if len(next_page) == 1:
                selected = pd.concat([selected, next_page])
                unselected = unselected[unselected['actual_page'] > last_page + 1]
                continue
            
            # if page still isn't found, find one with slight overlap
            next_page = unselected[(unselected['actual_page'] < last_page + 1)]
            if len(next_page) >= 1:
                selected = pd.concat([selected, next_page.tail(1)])
                unselected = unselected[unselected['actual_page'] > 
                                        next_page['actual_page'].max()]
                continue
            
        # clear unnecessary frames from memory
        frames = {i: frames[i] for i in selected.index if i not in frame_info.keys()}
            
        # get match info for all selected frames
        print(f'Selected {len(selected)} frames for analysis')
        matches = []
        for i, frame in frames.items():
            match_info = self.identify_image(frame)
            match_info['frame'] = i
            matches.append(match_info)
            
        return matches

    # =============================================================================
    # DETECTING ICONS
    # =============================================================================
    @timeit
    def identify_image(
            self,
            rgb_image: np.ndarray,
            crop: bool = True,
            analyze_scrollbar: bool = False,
            save_matches: bool = True,
            save_heatmaps: bool = False
            ):
        
        # make sure icons were loaded
        icons = self.card_icons
        if not icons:
            return print('Unable to load icons: exiting...')
        
        # get original icon size (should be 100 x 100 px)
        sample_icon = list(icons.values())[0]
        orig_h, orig_w = sample_icon.shape[0:2]
        
        # create a copy of the input to avoid modifying the original
        img_rgb = self.standardize_input_image(rgb_image.copy())
        
        # preprocess the image for detections
        img_gray = self.standardize_template(img_rgb, 1.0)

        # get optimal rescale factor for resizing card & orb icons
        if self.default_scale is None:
            std_temp = self.standard_template
            scale, info = get_best_scale(img_gray, std_temp)
            self.default_scale = scale
            
            # get the top boundary of the box region
            try:
                _h, _dh = info['boxes'][0][1], info['template_height']
                self.box_top_boundary = _h + _dh
            except:
                self.box_top_boundary = 0
            
        else:
            scale = self.default_scale
            
        # get new dimensions based on optimal rescale factor
        new_w, new_h = (int(scale*orig_w), int(scale*orig_h))
        print(f'Icon dimensions: ({new_w}, {new_h})')
        
        # store icon dimensions for access in other functions
        self.icon_w, self.icon_h = new_w, new_h
            
        # get the position of the bottom boundary
        # if self.box_bottom_boundary is None:
        self.bottom_boundary_info = self.find_bottom_boundary(img_gray, scale)
        self.box_bottom_boundary = self.bottom_boundary_info['boundary']
        
        # get scrollbar position
        scroll_info = None
        if analyze_scrollbar:
            scroll_info = self.get_scrollbar_info(img_rgb, scale)
            
        # trim the image to just the monster box so matching is faster
        img = self.trim_to_box(img_gray)
        img_rgb = self.trim_to_box(img_rgb)
        
        # get crop factor for icons - crop to exclude plusses, level, etc.
        if crop:
            dh0 = int(0.45 * new_h) # top offset ; 0.45 default
            dh1 = int(0.30 * new_h) # bottom offset ; 0.30 default
            dw0 = int(0.15 * new_w) # left offset ; 0.15 default
            dw1 = int(0.15 * new_w) # right offset ; 0.15 default
        else:
            dh0 = dh1 = dw0 = dw1 = int(0.1 * new_h)
            
        # make sure the crop dimensions are valid
        dh1, dw1 = max(1, dh1), max(1, dw1)
        dh0, dh1 = (0, 1) if (dh0 + dh1) >= new_h else (dh0, dh1)
        dw0, dw1 = (0, 1) if (dw0 + dw1) >= new_w else (dw0, dw1)
        icon_offsets = ((dw0, dw1), (dh0, dh1))
        print(f'Icon offsets: {icon_offsets}')
        
        # fix orb icons
        orbs = {k: self.standardize_template(v, scale) for k, v in self.orb_icons.items()}
        orb_h, orb_w = list(orbs.values())[0].shape[0:2]
        
        # detect orb positions
        print('\nDetecting attributes...\n')
        orb_results = {color: matches for (color, matches) in
                        (self.find_orbs(img, orb, color) 
                        for color, orb in orbs.items())}
        
        # save images of orb results
        [self.show_matches(img_rgb, orbs[color], res, False, save_heatmaps)
         for color, res in orb_results.items()]
        
        # report how many of each orb type was detected
        tot_orbs = sum(r['num_matches'] for r in orb_results.values())
        print(f'{tot_orbs} orb(s) detected:')
        for c, r in orb_results.items():
            n = r['num_matches']
            print(f'    {c}: {n}')
        
        # get orb match positions
        orb_positions = []
        for color, matches in orb_results.items():
            [orb_positions.append((color, box)) for box in matches['boxes']]
                
        # check if other orbs could be subattribute positions
        print('\nPredicting card positions...')
        orb_offset_w = int(self.orb_crop_factor * orb_w)
        orb_offset_h = int(self.orb_crop_factor * orb_h)
        def is_pair(orb_1, orb_2):
            pos_1, pos_2 = orb_1[1], orb_2[1]
            x1 = pos_1[0] - orb_offset_w
            y1 = pos_1[1] - orb_offset_h
            x2 = pos_2[0] + orb_w + orb_offset_w
            y2 = pos_2[1] + orb_h + orb_offset_h
            dx, dy = x2 - x1, y2 - y1
            return (abs(dx - new_w) < 15) and (abs(dy - new_h) < 15) # < 10 default
        
        # for checking if lefover orbs are actually single attribute cards or not
        def not_single(orb_1, orb_2):
            different = (orb_1 != orb_2)
            staggered = abs(orb_1[0] - orb_2[0]) > orb_w
            above = (orb_1[1] - orb_2[1]) < 0
            close = (pt_sep(orb_1, orb_2) < new_h)
            return staggered and above and close and different
        
        # get dual attribute cards
        classified_orbs = []
        dual_attr = []
        for orb_1 in orb_positions:
            if orb_1 in classified_orbs:
                continue
            for orb_2 in orb_positions:
                if orb_2 in classified_orbs:
                    continue
                paired = is_pair(orb_1, orb_2)
                if paired:
                    dual_attr.append((orb_1, orb_2))
                    classified_orbs += [orb_1, orb_2]
           
        # get single attribute cards - must not have any other orbs nearby
        not_classified = [orb for orb in orb_positions if orb not in classified_orbs]
        single_attr = [(orb, (None, (None, None))) for orb in orb_positions 
                       if orb not in classified_orbs and not any(
                       not_single(orb[1], orb2[1]) for orb2 in not_classified)]
        
        # combine the two to get all predicted card locations and attributes
        predicted_cards = dual_attr + single_attr
        num_expected_matches = len(predicted_cards)
        
        # display predicted card locations and get bounding boxes
        pairs_img = img_rgb.copy()
        card_bboxes = []
        for ((attr, tl), (sattr, br)) in predicted_cards:
            tl_x = tl[0] - orb_offset_w - LINEWIDTH
            tl_y = tl[1] - orb_offset_h - LINEWIDTH
            if sattr is not None:
                br_x = br[0] + orb_w + orb_offset_w + LINEWIDTH
                br_y = br[1] + orb_h + orb_offset_h + LINEWIDTH
            else:
                br_x = tl_x + new_w + LINEWIDTH
                br_y = tl_y + new_h + LINEWIDTH
            top_left = (tl_x, tl_y)
            bottom_right = (br_x, br_y)
            
            # make sure the region is not cutoff
            dx, dy = abs(br_x - tl_x), abs(br_y - tl_y)
            aspect = dx / dy
            aspect = 1 / aspect if aspect > 1 else aspect
            if aspect < 0.9:
                continue
            
            # get bounding box
            bbox = (top_left, bottom_right)
            sattr = sattr if sattr else ''
            
            # get region
            reg = extract_region(img, bbox, pad=0)
            
            # if the region is supposedly single attribute, check for orbs
            if sattr is None:
                num_orbs = self.count_orbs(reg, orbs, scale, standardize=False)
                if num_orbs > 1:
                    continue
            
            # store bounding box information for valid predicted regions
            card_bboxes.append((f'{attr}{sattr}', bbox))
            
            # show the box
            cv2.rectangle(pairs_img, top_left, bottom_right, LINECOLOR, LINEWIDTH)
            
        # report how many cards are predicted and group bboxes by attributes
        print(f'\n{num_expected_matches} card(s) predicted:')
        predictions = {c: [crd[1] for crd in card_bboxes if crd[0] == c]
                       for c in set(p[0] for p in card_bboxes)}
        [print(f'    {colors}: {len(cds)}') for colors, cds in predictions.items()]
            
        # save a picture of the predicted card positions
        pairs_path = os.path.abspath('./tests/predicted card positions.png')
        save_rgb(pairs_img, pairs_path)
        
        # identify regions by comparing to card icons
        print('\nIdentifying regions...')
        # TODO: skip icon regions that are cut off at top or bottom
        matched_cards = []
        unmatched_cards = {}
        # REGULAR
        for attrs, bboxes in predictions.items():
            matched, unmatched, unmatched_rgb = self.identify_region(
                                                        img, img_rgb, attrs, 
                                                        bboxes, icons, scale,
                                                        icon_offsets, crop)
            matched_cards += matched
            unmatched_cards[attrs] = unmatched_rgb
            
        # try again for unmatched regions with 1 attribute, checking again 
        # with all subattributes
        for attr, regions in unmatched_cards.items():
            if len(attr) > 1 or len(regions) == 0:
                continue
            for bbox, _ in regions.items():
                new_matches = []
                for attrs in [attr + s for s in ['r', 'b', 'g', 'l', 'd']]:
                    new_match, _, _ = self.identify_region(
                                                    img, img_rgb, attrs,
                                                    [bbox], icons, scale,
                                                    icon_offsets, crop)
                    if not new_match:
                        continue
                    new_matches += new_match
                
                # choose the best match if there are multiple
                if not new_matches:
                    continue
                best_match = sorted(new_matches, key=lambda m: m['confidence'])[-1]
                matched_cards += [best_match]
        
        # update the list of all matched cards
        self.matched_cards += matched_cards
    
        num_matches = len(matched_cards)
        print(f'\n{num_matches} cards matched ({num_expected_matches} predicted)\n')
        
        # print the matched icon results
        for res in matched_cards:
            print('Name:', res.get('name', ''))
            for category, value in res.items():
                if 'map' in category:
                    continue
                print(f'    {category}:   {value}')
            print('\n')
            
        # compile match info
        match_info = {
            'img_rgb': img_rgb,
            'scale': scale,
            'image_w': img_gray.shape[1],
            'image_h': img_gray.shape[0],
            'icon_w': self.icon_w,
            'icon_h': self.icon_h,
            'box_top_boundary': self.box_top_boundary,
            'box_bottom_boundary': self.box_bottom_boundary,
            'scroll_info': scroll_info,
            'unmatched_cards': unmatched_rgb,
            'matched_cards': matched_cards,
            'num_matches': num_matches,
            }
            
        return match_info
    
    def identify_all_images(self):
        for rgb_image in self.rgb_input_images:
            # asyncio.run(self.detect_icons(rgb_image))
            matches = self.identify_image(rgb_image)
            
        return self.matched_cards
        
# =============================================================================
# OPTIMIZING SCALE
# =============================================================================
def evaluate_scale(
        input_image: np.ndarray, 
        template: np.ndarray, 
        scale: float
        ) -> tuple:
    
    # print(f'Checking scale {scale:.3f}')
    
    # rescale the template
    img_h, img_w = input_image.shape[0:2]
    orig_h, orig_w = template.shape[0:2]
    new_w = min(int(scale*orig_w), img_w)
    new_h = min(int(scale*orig_h), img_h)
    new_size = (new_w, new_h)
    rescaled = cv2.resize(template, new_size, interpolation=INTERPOLATION)
    
    # find matches
    result = find_matches(input_image, rescaled)
    result['scale'] = scale
    result['dimensions'] = new_size
    
    # confidence = result['confidence']
    # print(f'Confidence: {confidence}')
    
    return (scale, result)

@timeit
def get_best_scale(
        input_image: np.ndarray, 
        template: np.ndarray,
        min_scale: float = 0.25, 
        num_scales: int = 50
        ) -> float:
    
    # TODO validate inputs
    
    # store original image
    orig_img = input_image.copy()
    orig_img_h, orig_img_w = orig_img.shape[0:2]
    
    # get downscaling factor to make sure input image is always 500px wide
    input_h, input_w = input_image.shape[0:2]
    downscale = min(1.0, DEFAULT_SCALE_WIDTH / input_w)
    print(f'Downscale factor: {downscale}')
    
    # rescale input image
    img_w, img_h = int(downscale * orig_img_w), int(downscale * orig_img_h)
    if orig_img_w != DEFAULT_SCALE_WIDTH:
        img = cv2.resize(orig_img, (img_w, img_h), interpolation=INTERPOLATION)
    else:
        img = orig_img
    
    # scaling factor based on width of standard template relative to single 
    # card icon from the same screenshot
    mult = (100 / 164) * downscale # raw icon / screenshot icon
    
    # rescale template
    orig_h, orig_w = int(mult * template.shape[0]), int(mult * template.shape[1])
    template = cv2.resize(template, (orig_w, orig_h), interpolation=INTERPOLATION)
    print(f'Original template height: {orig_h}\nOriginal template width: {orig_w}')
    
    # define minimum width based on fact that template should be at least 1/3
    # of the width of the original image (can change if matching template changes)
    min_w = int(img_w * 0.35)
    min_scale = max(min_scale, min_w/orig_w)
    print(f'Minimum scale: {min_scale}')
    
    # get maximum scale such that template isn't larger than the image
    max_scale = img_w / orig_w
    
    # get range of scales to test
    scales = np.linspace(min_scale, max_scale, num_scales)
    
    # test the different scales
    results = {res[0]: res[1] for res in [evaluate_scale(img, template, scale)
                                          for scale in scales]}
    
    # get the scale associated with the highest confidence value
    best_scale = sorted([(scale, res['confidence']) for scale, res in results.items()],
                        key = lambda i: i[1])[-1][0]
    
    # remove the downscaling factor
    best_scale /= downscale
    
    # for debugging purposes
    if DEBUG:
        # get the scale associated with the highest confidence result
        df = pd.DataFrame.from_dict(results, orient='index')
        df.sort_values(by='confidence', inplace=True, ascending=False)
        confidence = df['confidence'].values[0]
        best_scale = df['scale'].values[0]
        boxes = df['boxes'].values[0]
        size = df['dimensions'].values[0]
        print(df[['scale', 'matched', 'confidence', 'dimensions']].head(10))
        print(f'Best scale: {best_scale:.3f}')
        print(f'Confidence: {confidence:.3f}')
        w, h = size
        for box in boxes:
            cv2.rectangle(img, box, (box[0] + w, box[1] + h), LINECOLOR, LINEWIDTH)
        save_path = os.path.abspath(f'./tests/optimized scale {best_scale:.2f}.png')
        save_rgb(img, save_path)

    print(f'Best scale: {best_scale:.3f}')
    return best_scale, results[best_scale]

# =============================================================================
# GENERAL MATCHING FUNCTION
# =============================================================================
def find_matches(input_image: np.ndarray, template: np.ndarray) -> dict:
    # make sure image and template are valid
    if not isinstance(input_image, np.ndarray):
        print(f'Image must be a numpy array, not {type(input_image)}')
        return {}
    if not isinstance(template, np.ndarray):
        print(f'Template must be a numpy array, not {type(template)}')
        return {}
    
    # make sure the image and template are grayscale
    input_image = to_gray(input_image)
    template = to_gray(template)
    
    # make sure the template isn't too big
    temp_h, temp_w = template.shape[0:2]
    img_h, img_w = input_image.shape[0:2]
    if temp_h >= img_h or temp_w >= img_w:
        if temp_h >= img_h:
            old_h = temp_h
            temp_h = img_h - 1
            temp_w = int((temp_h/old_h) * temp_w)
        if temp_w >= img_w:
            old_w = temp_w
            temp_w = img_w - 1
            temp_h = int((temp_w/old_w) * temp_h)
        try:
            template = cv2.resize(template, (temp_w, temp_h), PREFERRED_METHOD)
        except:
            print(f'Unable to resize template to {(temp_w, temp_h)}')
    
    # get matches
    try:
        orig_res = cv2.matchTemplate(input_image, template, PREFERRED_METHOD)
        
    except Exception as ex:
        print(f'Could not perform matching: {ex}')
        print(f'Image dimensions: {input_image.shape}')
        print(f'Template dimensions: {template.shape}')
        return {
            'matched': False,
            'num_matches': 0,
            'boxes': [],
            'confidence': 0.0,
            'template_width': temp_w,
            'template_height': temp_h,
            }
    
    # SQDIFF methods use array minimum rather than maximum
    if PREFERRED_METHOD in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        orig_res = np.abs(orig_res - 1)
    
    # use array maximum as confidence level
    confidence = np.amax(orig_res)
    
    # refine to exclude bad matches
    if PREFERRED_METHOD == cv2.TM_SQDIFF_NORMED:
        min_confidence = 0.80 
    elif PREFERRED_METHOD == cv2.TM_CCORR_NORMED:
        min_confidence = 0.80
    elif PREFERRED_METHOD == cv2.TM_CCOEFF_NORMED:
        min_confidence = 0.45 # 0.45 default
    else:
        min_confidence = 0.60
    
    # res = np.where(orig_res > min_confidence, res, 0)
    res = orig_res
    
    # get locations of bounding boxes
    threshold = max(min_confidence, np.percentile(res, 99.99)) # 99.99
    loc = np.where(res >= threshold)
    
    # return early if there are way too many matches
    if len(loc[0]) > 1000:
        # TODO get the n best matches
        return {
            'matched': False,
            'num_matches': len(loc[0]),
            'boxes': [],
            'confidence': confidence,
            'confidence_map': orig_res,
            'template_width': temp_w,
            'template_height': temp_h,
            }
    
    # get box points
    pts = list(zip(*loc[::-1]))
    
    # filter results that are nearby
    boxes = []
    min_sep = template.shape[1] / 3
    for pt in pts:
        if not any(pt_sep(pt, box) < min_sep for box in boxes):
            boxes.append(pt)
    # [boxes.append(pt) for pt in pts if not 
     # any(pt_sep(pt, box) < min_sep for box in boxes)]
    
    # get number of matches
    num_matches = len(boxes)
    
    # get confidence levels
    confidence_levels = [res[box[1], box[0]] for box in boxes]
    
    # send results
    return {
        'threshold': threshold,
        'matched': num_matches > 0,
        'num_matches': num_matches,
        'boxes': boxes,
        'confidence': confidence,
        'confidence_levels': confidence_levels,
        'confidence_map': orig_res,
        'template_width': temp_w,
        'template_height': temp_h,
        }

# =============================================================================
# UTILITIES
# =============================================================================
def gauss_sharpen(img: np.ndarray, sigma1 = 0.5, sigma2 = 1.5) -> np.ndarray:
    return gaussian_filter(img, sigma1) - gaussian_filter(img, sigma2)

def to_numpy(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        print(f'Could not convert {type(image)} to numpy array')
        return image

def to_gray(image: np.ndarray) -> np.ndarray:
    try:
        if image.ndim == 2:
            return image
        elif image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            print('Could not convert image to grayscale')
            return image
    except Exception as ex:
        print(f'Unable to convert to grayscale: {ex}')
        return image
    
def to_rgb(image: np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image)
    if not isinstance(image, np.ndarray):
        print(f'Input should be numpy array, not {type(image)}')
        return image
    try:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.ndim == 2:
            return image
    except Exception as ex:
        print(f'Unable to convert image to RGB: {ex}')
        return image

def to_rgba(image: np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image)
    try:
        if image.shape[-1] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[-1] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        elif image.ndim == 2:
            return image
    except Exception as ex:
        print(f'Unable to convert image to RGBA: {ex}')
        return image

def rgb_to_sharp_gray(
        rgb_image: np.ndarray, 
        sigma1: float = 0.5, 
        sigma2: float = 1.0
        ) -> np.ndarray:
    return gauss_sharpen(to_gray(rgb_image), sigma1, sigma2)

def rescale(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[0:2]
    new_size = int(w * scale), int(h * scale)
    try: 
        return cv2.resize(image, new_size, interpolation=INTERPOLATION)
    except Exception as ex:
        print(f'Unable to rescale image by {scale}: {ex}')
        
def save_rgb(image: np.ndarray, path: str) -> None:
    try:
        if image.ndim == 3:
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        elif image.ndim == 2:
            cv2.imwrite(path, image)
        elif image.ndim == 4:
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    except Exception as ex:
        return print(f'Unable to save to {path}:\n{ex}')
    
def extract_region(image: np.ndarray, bbox: tuple, pad: int = 0) -> np.ndarray:
    ((y1, x1), (y2, x2)) = bbox
    pad = int(pad*y1) if (0 < abs(pad) < 1) else pad
    region = image[(x1-pad):(x2+pad), (y1-pad):(y2+pad)].copy()
    return region

def peek(array: np.ndarray) -> None:
    try:
        Image.fromarray(array).show() if not NO_PEEK else None
    except Exception as ex:
        print(f'Could not show image: {ex}')

# =============================================================================
# LOADING VIDEO
# =============================================================================
def load_video_frames(filepath: str):
    if not os.path.exists(filepath):
        return print(f'{filepath} does not exist')
    frames = None
    print(f'Loading video frames from {filepath}')
    try:
        # initialize video capture
        capture = cv2.VideoCapture(filepath)
        
        # create array to store frames in
        channels = 3
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((num_frames, frame_h, frame_w, channels), 'uint8')
        
        # read the video frames
        framecount = 0
        has_data = True
        while framecount < num_frames and has_data:
            has_data, frame = capture.read()
            frames[framecount] = to_rgb(frame)
            framecount += 1
            
        print(f'Loaded {frame_w} by {frame_h} video containing',
              f'{num_frames} frames in {time.time()-t0:.3f} seconds')
        return frames
    
    except OSError as ose:
        return print(f'Unable to load video frames: {ose}')
    except Exception as ex:
        return print(f'Unable to load video frames: {ex}')
        
# =============================================================================
# LOCAL TESTING
# =============================================================================
if __name__ == '__main__':
    t0 = time.time()
    
    os.chdir(os.path.dirname(__file__))
    ss_path = os.path.abspath('../assets/screenshots/box_test_scrolled.png')
    # ss_path = os.path.abspath('../assets/screenshots/box_test_medium.png')
    vid_path = os.path.abspath('../assets/screenshots/sample_video.mp4')
    
    # match_info = Matcher(ss_path).identify_all_images()
    match_info = Matcher(vid_path).identify_video(vid_path)
    
    # frames = load_video_frames(vid_path)
        
    print(f'Finished in {time.time()-t0:.3f} seconds')

# =============================================================================
# TODO LIST
# =============================================================================
'''
    - Read from videos
    - Store monster list
    - Create UI for interacting
    - Accept manual override grade (either add new or reject)
    - Create stitched image of all icons from box list

'''
