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
import asyncio

import concurrent.futures
from threading import Thread
from queue import Queue

import nest_asyncio
nest_asyncio.apply()

try:
    from .assetmanager import AssetManager
    from .utils import timeit, pt_sep, imread_rgb
except:
    os.chdir(os.path.dirname(__file__))
    from assetmanager import AssetManager
    from utils import timeit, pt_sep, imread_rgb

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
# using cv2.TM_CCORR_NORMED or cv2.TM_SQDIFF_NORMED generally works based on
# results of testing at a variety of scales
PREFERRED_METHOD = cv2.TM_CCOEFF_NORMED

# INTERPOLATION = cv2.INTER_LANCZOS4
INTERPOLATION = cv2.INTER_CUBIC
LINEWIDTH = 5
OFFSET_LINEWIDTH = True
LINECOLOR = (0, 255, 255)
DEFAULT_SCALE = 1.0
READ_MODE = cv2.IMREAD_UNCHANGED
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 2.5
SLEEP = 0.0001
DEFAULT_SCREENSHOT_WIDTH = 800
DEFAULT_SCALE_WIDTH = 800
DEBUG = False
# DEBUG = True

# =============================================================================
# MAIN MATCHING CLASS
# =============================================================================
class Matcher:
    def __init__(
            self, 
            input_images: list,
            standard_scale: float = None,
            orb_crop_factor: float = 0.15,
            
            **kwargs
            ) -> dict:
        
        # store input variables
        self.standard_scale = standard_scale
        self.orb_crop_factor = orb_crop_factor
        
        # parse the input images
        self.input_images = self.add_inputs(input_images)
        self.num_input_images = len(self.input_images)
        if self.num_input_images == 0:
            return print('No input images provided')
        print(f'Got {self.num_input_images} input images')
        
        # get event loop
        self.loops = {'main': asyncio.get_event_loop()}
        
        # initialize the asset manager and load required assets
        self._load_required_assets()
        
    def add_inputs(self, inputs: list):
        inputs = [inputs] if not isinstance(inputs, list) else inputs
        input_images = []
        for item in inputs:
            img = self.imarray(item)
            if isinstance(img, str):
                if not os.path.exists(img):
                    print(f'{img} does not exist')
                    continue
                img = load_img(img)
                # TODO: handle URLs and videos
            input_images.append(img) if img is not None else None
        return input_images
    
    # =============================================================================
    # CLASS METHODS
    # =============================================================================
    @staticmethod
    def standardize_input_image(image: np.ndarray) -> np.ndarray:
        image = imarray(image)
        rgb = to_rgb(image)
        h, w = image.shape[0:2]
        scale = min(1.0, DEFAULT_SCREENSHOT_WIDTH / w)
        rescaled = rescale(rgb, scale)
        return gauss_sharpen(rescaled)
    
    @staticmethod
    def crop_orb(orb: np.ndarray, crop_factor: float = 0.15) -> np.ndarray:
        h, w = orb.shape[0:2]
        dh, dw = int(crop_factor * h), int(crop_factor * w)
        return orb[dh:-dh, dw:-dw]
    
    @staticmethod
    def crop_icon(icon: np.ndarray, offsets: tuple) -> np.ndarray:
        (dw0, dw1), (dh0, dh1) = offsets
        return icon[dh0:-dh1, dw0:-dw1]
    
    @staticmethod
    def fix_template(template: np.ndarray, scale: float) -> np.ndarray:
        template = np.array(Image.fromarray(template, mode='RGBA').convert('L'))
        temp_w, temp_h = template.shape[1], template.shape[0]
        dsize = (int(scale * temp_w), int(scale * temp_h))
        template = cv2.resize(template, dsize=dsize, interpolation=INTERPOLATION)
        template = gauss_sharpen(template)    
        return template
    
    # =============================================================================
    # PRIVATE FUNCTIONS
    # =============================================================================
    def _load_required_assets(self):
        # TODO: make these loading functions async
        self.asset_manager = AssetManager()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            
            self.card_icons = self.asset_manager.load_all_icons()
            self.standard_template = self.asset_manager.get_standard_template()
            self.orb_icons = self.asset_manager.load_orb_icons()
            self.cards_by_attributes = self.asset_manager.load_cards_by_attributes()
    
    # =============================================================================
    # SUB-ROUTINES
    # =============================================================================
    @classmethod
    def find_orbs(
            cls,
            input_image: np.ndarray, 
            orb: np.ndarray, 
            color: str, 
            crop_factor: float = 0.15
            ) -> tuple:
        
        # crop the orb
        orb = cls.crop_orb(orb, crop_factor)
        dh, dw = crop_factor * orb.shape[0], crop_factor * orb.shape[1]
        offsets = (dw, dw), (dh, dh)
        
        # find matches for the orb in the input image
        matches = find_matches(input_image, orb)
        matches['name'] = color
        matches['offests'] = offsets
        return (color, matches)
    
    @classmethod
    def show_matches(
            cls,
            input_image: np.ndarray, 
            template: np.ndarray, 
            matches: dict,
            show_full_box: bool = True,
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
        if 'bbox' in matches.keys():
            start, end = matches['bbox']
            label_img(img, start, end)
        
        # otherwise, create bounding boxes for display
        else:
            h, w = template.shape[0:2]
            boxes = matches['boxes']
            lw = LINEWIDTH if OFFSET_LINEWIDTH else 0
            for box in boxes:
                # expand the box to account for line width and initial crop
                (dw0, dw1), (dh0, dh1) = offsets
                if show_full_box:
                    start = (box[0] - dw0 - lw, box[1] - dh0 - lw)
                    end = (box[0] + w + dw1 + lw, box[1] + h + dh1 + lw)
                else:
                    start = (box[0] - lw, box[1] - lw)
                    end = (box[0] + w + lw, box[1] + h + lw)
                label_img(img, start, end)
        
        # save result image
        match_path = os.path.abspath(f'./tests/{name}.png')
        cv2.imwrite(match_path, img)
            
        # save heatmap
        confidence_map = matches.get('confidence_map', None)
        if save_heatmap and confidence_map is not None:
            heatmap_path = os.path.abspath(f'./tests/{name}_map.png')
            cv2.imwrite(heatmap_path, confidence_map*(255*np.amax(confidence_map)))

    @classmethod
    async def identify_region(
            cls,
            input_image: np.ndarray,
            orig_img: np.ndarray,
            attributes: str,
            bbox: tuple,
            cards_by_attributes: dict,
            icons: dict, 
            scale: float,
            offsets: tuple,
            crop: bool = True
            ):
        
        # get list of cards to search over
        card_list = cards_by_attributes[attributes]
        
        # get the region
        (y1, x1), (y2, x2) = bbox
        region = input_image[x1:x2, y1:y2].copy()
        
        # compare provided region against all icons with the same attributes
        matched_cards = []
        for card_name in card_list:
            # prepare card icon
            card_icon = icons.get(card_name, None)
            if card_icon is None:
                continue
            card_icon = cls.fix_template(card_icon, scale)
            card_icon = cls.crop_icon(card_icon, offsets) if crop else card_icon
            
            # check for a match
            result = find_matches(region, card_icon)
            
            # don't store info for non-matches
            if not result['matched']:
                continue
            
            # store match information
            result['name'] = card_name
            result['bbox'] = bbox
            result['offsets'] = offsets
            matched_cards.append(result)
            print(f'Matched {card_name}')
            
            # show the matched region
            cls.show_boxes(orig_img, card_icon, result, True, False)
            
            # only match once
            break
        
        if not matched_cards:
            print(f'No match found for predicted {attributes} card')
            
        return matched_cards  

    # =============================================================================
    # DETECTING ICONS
    # =============================================================================
    @timeit
    async def detect_icons(
            self,
            input_image: np.ndarray,
            icons: dict = None,
            crop: bool = True,
            save_matches: bool = True,
            save_heatmaps: bool = False
            ):
        
        # create a copy of the input to avoid modifying the original
        img = input_image.copy()
        
        # get optimal rescale factor for matching
        scale = self.loops['main'].run_until_complete(get_scale(img))
        
        # make sure icons were loaded
        icons = {k: v for k, v in icons.items() if v is not None}
        if not icons:
            return print('Unable to load icons: exiting...')
        
        # get original icon size (should be 100 x 100 px)
        orig_h, orig_w = list(icons.values())[0].shape[0:2]
        
        # get new dimensions based on optimal rescale factor
        new_w, new_h = (int(scale*orig_w), int(scale*orig_h))
        
        # get crop factor for icons - crop to exclude plusses, level, etc.
        if crop:
            dh0 = int(0.45 * new_h) # top offset
            dh1 = int(0.30 * new_h) # bottom offset
            dw0 = int(0.15 * new_w) # left offset
            dw1 = int(0.15 * new_w) # right offset
        else:
            dh0 = dh1 = dw0 = dw1 = int(0.1 * new_h)
            
        # make sure the crop dimensions are valid
        dh1, dw1 = max(1, dh1), max(1, dw1)
        dh0, dh1 = (0, 1) if (dh0 + dh1) >= new_h else (dh0, dh1)
        dw0, dw1 = (0, 1) if (dw0 + dw1) >= new_w else (dw0, dw1)
        icon_offsets = ((dw0, dw1), (dh0, dh1))
        
        # load orbs
        orbs = {k: self.fix_template(v, scale) for k, v in orbs.items()}
        orb_h, orb_w = list(orbs.values())[0].shape
        
        # detect orb positions
        print('\nDetecting attributes...')
        orb_results = {color: matches for (color, matches) in
                        (self.find_orbs(img, orb, color, self.orb_crop_factor) 
                        for color, orb in orbs.items())}
        
        # save images of orb results
        [self.show_boxes(self.img_raw, orbs[color], res, False, save_heatmaps)
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
            return (abs(dx - new_w) < 10) and (abs(dy - new_h) < 10)
        
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
           
        # get single attribute cards
        single_attr = [(orb, (None, (None, None))) for orb in orb_positions 
                       if orb not in classified_orbs]
        
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
                br_x = tl_x + new_w
                br_y = tl_y + new_h
            top_left = (tl_x, tl_y)
            bottom_right = (br_x, br_y)
            cv2.rectangle(pairs_img, top_left, bottom_right, LINECOLOR, LINEWIDTH)
            
            # store bounding box information
            bbox = (top_left, bottom_right)
            sattr = sattr if sattr else ''
            card_bboxes.append((f'{attr}{sattr}', bbox))
            
        # report how many cards are predicted
        print(f'\n{num_expected_matches} card(s) predicted:')
        predictions = {c: sum(1 for crd in card_bboxes if crd[0] == c)
                       for c in set(p[0] for p in card_bboxes)}
        [print(f'    {colors}: {num}') for colors, num in predictions.items()]
            
        # save a picture of the predicted card positions
        pairs_path = os.path.abspath('./tests/predicted card positions.png')
        cv2.imwrite(pairs_path, pairs_img)
        
        # identify regions by comparing to card icons
        print('\nIdentifying regions...\n')
        print(cards_by_attributes)
        matched_cards = await asyncio.gather(*(identify_region(
                                                img, img_rgb, attrs, bbox, 
                                                cards_by_attributes, icons,
                                                scale, icon_offsets, crop)
                                                for (attrs, bbox) in card_bboxes))
        matched_cards = [m for mc in matched_cards for m in mc]
        
    
        # show the matched icon results
    
        # TODO: skip icon regions that are cut off at top or bottom
                    
        num_matches = len(matched_cards)
        print(f'\n{num_matches} cards matched ({num_expected_matches} predicted)\n')
        
        for res in matched_cards:
            print('Name:', res.get('name', ''))
            for category, value in res.items():
                if 'map' in category:
                    continue
                print(f'    {category}:   {value}')
            print('\n')
        return matched_cards
        
# =============================================================================
# OPTIMIZING SCALE
# =============================================================================
async def evaluate_scale(
        input_image: np.ndarray, 
        template: np.ndarray, 
        scale: float
        ) -> tuple:
    
    print(f'Checking scale {scale:.3f}')
    
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
    
    confidence = result['confidence']
    print(f'Confidence: {confidence}')
    
    return (scale, result)

@timeit
async def get_scale(
        input_image: np.ndarray, 
        min_scale: float = 0.25, 
        num_scales: int = 50
        ) -> float:
    
    # store original image
    orig_img = input_image.copy()
    orig_img_h, orig_img_w = orig_img.shape[0:2]
    
    # load a standard icon with the same width as card icons
    template = AssetManager.get_standard_template()
    if template is None:
        print('Unable to calibrate scale: standard matching template not found')
        print(f'Using default scale: {DEFAULT_SCALE}')
        return DEFAULT_SCALE
    
    # get downscaling factor to make sure input image is always 500px wide
    input_h, input_w = input_image.shape[0:2]
    downscale = min(1.0, DEFAULT_SCALE_WIDTH / input_w)
    print(f'Downscale factor: {downscale}')
    
    # rescale input image
    img_w, img_h = int(downscale * orig_img_w), int(downscale * orig_img_h)
    if orig_img_w != DEFAULT_SCALE_WIDTH:
        img = cv2.resize(orig_img, (img_w, img_h), interpolation=INTERPOLATION)
    
    # scaling factor based on width of standard template relative to single 
    # card icon from the same screenshot
    mult = (100 / 164) * downscale # raw icon / screenshot icon
    
    # rescale template
    orig_h, orig_w = int(mult * template.shape[0]), int(mult * template.shape[1])
    template = cv2.resize(template, (orig_w, orig_h), interpolation=INTERPOLATION)
    print(f'Original template height: {orig_h}\nOriginal template width: {orig_w}')
    
    # image processing on the template
    template = to_gray(template)
    template = gauss_sharpen(template)
    
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
    results_list = await asyncio.gather(*(evaluate_scale(img, template, scale) 
                                          for scale in scales))
    results = {scale: result for (scale, result) in results_list}
    
    # get the scale associated with the highest confidence result
    df = pd.DataFrame.from_dict(results, orient='index')
    df.sort_values(by='confidence', inplace=True, ascending=False)
    confidence = df['confidence'].values[0]
    best_scale = df['scale'].values[0]
    
    # remove the downscaling factor
    best_scale /= downscale
    
    # for debugging purposes
    if DEBUG:
        boxes = df['boxes'].values[0]
        size = df['dimensions'].values[0]
        
        print(df[['scale', 'matched', 'confidence', 'dimensions']].head(10))
        print(f'Best scale: {best_scale:.3f}')
        print(f'Confidence: {confidence:.3f}')
    
        w, h = size
        for box in boxes:
            cv2.rectangle(img, box, (box[0] + w, box[1] + h), LINECOLOR, LINEWIDTH)
        
        save_path = os.path.abspath(f'./tests/optimized scale {best_scale:.2f}.png')
        cv2.imwrite(save_path, img)
    
    print(f'Best scale: {best_scale:.3f}')
    return best_scale

# =============================================================================
# GENERAL MATCHING FUNCTION
# =============================================================================
def find_matches(image: np.ndarray, template: np.ndarray) -> dict:
    # make sure image and template are valid
    if not isinstance(image, np.ndarray):
        print(f'Image must be a numpy array, not {type(image)}')
        return {}
    if not isinstance(template, np.ndarray):
        print(f'Template must be a numpy array, not {type(template)}')
        return {}
    
    # make sure the template isn't too big
    temp_h, temp_w = template.shape
    img_h, img_w = image.shape
    if temp_h >= img_h or temp_w >= img_w:
        if temp_h >= img_h:
            old_h = temp_h
            temp_h = img_h - 1
            temp_w = int((temp_h/old_h) * temp_w)
        if temp_w >= img_w:
            old_w = temp_w
            temp_w = img_w - 1
            temp_h = int((temp_w/old_w) * temp_h)
        template = cv2.resize(template, (temp_w, temp_h), PREFERRED_METHOD)
    
    # get matches
    try:
        orig_res = cv2.matchTemplate(image, template, PREFERRED_METHOD)
        
    except Exception as ex:
        print(f'Could not perform matching: {ex}')
        print(f'Image dimensions: {image.shape}')
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
        min_confidence = 0.45
    else:
        min_confidence = 0.60
    
    # res = np.where(orig_res > min_confidence, res, 0)
    res = orig_res
    
    # get locations of bounding boxes
    threshold = max(min_confidence, np.percentile(res, 99.99))
    loc = np.where(res >= threshold)
    
    # return early if there are way too many matches
    if len(loc[0]) > 1000:
        # # only get the best box
        # loc = np.where(res == np.amax(res))
        # # boxes = list(zip(*loc[::-1]))
        # num_matches = len(loc[0])
        # if num_matches > 100:
        #     # boxes = zip(*loc[0, 0, -1])
        #     boxes = []
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
    min_sep = orig_res.shape[1] / 8
    [boxes.append(pt) for pt in pts if not 
     any(pt_sep(pt, box) < min_sep for box in boxes)]
    
    # base = int(template.shape[0] / 8)
    # rounded = [(round_base_n(y, base), round_base_n(x, base)) for y, x in pts]
    
    # use actual coordinates for points
    # for box in set(rounded):
    #     # get confidence level
    #     boxes.append(pts[rounded.index(box)])
        
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
def gauss_sharpen(img: np.ndarray, sigma1 = 0.5, sigma2 = 1.0) -> np.ndarray:
    return gaussian_filter(img, sigma1) - gaussian_filter(img, sigma2)

def imarray(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        print(f'Could not convert {type(image)} to numpy array')
        return image

def load_img(filepath: str, rgb: bool = True):
    if not os.path.exists(filepath):
        return print(f'{filepath} does not exist')
    try:
        return imread_rgb(filepath)
    except Exception as ex:
        return print(f'Unable to read {filepath}:\n{ex}')
    
def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    elif image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        print('Could not convert image to grayscale')
        return image
    
def to_rgb(image: np.ndarray):
    if isinstance(image, Image.Image):
        return np.array(image)
    if not isinstance(np.ndarray):
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

def to_rgba(image: np.ndarray):
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

def rescale(image: np.ndarray, scale: float):
    h, w = image.shape[0:2]
    new_size = int(w * scale), int(h * scale)
    try: 
        return cv2.resize(image, new_size, interpolation=INTERPOLATION)
    except Exception as ex:
        print(f'Unable to rescale image by {scale}: {ex}')

# =============================================================================
# ASYNC HELPER FUNCTIONS
# =============================================================================
async def msg(text: str):
    await asyncio.sleep(0.1)
    print(text)

async def load_cards_by_attributes(am: AssetManager):
    print('loading cards')
    return am.get_cards_by_attributes(require_icon = True)

async def load_video_frames(filepath: str):
    if not os.path.exists(filepath):
        return print(f'{filepath} does not exist')
    frames = None
    print(f'Loading video frames from {filepath}')
    try:
        # frames = vread(filepath)
        
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
            has_data, frames[framecount] = capture.read()
            framecount += 1
            
        # await asyncio.sleep(SLEEP)
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
    ss_path = os.path.abspath('../assets/screenshots/box_test_medium.png')
    vid_path = os.path.abspath('../assets/screenshots/sample_video.mp4')
    
    matches = LOOP.run_until_complete(detect_icons(ss_path))
    
    # frames = asyncio.run(load_video_frames(vid_path))
        
    print(f'Finished in {time.time()-t0:.3f} seconds')
