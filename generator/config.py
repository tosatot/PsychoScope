import math
import colorsys
import matplotlib.colors as mcolors
import numpy as np

# Define the order for models, personas, and variability types
order_dict = {
    'model': ['llama3.2-1','llama3.2-3','llama3.1-8', 'llama3.1-70', 'llama3.1-405', 
              'qwen2.5-3', 'qwen2.5-7', 'qwen2.5-14', 'qwen2.5-32', 'qwen2.5-72',
              'gemma2-2','gemma2-9', 'gemma2-27',
              'hist-8', 'hist-70', 'hist-405'],  # Added hist models],
    'persona': ['assistant', 'buddhist', 'teacher', 'antisocial', 'anxiety', 'depression', 'schizophrenia', 'psychopath', 'random'],
    'variability': ['shuffle', 'paraphrase']
}


def get_model_base_colors():
    """Define distinct base colors for each model family"""
    return {
        'llama': '#4169E1',  # Royal Blue
        'qwen': '#228B22',   # Forest Green 
        'gemma': '#8B0000',   # Dark Red
        'hist': '#9932CC'    # Dark Orchid - adding hist model
    }

def get_size_scale(model_name, model_size):
    """
    Calculate relative size within model family using log scale
    Returns value between 0 and 1
    """
    size_ranges = {
        'llama': {'min': 8, 'max': 405},
        'qwen': {'min': 3, 'max': 72},
        'gemma': {'min': 2, 'max': 27},
        'hist': {'min': 8, 'max': 405}
    }
    
    for family, range_info in size_ranges.items():
        if family in model_name.lower():
            min_size = range_info['min']
            max_size = range_info['max']
            # Use log scale to better represent size differences
            log_size = np.log(model_size)
            log_min = np.log(min_size)
            log_max = np.log(max_size)
            return (log_size - log_min) / (log_max - log_min)
    
    return 0.5  # Default if family not found

def get_model_color(model_name, model_size):
    """
    Generate color for model based on family and size
    Args:
        model_name: Name of the model (e.g., 'llama', 'qwen', 'gemma')
        model_size: Size in billions of parameters
    Returns:
        Hex color code
    """
    base_colors = get_model_base_colors()
    family = next((f for f in base_colors.keys() if f in model_name.lower()), 'default')
    base_color = base_colors.get(family, '#808080')
    
    # Convert base color to HSV
    rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    
    # Adjust color based on model size
    size_scale = get_size_scale(model_name, model_size)
    
    # Larger models get more saturated and brighter colors
    s = min(1.0, s + 0.3 * size_scale)
    v = 0.5 + 0.5 * size_scale
    
    # Convert back to RGB and then hex
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

def get_size_in_billions(model_str):
    """Extract size in billions from model string"""
    size_str = model_str.split('-')[1].rstrip('b')
    return float(size_str)

def sort_models_by_size(models):
    """Sort models by their size in billions of parameters"""
    return sorted(models, key=lambda x: get_size_in_billions(x))

# Color cache for performance
_color_cache = {}

def get_cached_model_color(model_name, model_size):
    """Cached version of get_model_color for better performance"""
    cache_key = f"{model_name}_{model_size}"
    if cache_key not in _color_cache:
        _color_cache[cache_key] = get_model_color(model_name, model_size)
    return _color_cache[cache_key]




# OLD VERSION OF get_model_color

# def get_model_color(model_name, model_size):
#     # Base colors and size ranges for each family
#     family_info = {
#         'llama': {'color': '#4C72B0', 'sizes': [8, 70, 405]},
#         'qwen': {'color': '#55A868', 'sizes': [3, 7, 14, 32, 72]},
#         'gemma': {'color': '#C44E52', 'sizes': [2, 9, 27]}
#     }
    
#     # Get the color and size range for the model family
#     for family, info in family_info.items():
#         if family in model_name.lower():
#             base_color = info['color']
#             size_range = info['sizes']
#             break
#     else:
#         # Default to gray if family not recognized
#         return '#7F7F7F'
    
#     min_size, max_size = min(size_range), max(size_range)
#     log_relative_size = (math.log(model_size) - math.log(min_size)) / (math.log(max_size) - math.log(min_size))
    
#     h, s, v = colorsys.rgb_to_hsv(*mcolors.to_rgb(base_color))
    
#     # Adjust hue slightly
#     h = (h + 0.05 * log_relative_size) % 1.0
    
#     # Increase saturation for larger models
#     s = min(1.0, s + 0.2 * log_relative_size)
    
#     # Make larger models brighter
#     v = 0.5 + 0.2 * log_relative_size

#     # # Calculate relative size within the family (linear scale)
#     # min_size, max_size = min(size_range), max(size_range)
#     # relative_size = (model_size - min_size) / (max_size - min_size)
    
#     # # Convert base color to HSV
#     # h, s, v = colorsys.rgb_to_hsv(*mcolors.to_rgb(base_color))
    
#     # # Adjust saturation and value based on relative size
#     # s = min(1.0, s + 0.3 * relative_size)
#     # v = max(0.5, v - 0.3 * (1 - relative_size))
    
#     # Convert back to RGB
#     r, g, b = colorsys.hsv_to_rgb(h, s, v)
    
#     return mcolors.to_hex([r, g, b])