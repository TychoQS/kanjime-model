from pathlib import Path
from matplotlib import font_manager
import matplotlib.pyplot as plt 

def load_font(font_path):
    """Load a Japanese font in order to display kanjis"""
    font_path = Path(font_path)

    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
        print("Font loaded.")
    else:
        print(f"WARNING: Font not found at {font_path}.")