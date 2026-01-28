from pathlib import Path
from matplotlib import font_manager
import matplotlib.pyplot as plt 
 
FONT_PATH = "../Noto_Sans_JP/static/NotoSansJP-Regular.ttf"

def load_font():
    """Load a Japanese font in order to display kanjis"""
    font_path = Path(FONT_PATH)

    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
        print("Font loaded.")
    else:
        print(f"WARNING: Font not found at {font_path}.")