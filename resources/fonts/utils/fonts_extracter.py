import os
import shutil
from pathlib import Path
from fontTools.ttLib import TTFont
from tqdm import tqdm

REPO_PATH = Path("google_fonts_repo/ofl") # Directory with OFL license fonts
DEST_PATH = Path("./extracted_fonts") # Directory to save extracted fonts
KANJI_FILE = Path("../../training/training_output/kanjis.txt") # Kanjis supported by model

DEST_PATH.mkdir(parents=True, exist_ok=True)

def check_font_coverage(font_path, target_chars):
    """Checks if a font supports all the target kanjis"""
    try:
        font = TTFont(font_path)
        supported_chars = set()
        for table in font['cmap'].tables:
            supported_chars.update(table.cmap.keys())
        
        for char in target_chars:
            if ord(char) not in supported_chars:
                return False
        return True
    except Exception:
        return False

# Loading supported kanjis
if not KANJI_FILE.exists():
    exit()

with open(KANJI_FILE, "r", encoding="utf-8") as f:
    target_kanjis = "".join(f.read().split())

# Loading fonts
folders = list(REPO_PATH.iterdir())

for folder in tqdm(folders):
    if not folder.is_dir():
        continue
    
    is_valid_japanese = False
    is_ofl = False
    metadata_file = folder / "METADATA.pb"
    
     # Checking in metadata if is a japanese font and OFL
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
                if 'license: "ofl"' in content:
                    is_ofl = True
                if 'subsets: "japanese"' in content or 'primary_script: "jpan"' in content:
                    is_valid_japanese = True
        except:
            continue

    if not is_valid_japanese and is_ofl:
        folder_name = folder.name.lower()
        keywords = ["jp", "japanese", "kanji", "mincho", "gothic"]
        if any(word in folder_name for word in keywords):
            is_valid_japanese = True

    if is_valid_japanese and is_ofl:
        for font_file in folder.glob("*.ttf"):
            # If font supports all the target kanjis is copied to the destination path
            if check_font_coverage(font_file, target_kanjis):
                clean_name = font_file.name.replace(" ", "_")
                shutil.copy(font_file, DEST_PATH / clean_name)