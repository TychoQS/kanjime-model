# Script to extract japanese fonts from google fonts
git clone --depth 1 https://github.com/google/fonts.git google_fonts_repo
conda run -n TFG-Preprocess python3 utils/fonts_extracter.py
rm -rf google_fonts_repo
