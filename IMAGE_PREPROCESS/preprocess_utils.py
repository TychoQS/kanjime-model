import os
import matplotlib.pyplot as plt

class ImageOutputSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_figure(self, filename, subfolder=None):
        if subfolder is not None:
            os.makedirs(os.path.join(self.output_dir, subfolder), exist_ok=True)
            output_path = os.path.join(self.output_dir, subfolder, filename)
        else:
            output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
