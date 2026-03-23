import os
import matplotlib.pyplot as plt


class ImageOutputSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_mosaic(self, steps, filename, subfolder=None):
        fig, axes = plt.subplots(1, len(steps), figsize=(3 * len(steps), 4))
        for ax, (image, title) in zip(axes, steps):
            ax.imshow(image, cmap=None if image.ndim == 3 else "gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        plt.tight_layout()

        if subfolder is not None:
            os.makedirs(os.path.join(self.output_dir, subfolder), exist_ok=True)
            output_path = os.path.join(self.output_dir, subfolder, filename)
        else:
            output_path = os.path.join(self.output_dir, filename)

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")