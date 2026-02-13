import matplotlib.pyplot as plt

class TrainingPlotter:
    def __init__(self, history):
        """
        Display plots about training process based of a certain history.
        """
        self.history = history
        self.epochs = range(1, len(history.get('train_acc', [])) + 1)

    def _plot_metric(self, train_key, val_key, title, ylabel, label_suffix, color_train='blue', color_val='orange'):
        """Generic internal method to generate all the plots."""
        if train_key not in self.history or val_key not in self.history:
            print(f"Warning: Keys {train_key} or {val_key} not found in history.")
            return

        plt.plot(self.epochs, self.history[train_key], label=f'Training {label_suffix}', color=color_train)
        plt.plot(self.epochs, self.history[val_key], label=f'Validation {label_suffix}', color=color_val)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.grid(True)

    def plot_loss(self):
        plt.figure(figsize=(6, 4))
        self._plot_metric('train_loss', 'val_loss', 'Training and Validation Loss', 'Loss', 'Loss', 'blue', 'orange')
        plt.show()

    def plot_kanji_accuracy(self):
        plt.figure(figsize=(6, 4))
        self._plot_metric('train_acc', 'val_acc', 'Kanji Classification Accuracy', 'Accuracy', 'Accuracy', 'blue', 'orange')
        plt.show()

    def plot_radical_accuracy(self):
        plt.figure(figsize=(6, 4))
        self._plot_metric('train_radical_acc', 'val_radical_acc', 'Radical Classification Accuracy', 'Accuracy', 'Radical Acc', 'green', 'red')
        plt.show()

    def plot_stroke_accuracy(self):
        plt.figure(figsize=(6, 4))
        self._plot_metric('train_stroke_acc', 'val_stroke_acc', 'Stroke Classification Accuracy', 'Accuracy', 'Stroke Acc', 'purple', 'brown')
        plt.show()

    def plot_all(self):
        """Generates a dashboard with the 4 plots in a single row."""
        plt.figure(figsize=(24, 6))
        
        plt.subplot(1, 4, 1)
        self._plot_metric('train_loss', 'val_loss', 'Training and Validation Loss', 'Loss', 'Loss')

        plt.subplot(1, 4, 2)
        self._plot_metric('train_acc', 'val_acc', 'Kanji Classification Accuracy', 'Accuracy', 'Accuracy')

        plt.subplot(1, 4, 3)
        self._plot_metric('train_radical_acc', 'val_radical_acc', 'Radical Classification Accuracy', 'Accuracy', 'Radical Acc', 'green', 'red')

        plt.subplot(1, 4, 4)
        self._plot_metric('train_stroke_acc', 'val_stroke_acc', 'Stroke Classification Accuracy', 'Accuracy', 'Stroke Acc', 'purple', 'brown')

        plt.tight_layout()
        plt.show()