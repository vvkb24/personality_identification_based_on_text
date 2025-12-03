"""Audio evaluation script (demo)."""
from .eval_text import eval_predictions


def eval_audio_demo():
    # Demo stub: in real usage, load model outputs
    import numpy as np
    traits_true = np.random.rand(10, 5)
    traits_pred = traits_true + 0.05 * np.random.randn(*traits_true.shape)
    emotions_true = np.zeros(10, dtype=int)
    emotions_pred = np.zeros(10, dtype=int)
    eval_predictions(traits_true, traits_pred, emotions_true, emotions_pred)


if __name__ == "__main__":
    eval_audio_demo()
