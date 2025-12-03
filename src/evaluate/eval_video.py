"""Video evaluation script (demo)."""
from .eval_text import eval_predictions


def eval_video_demo():
    import numpy as np
    traits_true = np.random.rand(8, 5)
    traits_pred = traits_true + 0.02 * np.random.randn(*traits_true.shape)
    emotions_true = np.zeros(8, dtype=int)
    emotions_pred = np.zeros(8, dtype=int)
    eval_predictions(traits_true, traits_pred, emotions_true, emotions_pred)


if __name__ == "__main__":
    eval_video_demo()
