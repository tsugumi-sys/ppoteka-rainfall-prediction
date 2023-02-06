import contextlib
import joblib


@contextlib.contextmanager
def custom_progressbar(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwards):
            super().__init__(*args, **kwards)

        def __call__(self, *args, **kwards):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwards)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()