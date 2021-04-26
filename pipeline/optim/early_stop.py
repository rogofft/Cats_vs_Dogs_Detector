def get_early_stop_detector(config: str, *args, **kwargs):
    if config == 'default':
        return EarlyStopDetector(*args, **kwargs)
    else:
        raise BaseException('Early stop detector type not found!')


# User's Early stop detectors


class EarlyStopDetector:
    def __init__(self, max_steps=5, reverse=False):
        self.reverse = reverse
        self.step = 0
        self.max_steps = max_steps

        if not self.reverse:
            # bigger value is better
            self.best_score = float('-inf')
        else:
            # smaller value is better
            self.best_score = float('inf')

    def check_for_best_score(self, score):
        if ((not self.reverse) and score > self.best_score) or (self.reverse and score < self.best_score):
            self.step = 0
            self.best_score = score
            result = True
        else:
            self.step += 1
            result = False

        return result

    def check_for_stop(self):
        return True if self.step >= self.max_steps else False
