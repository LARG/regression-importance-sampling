

class ReplayBuffer(object):

    def __init__(self, max_size=None, discard_rule=None):

        self._max_size = max_size
        self._discard_rule = discard_rule
        if self._max_size is not None and self._discard_rule is None:
            self._discard_rule = 'recency'
        self._paths = []
        self._returns = []

    @property
    def paths(self):
        return self._paths

    @property
    def size(self):
        return len(self._paths)

    def add_path(self, path, ret):
        self._paths.append(path)
        self._returns.append(ret)
        if self._max_size is not None and self.size >= self._max_size:
            if self._discard_rule == 'recency':
                self._paths.pop(0)
            elif self._discard_rule == 'priority':
                min_return = min(self._returns)
                min_ind = self._returns.index(min_return)
                self._paths.pop(min_ind)

    def empty(self):
        self._paths = []
        self._returns = []
