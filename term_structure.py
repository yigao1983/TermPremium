import abc


class TermStructure(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.__params = {}
        for key, val in kwargs.items():
            self.__params.setdefault(key, val)

    @property
    def params(self):
        return self.__params

    @abc.abstractmethod
    def get_yield(self, time2mat, **kwargs):
        pass

    @abc.abstractmethod
    def get_expected_yield(self, fwd_time, tenor, **kwargs):
        pass
