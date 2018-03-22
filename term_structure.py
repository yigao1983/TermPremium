import abc


class Parameters(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TermStructure(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.__params = Parameters()
        for key, val in kwargs.items():
            self.__params.setdefault(key, val)

    @property
    def params(self):
        return self.__params

    @abc.abstractmethod
    def get_yield(self, tenor, **kwargs):
        pass

    @abc.abstractmethod
    def get_fwd_rate(self, fwd_time, tenor, **kwargs):
        pass

    @abc.abstractmethod
    def get_expected_yield(self, fwd_time, tenor, **kwargs):
        pass

    def update_param(self, **kwargs):

        for param_key, param in kwargs.items():
            if param_key in self.__params:
                self.__params.update({param_key: param})
