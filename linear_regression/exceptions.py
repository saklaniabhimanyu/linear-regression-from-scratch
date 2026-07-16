class ModelNotTrainedError(Exception):
    '''
    Raised when predict() is called before fit()
    '''
    pass

class InvalidShapeError(Exception):
    '''
    Raised when input shape is incompatible
    '''
    pass


class InvalidDataError(Exception):
    '''
    Raised when input contains invalid values
    '''
    pass
