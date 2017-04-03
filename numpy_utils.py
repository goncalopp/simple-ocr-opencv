import numpy


class OverflowPreventer(object):
    """
    A context manager that exposes a numpy array preventing simple operations from overflowing
    Example:
    array= numpy.array( [255], dtype=numpy.uint8 )
    with OverflowPreventer( array ) as prevented:
        prevented+=1
    print array
    """

    inverse_operator = {'__iadd__': '__sub__', '__isub__': '__add__', '__imul__': '__div__', '__idiv__': '__mul__'}
    bypass_operators = ['__str__', '__repr__', '__getitem__']

    def __init__(self, matrix):
        class CustomWrapper(object):
            def __init__(self, matrix):
                assert matrix.dtype == numpy.uint8
                self.overflow_matrix = matrix
                self.overflow_lower_range = float(0)
                self.overflow_upper_range = float(2 ** 8 - 1)
                for op in OverflowPreventer.bypass_operators:
                    setattr(CustomWrapper, op, getattr(self.overflow_matrix, op))

            def _overflow_operator(self, b, forward_operator):
                m, lr, ur = self.overflow_matrix, self.overflow_lower_range, self.overflow_upper_range
                assert type(b) in (int, float)
                reverse_operator = OverflowPreventer.inverse_operator[forward_operator]
                uro = getattr(ur, reverse_operator)
                lro = getattr(lr, reverse_operator)
                afo = getattr(m, forward_operator)
                overflows = m > uro(b)
                underflows = m < lro(b)
                afo(b)
                m[overflows] = ur
                m[underflows] = lr
                return self

            def __getattr__(self, attr):
                if hasattr(self.wrapped, attr):
                    return getattr(self.wrapped, attr)
                else:
                    raise AttributeError

        self.wrapper = CustomWrapper(matrix)
        import functools
        for op in OverflowPreventer.inverse_operator.keys():
            setattr(CustomWrapper, op, functools.partial(self.wrapper._overflow_operator, forward_operator=op))

    def __enter__(self):
        return self.wrapper

    def __exit__(self, type, value, tb):
        pass
