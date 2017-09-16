def _same_type(a, b):
    type_correct = False
    if type(a) == type(b):
        type_correct = True
    try:
        if isinstance(a, b):
            type_correct = True
    except TypeError:  # v may not be a class or type, but an int, a string, etc
        pass
    return type_correct


def _broadcast(src_processor, src_atr_name, dest_processors, dest_atr_name, transform_function):
    """
    To be used exclusively by create_broadcast.
    A broadcast function gets an attribute on the src_processor and
    sets it (possibly under a different name) on dest_processors
    """
    value = getattr(src_processor, src_atr_name)
    value = transform_function(value)
    for d in dest_processors:
        setattr(d, dest_atr_name, value)


def create_broadcast(src_atr_name, dest_processors, dest_atr_name=None, transform_function=lambda x: x):
    """
    This method creates a function, intended to be called as a
    Processor posthook, that copies some of the processor's attributes
    to other processors
    """
    from functools import partial
    if dest_atr_name == None:
        dest_atr_name = src_atr_name
    if not hasattr(dest_processors, "__iter__"):  # a single processor was given instead
        dest_processors = [dest_processors]
    return partial(_broadcast, src_atr_name=src_atr_name, dest_processors=dest_processors, dest_atr_name=dest_atr_name,
                   transform_function=transform_function)


class Parameters(dict):
    def __add__(self, other):
        d3 = Parameters()
        d3.update(self)
        d3.update(other)
        return d3


class Processor(object):
    """
    In goes something, out goes another. Processor.process() models
    the behaviour of a function, where there are some stored parameters
    in the Processor instance. Further, it optionally calls arbitrary
    functions before and after processing (prehooks, posthooks)
    """

    PARAMETERS = Parameters()

    def __init__(self, **args):
        """sets default parameters"""
        for k, v in self.PARAMETERS.items():
            setattr(self, k, v)
        self.set_parameters(**args)
        self._prehooks = []  # functions (on input) to be executed before processing
        self._poshooks = []  # functions (on output) to be executed after processing

    def get_parameters(self):
        """returns a dictionary with the processor's stored parameters"""
        parameter_names = self.PARAMETERS.keys()
        # TODO: Unresolved reference for processor
        parameter_values = [getattr(processor, n) for n in parameter_names]
        return dict(zip(parameter_names, parameter_values))

    def set_parameters(self, **args):
        """sets the processor stored parameters"""
        for k, v in self.PARAMETERS.items():
            new_value = args.get(k)
            if new_value != None:
                if not _same_type(new_value, v):
                    raise Exception(
                        "On processor {0}, argument {1} takes something like {2}, but {3} was given".format(self, k, v,
                                                                                                            new_value))
                setattr(self, k, new_value)
        not_used = set(args.keys()).difference(set(self.PARAMETERS.keys()))
        not_given = set(self.PARAMETERS.keys()).difference(set(args.keys()))
        return not_used, not_given

    def _process(self, arguments):
        raise NotImplementedError(str(self.__class__) + "." + "_process")

    def add_prehook(self, prehook_function):
        self._prehooks.append(prehook_function)

    def add_poshook(self, poshook_function):
        self._poshooks.append(poshook_function)

    def process(self, arguments):
        self._input = arguments
        for prehook in self._prehooks:
            prehook(self)
        output = self._process(arguments)
        self._output = output
        for poshook in self._poshooks:
            poshook(self)
        return output


class DisplayingProcessor(Processor):
    def display(self, display_before=False):
        """
        Show the last effect this processor had - on a GUI, for
        example. If show_before is True, show the "state before
        processor"
        """
        raise NotImplementedError


class ProcessorStack(Processor):
    """a stack of processors. Each processor's output is fed to the next"""

    def __init__(self, processor_instances=[], **args):
        self.set_processor_stack(processor_instances)
        Processor.__init__(self, **args)

    def set_processor_stack(self, processor_instances):
        assert all(isinstance(x, Processor) for x in processor_instances)
        self.processors = processor_instances

    def get_parameters(self):
        """gets from all wrapped processors"""
        d = {}
        for p in self.processors:
            parameter_names = list(p.PARAMETERS.keys())
            parameter_values = [getattr(p, n) for n in parameter_names]
            d.update(dict(zip(parameter_names, parameter_values)))
        return d

    def set_parameters(self, **args):
        """sets to all wrapped processors"""
        not_used = set()
        not_given = set()
        for p in self.processors:
            nu, ng = p.set_parameters(**args)
            not_used = not_used.union(nu)
            not_given = not_given.union(ng)
        return not_used, not_given

    def _process(self, arguments):
        for p in self.processors:
            arguments = p.process(arguments)
        return arguments


class DisplayingProcessorStack(ProcessorStack):
    def display(self, display_before=False):
        if display_before:
            pr = self.processors[1:]
            self.processors.display(display_before=True)
        else:
            pr = self.processors
        for p in pr:
            if hasattr(p, "display"):
                p.display(display_before=False)
