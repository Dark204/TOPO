import sys, types
def _install_optional_stubs() -> None:
    """Make the author's unused GUI/optional imports loadable on CPU runtime."""

    class _NoopAttrs:

        def __getattr__(self, _name: str) -> str:
            return ''
    if 'colorama' not in sys.modules:
        mod = types.ModuleType('colorama')
        mod.Back = _NoopAttrs()
        mod.Fore = _NoopAttrs()
        mod.Style = _NoopAttrs()
        mod.init = lambda *args, **kwargs: None
        sys.modules['colorama'] = mod
    if 'tkinter' not in sys.modules:
        mod = types.ModuleType('tkinter')
        mod.Tk = type('Tk', (), {'withdraw': lambda self: None, 'destroy': lambda self: None})
        filedialog = types.ModuleType('tkinter.filedialog')
        filedialog.askopenfilename = lambda *args, **kwargs: ''
        mod.filedialog = filedialog
        sys.modules['tkinter'] = mod
        sys.modules['tkinter.filedialog'] = filedialog
    if 'matplotlib' not in sys.modules:
        mod = types.ModuleType('matplotlib')
        mod.use = lambda *args, **kwargs: None
        pyplot = types.ModuleType('matplotlib.pyplot')
        mod.pyplot = pyplot
        sys.modules['matplotlib'] = mod
        sys.modules['matplotlib.pyplot'] = pyplot
    if 'plotly' not in sys.modules:
        mod = types.ModuleType('plotly')
        graph_objs = types.ModuleType('plotly.graph_objs')
        mod.graph_objs = graph_objs
        sys.modules['plotly'] = mod
        sys.modules['plotly.graph_objs'] = graph_objs
    if 'community' not in sys.modules:
        mod = types.ModuleType('community')
        mod.best_partition = lambda graph, *args, **kwargs: {}
        sys.modules['community'] = mod
    if 'adjustText' not in sys.modules:
        mod = types.ModuleType('adjustText')
        mod.adjust_text = lambda *args, **kwargs: None
        sys.modules['adjustText'] = mod
    if 'infomap' not in sys.modules:
        mod = types.ModuleType('infomap')

        class Infomap:

            def __init__(self, *args, **kwargs):
                self._modules = {}

            def addLink(self, *args, **kwargs):
                return None

            def run(self):
                return None

            def get_modules(self, *args, **kwargs):
                return self._modules
        mod.Infomap = Infomap
        sys.modules['infomap'] = mod
    if 'diskcache' not in sys.modules:
        mod = types.ModuleType('diskcache')

        class Cache(dict):

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False
        mod.Cache = Cache
        sys.modules['diskcache'] = mod
    if 'loguru' not in sys.modules:
        mod = types.ModuleType('loguru')

        class Logger:

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None
        mod.logger = Logger()
        sys.modules['loguru'] = mod
    if 'click' not in sys.modules:
        mod = types.ModuleType('click')
        mod.command = lambda *args, **kwargs: lambda function: function
        mod.option = lambda *args, **kwargs: lambda function: function
        sys.modules['click'] = mod
    if 'seaborn' not in sys.modules:
        sys.modules['seaborn'] = types.ModuleType('seaborn')
    if 'psutil' not in sys.modules:
        mod = types.ModuleType('psutil')

        class Process:

            def __init__(self, *args, **kwargs):
                pass

            def memory_info(self):
                return types.SimpleNamespace(rss=0)
        mod.Process = Process
        mod.cpu_percent = lambda *args, **kwargs: 0.0
        sys.modules['psutil'] = mod
