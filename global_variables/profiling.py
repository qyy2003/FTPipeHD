static_profiler = None

def set_static_profiler(profiler_):
    global static_profiler
    static_profiler = profiler_


def get_static_profiler():
    global static_profiler
    return static_profiler
