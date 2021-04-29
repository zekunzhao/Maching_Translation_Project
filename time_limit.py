import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass


def long_function_call():
    a = 0
    a = a+1
        # print("running!")
    return

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


try:
    with time_limit(10):
        long_function_call()
except TimeoutException as e:
    print("Timed out!")
