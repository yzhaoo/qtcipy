
# functions to recover a failed Julia execution



def retry(robust=True, initialize=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not robust:
                return func(*args, **kwargs)
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Something went wrong in Julia execution, retrying: {e}")
                    if initialize is not None:
                        initialize()
        return wrapper
    return decorator
