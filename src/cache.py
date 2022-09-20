import pickle
import os


class Cache:

  def __init__(self, path: str, log_fn = None):
    self.path = path
    self.log_fn = log_fn or (lambda x: x)
    self.registry = {}
  
  def register_tuple(self, object_names, work_fn):
    for object_name in object_names:
      self.registry[object_name] = (object_names, work_fn)
  
  def register(self, object_name, work_fn):
    self.register_tuple((object_name,), lambda: (work_fn(),))

  def get(self, object_name: str):
    obj = self.load(object_name)
    if obj is not None:
      return obj
    object_names, work_fn = self.registry[object_name]
    self.log_fn(f"Computing {object_name}...")
    results = work_fn()
    for result, obj_name in zip(results, object_names):
      self.save(result, obj_name)
      if obj_name == object_name:
        final_result = result
    return final_result

  def save(self, obj, object_name: str):
    if not os.path.exists(self.path):
      os.makedirs(self.path)
    filename = os.path.join(self.path, object_name + ".pkl")
    with open(filename, "wb") as fh:
      self.log_fn(f"Saving {filename}...")
      pickle.dump(obj, fh)

  def load(self, object_name: str):
    filename = os.path.join(self.path, object_name + ".pkl")
    if not os.path.exists(filename):
      return None
    with open(filename, "rb") as fh:
      self.log_fn(f"Loading {filename}...")
      return pickle.load(fh)


if __name__ == "__main__":
  cache = Cache(".cache", log_fn=print)
  cache.register("one", lambda: 1)
  one = cache.get("one")
  assert one == 1
  assert os.path.exists(".cache/one.pkl")