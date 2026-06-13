import os, glob, f5_tts
print("f5_tts.__file__ =", getattr(f5_tts, "__file__", None))
paths = list(getattr(f5_tts, "__path__", []))
print("f5_tts.__path__ =", paths)
root = paths[0]
ex = os.path.join(root, "infer", "examples")
print("examples dir exists:", os.path.isdir(ex))
for f in sorted(glob.glob(os.path.join(ex, "**", "*"), recursive=True)):
    if os.path.isfile(f):
        print("  ", f.replace(root, "<pkg>"))
