import importlib.util, os, sys
spec=importlib.util.spec_from_file_location("appmod","Rice Leaf Disease Detection System/app.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sol={'cause':'Test cause','treatment':'Test treatment','severity_note':'note'}
en,hi,kn=mod.generate_voice(sol,'Blast','Moderate')
print('saved:', en, hi, kn)
print('exists:', [os.path.exists(p) for p in (en,hi,kn)])
print('dir:', os.listdir(os.path.dirname(en)))
