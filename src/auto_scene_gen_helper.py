import os

for k in [43, 44, 45, 46]:
    os.system("python src/construct_pointclouds.py data/scene_%09d" % k)
