from ultralytics.nn.tasks import yaml_model_load

d = yaml_model_load("yolov8-dat.yaml")
for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
    if m == "C2f_DAttn":
        print("layer index:", i)
        print("f (from):", f)
        print("n (repeats):", n)
        print("m (module):", m)
        print("args:", args)