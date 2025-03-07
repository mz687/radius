import datasets, os

print("Loading openwebtxt...")
openwebtxt = datasets.load_dataset("Skylion007/openwebtext", split="train",trust_remote_code=True)
print("Finished loading openwebtxt...")
dump_path = "./openwebtxt"
os.makedirs(dump_path,exist_ok=True)
print("Saving json file...")
openwebtxt.to_json(os.path.join(dump_path,"openwebtxt.json"))
print("openwebtxt has been dumped to {}".format(os.path.join(dump_path,"openwebtxt.json")))