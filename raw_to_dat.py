import os

def main(path = "./"):
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith(".raw"):
				print(os.path.join(root, file))
				os.system(f"metavision_raw_to_dat -i '{os.path.join(root,file)}'")
				print("*"*20)
	print("Done")


if __name__=="__main__":
	main()
