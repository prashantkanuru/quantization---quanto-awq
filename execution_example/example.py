phi3mini_128k="/home/ubuntu/ssl/ms_phi3mini_128k_model"
def create_directory_for_model_store(directory_path:str):
    "Creates the directory if it does not exists else states that the diectory exists"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        if os.path.exists(directory_path):
            print(f"Directory {directory_path} created successfully")
        else:
            print(f"Diectory could not be created, please check the path you have created")
    else:
        print(f"Directory {directory_path} already exists")


