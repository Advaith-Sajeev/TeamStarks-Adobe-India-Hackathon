import gdown

url = "https://drive.google.com/drive/folders/1ZWRZD7srpLPJzBEcogFj8R-bjpPTejRV?usp=sharing"
gdown.download_folder(url, quiet=False, use_cookies=False)