
# Загрузка датасета через API

def getting_dataset_by_api(ds_name: str = 'chaitanyakck/medical-text', path: str = os.getcwd(), unzip: bool = True) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(ds_name, path, unzip)