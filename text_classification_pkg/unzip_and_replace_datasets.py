# Разархивирование загруженного архива в рабочую директорию
def unzip_and_replace_datasets(zip_path: str ="C:\\Users\\Allen\\Downloads\\archive.zip", 
                              extract_to: str = "D:\\GeekBrains\\Data_engineer_Diploma_project") -> None:
    '''The function unzips the downloaded archive into the working directory'''

    # Импорт библиотек
    import zipfile
    import os

    # Проверка существования файла
    if not os.path.exists(zip_path):
        print(f"The file {zip_path} does not exist.")
        return

    # Разархивирование архива
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Archive extracted to {extract_to}")
    except zipfile.BadZipFile:
        print("The file is a bad zip file and cannot be extracted.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    zip_file_path = "C:\\Users\\Allen\\Downloads\\archive.zip"
    destination_directory = "D:\\GeekBrains\\Data_engineer_Diploma_project"
    unzip_and_replace_datasets(zip_file_path, destination_directory)  
