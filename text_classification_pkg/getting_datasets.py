# Загрузка датасета
def getting_dataset() -> None:
    '''The result of executing this function is a dataset downloaded into the directory "Downloads"'''

    # Импорт библиотек
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from urllib.parse import urljoin
    import time

    # Инициализируем WebDriver:
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 YaBrowser/24.1.0.0 Safari/537.36'
    chrome_options = Options()
    chrome_options.add_argument(f'user-agent={USER_AGENT}')
    driver = webdriver.Chrome()

    main_url = 'https://www.kaggle.com'
    sign_in_url = 'https://www.kaggle.com/account/login'
    # Датасет находится на странице: https://www.kaggle.com/datasets/chaitanyakck/medical-text/data
    dataset_url = "https://www.kaggle.com/datasets/chaitanyakck/medical-text/data"

    try:
        # Перейдём на страницу входа в аккаунт Kaggle для авторизированного скачивания датасета
        driver.get(sign_in_url)

        # Нажмём на кнопку "Sign in with Google" для авторизации (подразумевая, что аккаунт уже зарегистрирован)
        google_sign_in_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(., "Sign in with Google")]'))
            )        
        google_sign_in_button.click()

        # Добавим ожидание с целью снижения нагрузки на сервер.
        time.sleep(3)

        try:
            # Проверим, что мы вошли в аккаунт и можем скачать датасет под своим аккаунтом.
            if driver.find_element((By.XPATH, '//h1[contains(., "Welcome")]')):

                # Теперь мы вошли в систему и можем переходить к дальнейшим действиям.
                # Откроем веб-страницу с датасетом:
                driver.get(dataset_url)

                # Найдём кнопку "Download":
                download_button = driver.find_element(By.XPATH, '//button[contains(., "file_download")]')
                # Если кнопка найдена, нажмём на неё
                if download_button:
                    download_button.click()
                # В противном случае воспользуемся альтернативным способом получения датасета - 
                # непосредственным ереходом по ссылке загрузки датасета
                else:
                    # найдём элемент, в котором содержится относительная ссылка
                    href_element = driver.find_element(By.XPATH, '//div[@class="sc-emfenM sc-fnpAPw cvuSKw gzjyQr"]/a')
                    # извлечём относительную ссылку
                    rel_link = href_element.get_attribute('href')
                    # составим абсолютный путь на скачивание архива
                    ds_download_link = urljoin(main_url, rel_link)
                    # перейдём по прямой ссылке загрузки
                    driver.get(ds_download_link)


                # Можно использовать аналогичный код:
                # В этом варианте кода заменён time.sleep() на явные ожидания WebDriverWait(), которые, возможно, являются более надежными.
                
                # try:
                #     download_button = WebDriverWait(driver, 10).until(
                #         EC.element_to_be_clickable((By.XPATH, '//button[contains(., "file_download")]'))
                #     )
                #     download_button.click()
                # except Exception as e:
                #     href_element = WebDriverWait(driver, 10).until(
                #         EC.presence_of_element_located((By.XPATH, '//div[@class="sc-emfenM sc-fnpAPw cvuSKw gzjyQr"]/a'))
                #     )
                #     rel_link = href_element.get_attribute('href')
                #     ds_download_link = urljoin(main_url, rel_link)
                #     driver.get(ds_download_link)

                # WebDriverWait(driver, 10).until(
                #     EC.presence_of_element_located((By.CLASS_NAME, "download-modal"))
                # )

                # Также, можно добавить контекстный менеджер with для инициализации и автоматического закрытия WebDriver после завершения работы функции.

        except Exception as e:
            print(f'Произошла ошибка в процессе поиска элемента на странице - {e}')
        # Подождём, пока загрузится файл:
        # Используем ожидание появления элемента с определенным классом, указывающим на загрузку
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "download-modal"))
        )
    except Exception as E:
        print(f'Произошла ошибка в процессе авторизации - {E}')
    # Закроем браузер в любом случае
    finally:
        driver.quit()


if __name__ == "__main__":
    getting_dataset()       