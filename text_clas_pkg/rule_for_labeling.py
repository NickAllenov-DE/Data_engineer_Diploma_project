# Проводим разметку на основе правил.
def rule_for_labeling(text: str) -> str:
    '''The function defines a rule for assigning a label to the text and performs markup'''
    
    # Определяем списки с ключевыми значениями по каждой из четырех категорий 
    neoplasms_list = [
        'neoplas', 'tumor', 'cancer', 'lymphom', 'blastoma', 'malign', 'benign', 'melanom', 'leukemi', 'metasta', 'carcinom', 'sarcoma', 'glioma',
        'adenoma', 'chemotherapy', 'radiotherapy', 'oncology', 'carcinogenesis', 'mutagen', 'angiogenesis', 'radiation', 'immunotherapy', 'biopsy',
        'brachytherapy', 'metastasis', 'prognosis', 'biological therapy', 'carcinoma', 'myeloma', 'genomics', 'immunology', 'cell stress',
        'oncogene', 'tumorigenesis', 'cytology', 'histology', 'oncologist', 'neoplasm', 'oncogenic', 'tumor suppressor genes', 'malignancy',
        'cancerous', 'non-cancerous', 'solid tumor', 'liquid tumor', 'tumor marker', 'oncogenesis', 'tumor microenvironment', 'carcinogenesis', 
        'adenocarcinoma', 'squamous cell carcinoma'
    ]

    digestive_list = [
        'digestive', 'esophag', 'stomach', 'gastr', 'liver', 'cirrhosis', 'hepati', 'pancrea', 'intestin', 'sigmo', 'recto', 'rectu', 'cholecyst', 
        'gallbladder', 'portal pressure', 'portal hypertension', 'appendic', 'ulcer', 'bowel', 'dyspepsia', 'colitis', 'enteritis', 'gastroenteritis', 
        'endoscopy', 'colonoscopy', 'peptic', 'gastrointestinal', 'abdominal', 'dysphagia', 'diverticulitis', 'irritable bowel syndrome', 
        'inflammatory bowel disease', 'gastroesophageal reflux', 'celiac disease', 'crohn\'s disease', 'ulcerative colitis',
        'gastroscopy', 'biliary', 'esophageal', 'gastritis', 'hepatic', 'lactose intolerance', 'gastroenterologist', 'digestion', 'absorption', 
        'malabsorption', 'intestinal flora', 'microbiota', 'probiotics', 'prebiotics', 'dietary fiber', 'nutrition'
    ]

    neuro_list = [
        'neuro', 'nerv', 'reflex', 'brain', 'cerebr', 'white matter', 'subcort', 'plegi', 'intrathec', 'medulla', 'mening', 'epilepsy', 
        'multiple sclerosis', 'parkinson\'s disease', 'alzheimer\'s disease', 'seizure', 'paresthesia', 'dementia', 'encephalopathy', 
        'neuropathy', 'neurodegeneration', 'stroke', 'cerebral', 'spinal cord', 'neurotransmitter', 'synapse', 'neuralgia', 'neurology', 
        'neurosurgery', 'neurooncology', 'neurovascular', 'autonomic nervous system', 'central nervous system', 'peripheral nervous system', 
        'brain injury', 'concussion', 'traumatic brain injury', 'spinal injury', 'neurological disorder', 'neurodevelopmental disorders',
        'neurodegenerative disorders', 'neuroinflammation', 'neuroimaging', 'neuroscience', 'neurophysiology', 'neurotransmission', 
        'neuroplasticity', 'neurogenesis', 'neuroendocrinology', 'neuropsychology', 'neurotoxicity', 'neuromodulation', 'neuroprotection', 
        'neuropathology'
    ]

    cardio_list = [
        'cardi', 'heart', 'vascul', 'embolism', 'stroke', 'reperfus', 'thromboly', 'ischemi', 'hypercholesterolemia', 'hyperten', 'blood pressure', 
        'valv', 'ventric', 'aneurysm', 'coronar', 'arter', 'aort', 'electrocardiogra', 'arrhythm', 'clot', 'mitral', 'endocard', 'hypertension', 
        'myocardial', 'infarction', 'cardiover', 'fibrillat', 'bypass', 'pericarditis', 'cardiomyopathy', 'hypotension', 'angiography', 'stenting', 
        'cardiac catheterization', 'vascular', 'echocardiogram', 'cardiogenic', 'angioplasty', 'cardiac arrest', 'heart failure', 
        'cardiac rehabilitation', 'electrophysiology', 'heart valve disease', 'cardiopulmonary', 'cardiothoracic surgery', 'vascular surgery', 
        'cardiovascular disease', 'cardiovascular health', 'cardiovascular risk', 'cardiovascular system', 'cardioprotection', 'cardiovascular imaging', 
        'cardiovascular physiology', 'cardiovascular pharmacology', 'cardiovascular intervention', 'cardiovascular diagnostics', 'cardiovascular genetics'
    ]

    # Приведем текст аннотаций к нижнему регистру
    row = text.lower()
    
    # В используемом датасете используется следующая маркировка:
    # neoplasms = 1
    # digestive system diseases = 2
    # nervous system diseases = 3
    # cardiovascular diseases = 4
    # general pathological conditions = 5

    # Создаём словарь в котором ключи - категории заболеваний, а значения - количество ключевых значений в тексте по каждой категории
    res_dict = {
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0
    }
    # Рассчитываем количество ключевых значений в тексте и заполняем словарь
    for p in neoplasms_list:
        res_dict['1'] += row.count(p)
    for d in digestive_list:
        res_dict['2'] += row.count(d)
    for n in neuro_list:
        res_dict['3'] += row.count(n)
    for c in cardio_list:
        res_dict['4'] += row.count(c)
    
    # Рассчитываем наиболее часто встречаемую категорию в тексте и её отношение ко всем выявленным значения по всем категориям.
    # Для отнесения текста к определенной категории его доля должна превышать условно взятое значение - 0,3.
    # Если не превышает, то текст будет отнесён к категории 'general pathological conditions' и ему будет присвоена метка - 5
    most_frequent = max(res_dict.values())
    divisor = sum(res_dict.values())
    if divisor > 0 and (most_frequent / divisor) > 0.3:
        for key, value in res_dict.items(): 
            if value == most_frequent:
                return int(key)
    else:
        return int(5)