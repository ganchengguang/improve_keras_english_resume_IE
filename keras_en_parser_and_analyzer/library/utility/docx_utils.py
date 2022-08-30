import docx
from tensorboard import errors

from keras_en_parser_and_analyzer.library.utility.text_utils import preprocess_text


def docx_to_text(file_path):
    #doc = open(file_path,encoding='UTF-8')
    #file = doc.readlines()
    #result = []
    #for p in file:
    #    txt = p.strip()
    #    if txt != '':
    #        txt = preprocess_text(txt)
    #        result.append(txt)
    #return result
    doc = docx.Document(file_path)
    result = []
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt != '':
            txt = preprocess_text(txt)
            result.append(txt)
    return result