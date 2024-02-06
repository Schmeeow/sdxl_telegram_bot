from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# проверяем, надо ли переводить промт (есть ли символы на русском)
def ru_detector(s):
    chars = set('абвгдеёжзиклмнопрстуфхцшщэюяъь')
    if any((c in chars) for c in s):
       return True
    else:
       return False

# переводим с русского на английский
def ru2en(ru_text):
    model_name = 'Helsinki-NLP/opus-mt-ru-en'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_ids = tokenizer.encode(ru_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=100)
    en_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Запрос на русском! Перевод: " + en_text + "\n")
    return(en_text)
