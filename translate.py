from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def ru2en(ru_text):
    model_name = 'Helsinki-NLP/opus-mt-ru-en'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_ids = tokenizer.encode(ru_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=100)
    en_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return(en_text)
