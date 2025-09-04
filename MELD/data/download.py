from transformers import AutoModel, AutoProcessor, AutoImageProcessor, RobertaTokenizer, RobertaModel

# # Audio model and processor
# print('Downloading facebook/data2vec-audio-base-960h...')
# AutoModel.from_pretrained('facebook/data2vec-audio-base-960h')
# AutoProcessor.from_pretrained('facebook/data2vec-audio-base-960h')
# print("Done")

# # Video model and processor
# print('Downloading facebook/timesformer-base-finetuned-k400...')
# AutoModel.from_pretrained('facebook/timesformer-base-finetuned-k400')
# AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k400')
# print("Done")
# # Text model and tokenizer
print('Downloading roberta-large...')
RobertaModel.from_pretrained('roberta-large')
print("Done 1")
RobertaTokenizer.from_pretrained('roberta-large')
print("Done 2")
print('All models downloaded to the default Hugging Face cache.') 