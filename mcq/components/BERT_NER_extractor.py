from transformers import BertTokenizer, BertForTokenClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import pipeline
class NER_extractor():

    def __init__(this,
                 model: BertForTokenClassification,
                 tokenizer: BertTokenizer,):
        this.model = model
        this.tokenizer = tokenizer
        this.ner_pipeline = pipeline(task="ner",
                                     model=this.model,
                                     tokenizer=this.tokenizer)
        this.threshold = 0.97

    def aggregate_entities(this, ner_results):
        aggregated_entities = []
        current_entity = None
        
        for token in ner_results:
            entity_type = token['entity'].split('-')[-1] 
            if (entity_type == "ORD" or entity_type == "CRD") or token["score"] < this.threshold: 
                continue
            if current_entity and token['entity'].startswith("I-") and entity_type == current_entity['entity']:
                current_entity['word'] += " " + token['word']
            else:
                if current_entity:
                    aggregated_entities.append(current_entity["word"])
                current_entity = {"entity": entity_type, "word": token['word']}
        
        if current_entity:
            aggregated_entities.append(current_entity["word"])
        
        return aggregated_entities

    def __call__(this, text: str, **kwargs):
        return this.aggregate_entities(this.ner_pipeline(text))
"""
if __name__ == "__main__":
    model = BertForTokenClassification.from_pretrained("cahya/bert-base-indonesian-NER")
    tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-NER")

    ner = NER_extractor(model,tokenizer)
    text = "Sejarah Indonesia dimulai sejak zaman prasejarah ketika manusia pertama kali mendiami Nusantara sekitar 1,5 juta tahun yang lalu. Salah satu peradaban paling awal yang tercatat adalah Kerajaan Kutai di Kalimantan Timur pada abad ke-4 Masehi. Kemudian muncul kerajaan-kerajaan besar lainnya seperti Sriwijaya di Sumatera dan Majapahit di Jawa, yang masing-masing mencapai puncak kejayaannya pada abad ke-7 dan ke-14. Pengaruh Hindu-Buddha sangat kuat pada periode ini, yang kemudian diikuti oleh masuknya agama Islam pada abad ke-13 melalui para pedagang dari Arab dan Gujarat. Kolonialisasi Eropa dimulai pada abad ke-16 dengan kedatangan Portugis, diikuti oleh Belanda yang menjajah Indonesia selama lebih dari 300 tahun. Indonesia akhirnya meraih kemerdekaan pada 17 Agustus 1945 setelah perjuangan panjang melawan penjajahan."

    ner_pred = ner(text)
    print(ner_pred)
"""