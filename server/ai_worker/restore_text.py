
from transformers import pipeline

# load model once (important for performance)
restorer = pipeline(
    "text-generation",
    #model="prithivida/grammar_error_correcter_v1"
    model="t5-small"
)


def chunk_text(text, size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)

    return chunks


def restore_text(text):

    chunks = chunk_text(text)

    restored_chunks = []

    for chunk in chunks:
        prompt = "gec: " + chunk

        result = restorer(prompt, max_length=512)

        restored_chunks.append(result[0]['generated_text'])

    # join all chunks back
    final_text = "\n".join(restored_chunks)

    return final_text
    

'''
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load once
tokenizer = AutoTokenizer.from_pretrained(
    "prithivida/grammar_error_correcter_v1"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "prithivida/grammar_error_correcter_v1"
)


def chunk_text(text, size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))

    return chunks


def restore_text(text):

    chunks = chunk_text(text)

    restored_chunks = []

    for chunk in chunks:

        # IMPORTANT: model expects this prefix
        input_text = "gec: " + chunk
        #input_text = chunk

        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            truncation=True
        )

        outputs = model.generate(
            inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

        decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        restored_chunks.append(decoded)

    return "\n\n".join(restored_chunks)
'''
