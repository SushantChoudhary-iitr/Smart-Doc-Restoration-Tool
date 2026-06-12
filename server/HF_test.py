from transformers import pipeline

# load model once (important for performance)
restorer = pipeline(
    "text-generation",
    #model="prithivida/grammar_error_correcter_v1"
    model="Qwen/Qwen2.5-1.5B"
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
        #prompt = "gec: " + chunk
        prompt = chunk

        result = restorer(prompt, max_length=512)

        restored_chunks.append(result[0]['generated_text'])

    # join all chunks back
    final_text = "\n".join(restored_chunks)

    return final_text

if __name__ == "__main__":
    text = "This is a test sentence with a grammar error."
    restored_text = restore_text(text)
    print(f"here si the restored text: {restored_text}")

