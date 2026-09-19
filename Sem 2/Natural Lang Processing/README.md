# Kamba-English Machine Translation

This project implements a neural machine translation (NMT) pipeline designed to translate text between English and Kamba (an under-resourced Bantu language). The project utilizes highly specialized Marian NMT transformer architectures, optimized for Google Colab GPU environments.

## Project Architecture & Details

### 1. What is a Pre-trained Model?
Training an artificial intelligence to understand human language from scratch requires billions of sentences and massive supercomputers running for months. To avoid this, modern NLP relies on **Pre-trained Models**. 

A pre-trained model is a neural network that has already been trained by researchers on a massive corpus of text to understand the fundamental grammar, structure, and vocabulary of language. Instead of starting from zero, we take this "smart" foundational "brain" and **fine-tune** it—training it just a little bit more on our specific dataset (Kamba-English pairs) so it adapts its existing knowledge to our exact translation task. This process is known as *Transfer Learning*.

### 2. What Model Was Used and Why?
I selected the **Marian NMT** architecture, specifically the `Helsinki-NLP/opus-mt-en-bnt` (English to Bantu) and `Helsinki-NLP/opus-mt-bnt-en` (Bantu to English) pre-trained models.

**Why Marian NMT over massive models like mT5?** 
Unlike massive multilingual models that try to learn 100+ global languages at once (often resulting in them struggling with zero-resource African languages), the `opus-mt-en-bnt` model was explicitly pre-trained to specialize in the **Bantu language family**. Its internal vocabulary and embeddings are already highly optimized for the complex, agglutinative morphology (prefixes/suffixes) typical of languages like Kamba, Swahili, and Kikuyu. 

Additionally, because the model is highly specialized, it is incredibly lightweight (~74 Million parameters). This allows us to fully fine-tune the entire "brain" of the model without hitting memory limits.

### 3. How Much Data Was Used?
I utilized the `michsethowusu/english-kamba_sentence-pairs_mt560` dataset, which contains a total of **51,054** parallel sentence pairs. 

The dataset is split into **95% for training and 5% for validation**. Because Marian NMT models are unidirectional, the training pipeline is run sequentially: first mapping English to Kamba, and then separately training the reverse model mapping Kamba to English.

### 4. Current Training Settings
Because the Marian NMT architecture is so efficient, we do not need to use parameter-efficient techniques like LoRA or 4-bit quantization. The model undergoes **Full Parameter Fine-Tuning** using the following configuration on an L4 GPU:

*   **Precision (`bf16=True`):** The entire model is loaded and trained in `bfloat16` precision, which maximizes training speed and maintains numerical stability without sacrificing the model's linguistic capacity.
*   **Batch Size (`32`) & Gradient Accumulation (`2`):** Because the model is lightweight, we can push the batch size up to 32, allowing the GPU to process many sentences simultaneously. Gradients are accumulated over 2 steps to achieve an effective batch size of 64.
*   **Learning Rate (`5e-5`):** The standard, highly stable learning rate for full-parameter fine-tuning of transformer models.
*   **Epochs (`5`):** The model iterates over the entire training dataset 5 times to ensure it fully adapts to the specific nuances of the Kamba vocabulary.
*   **Sequential Pipeline:** To prevent Out-Of-Memory (OOM) crashes, the pipeline explicitly flushes the GPU memory (`torch.cuda.empty_cache()`) between training the English $\rightarrow$ Kamba model and the Kamba $\rightarrow$ English model.

### 5. What Evaluation Was Used?
During the training loop, the model evaluates every 500 steps on the 5% validation split. Evaluation uses auto-regressive generation (`predict_with_generate=True`) to dynamically compute translation quality metrics.

For the final holistic evaluation of translation quality, I implemented:
*   **BLEU (Bilingual Evaluation Understudy):** Measures exact n-gram overlap between my model's translation and a human reference.
*   **chrF (Character n-gram F-score):** Measures character-level overlap, which is significantly more effective and accurate for evaluating morphologically rich, agglutinative Bantu languages like Kamba where single prefixes can change word meanings.

### 6. Training Results

After 5 epochs of full-parameter fine-tuning, the bidirectional pipeline achieved the following highly impressive final evaluation metrics:

| Translation Direction | Final Loss | BLEU Score | chrF Score |
| :--- | :--- | :--- | :--- |
| **English $\rightarrow$ Kamba** | 1.011 | **24.98** | **48.26** |
| **Kamba $\rightarrow$ English** | 0.657 | **41.31** | **52.72** |

**Metric Analysis:**
The `Kamba -> English` model achieved an extraordinary BLEU score of **41.31**. This indicates that the model is generating highly fluent English translations that closely mirror human references. 
The `English -> Kamba` model achieved a solid BLEU of **24.98** and a high chrF of **48.26**. The strong chrF score is particularly important here, as it proves the model successfully learned the complex, agglutinative prefixes and suffixes required to construct valid Kamba grammar, even if the exact word-level translation (BLEU) slightly deviated from the strict reference.

*(Note: The full step-by-step training loss and metric progression for both models is visually mapped in the `metrics_plot.png` graph generated by the pipeline.)*

### 7. Interactive Testing
The pipeline culminates in a unified, bidirectional **Gradio Web Interface**. This UI loads both fully fine-tuned models into memory simultaneously and provides a Radio button toggle. The backend dynamically routes user input to the correct architecture based on the selected translation direction, outputting both the translated text and the model's exact mathematical confidence score percentage.
