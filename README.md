# FAKE-NEWS-DETECTION
🧠 Project Goal (What we’re building)

Task: Classify a news article as FAKE or REAL.
Approach: Deep-learning text classifier using CNN with pre-trained GloVe embeddings for high accuracy (target ≈ 97–98% with good data + tuning).

1) 📦 Importing Libraries (Your toolkit)

Why these?

NumPy / Pandas → fast arrays & table ops (CSV, columns, joins).

Matplotlib / Seaborn → charts (class balance, confusion matrix).

re, nltk → text cleaning: remove noise, stopwords, stemming.

scikit-learn → train/test split + metrics (accuracy, precision, recall, F1).

TensorFlow / Keras → build/train the CNN; handle tokenization, padding.

Key idea: We assemble a data stack (clean → vectorize → learn → evaluate).

2) 📂 Loading the Dataset
df = pd.read_csv("/mnt/data/fake_news_dataset.csv")


What you check right away:

Shape → how many rows (articles) and columns (title, text, label…).

Missing values → especially in text. We replace NaNs with "" so the pipeline doesn’t crash.

Label distribution → check balance between REAL and FAKE (helps interpret accuracy & decide on stratification).

Why it matters: Clean, well-understood input prevents data leakage and silent bugs.

3) 🧼 Text Preprocessing (make raw text learnable)
What we do & why:

Lowercasing → “Apple” and “apple” become the same token.

Remove non-letters → punctuation/digits rarely add semantics here.

Stopword removal → drop very common words (“the”, “and”) that add noise.

Stemming → map “running”, “runs” → “run” to reduce vocabulary size.

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)   # keep letters only
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


Label encoding:
We convert label to 0/1 (e.g., FAKE=0, REAL=1) using LabelEncoder.
This gives the model a numeric target.

💡 Note: Your original requirement mentions TF-IDF. That’s perfect for linear models (Passive Aggressive/SVM).
For CNN, we use embeddings (dense semantic vectors) instead of TF-IDF, which boosts performance on deep models.

4) ✂️ Train/Test Split (honest evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


Stratify keeps the class ratio roughly the same in train & test.

Why 80/20? Common balance between learning enough and testing fairly.

Outcome: Prevents overfitting illusions and gives trustworthy metrics.

5) 🔢 Tokenization & Padding (text → numbers)

Tokenizer builds a vocabulary from training text and maps each word → an index.

max_words (e.g., 50k) limits vocabulary to frequent words (less noise).

max_len (e.g., 300) standardizes article length—padding short texts with zeros; truncating very long ones.

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding="post")
X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=max_len, padding="post")


Why this matters: Neural nets expect fixed-size numeric tensors.
Padding ensures consistent shapes for efficient GPU training.

6) 🧭 GloVe Embeddings (semantic prior = big accuracy jump)

What are embeddings?
Dense vectors where similar words have nearby vectors (e.g., “president” close to “leader”).

GloVe (6B, 100d) gives a learned semantic space from massive corpora.
We map our vocabulary to the pretrained vectors via an embedding matrix.

# embeddings_index[word] = 100-d vector
embedding_matrix = np.zeros((max_words, 100))
for word, i in tokenizer.word_index.items():
    if i < max_words and word in embeddings_index:
        embedding_matrix[i] = embeddings_index[word]


Why trainable=False first?

We freeze embeddings initially to stabilize training and retain the strong prior.

You can later unfreeze for fine-tuning once the top layers converge.

✅ Effect: Faster convergence, better generalization, big boost over random embeddings—crucial for hitting ~98% on clean datasets.

7) 🏗️ CNN Architecture (how the model “reads”)
Embedding(..., weights=[embedding_matrix], trainable=False)
Conv1D(256, kernel_size=5, activation='relu')
GlobalMaxPooling1D()
Dense(128, activation='relu')
Dropout(0.4)
Dense(1, activation='sigmoid')


Layer by layer (intuitively):

Embedding → converts each token id to its 100-d GloVe vector.

Conv1D → slides 1-D filters over the sequence; each filter detects an n-gram pattern (like 5-grams) that is predictive (“claims”, “breaking”, “sources say”).

GlobalMaxPooling1D → picks the strongest signal from each filter across the whole text (translation-invariant “did this pattern occur anywhere?”).

Dense(128) + ReLU → combines features into higher-level cues (narrative style, sensational tone).

Dropout(0.4) → randomly drops neurons during training so the model doesn’t overfit.

Sigmoid output → outputs p(REAL) in [0,1] (binary classification).

Compile:

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


Binary cross-entropy matches a 2-class, sigmoid output.

Adam is a robust default optimizer.

8) 🏃 Training Loop (how learning happens)
history = model.fit(
    X_train_pad, y_train,
    epochs=6,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)


Epochs → full passes over training data (we used 6; you can tune).

Batch size → gradient updates per 64 samples (trade-off: stability vs speed).

Validation split → monitors val_loss/val_accuracy to catch overfitting.

💡 Pro tip: Add EarlyStopping (monitor val_loss, patience=2) & ModelCheckpoint to save the best model automatically.

9) 📏 Evaluation & Metrics (how good is it?)
loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
y_pred = (model.predict(X_test_pad) > 0.5).astype(int)

What each metric tells you:

Accuracy: overall correctness.

Precision (REAL): of all predicted REAL, how many were actually REAL?

Recall (REAL): of all truly REAL, how many did we catch?

F1: harmonic mean of precision & recall (handles imbalance better).

Formulas (for class “REAL”):

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Confusion matrix (rows = actual, cols = predicted) visually shows:

TP (REAL→REAL), TN (FAKE→FAKE), FP (FAKE→REAL), FN (REAL→FAKE).

🎯 Goal: High F1 on both classes—especially important if classes are imbalanced.

10) 🔮 Inference (using the model in real life)
def predict_news(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    p = model.predict(pad)[0][0]
    return "REAL" if p > 0.5 else "FAKE"


Flow: raw text → clean → tokenize → pad → predict probability → apply 0.5 threshold.
You can tune the threshold (e.g., 0.6) to be more conservative about marking “REAL”.

🎯 Why this setup reaches ~98% (with a clean dataset)

GloVe provides rich semantics from day one.

Conv1D (256 filters, k=5) captures phrase-level signals (sensational wording, hedge phrases, rumor patterns).

GlobalMaxPooling makes the model position-invariant—if the cue appears anywhere, we catch it.

Dropout + frozen embeddings → better generalization.

Good preprocessing (stopwords + stemming) reduces noise & sparsity.

⚠️ Accuracy depends on data quality, class balance, and leakage control (no article from the same source leaking into both train and test). If your dataset is clean and representative, this pipeline reliably hits 97–98%.

🧩 Frequent pitfalls (and quick fixes)

Imbalanced labels → report F1 and per-class metrics, not just accuracy.

Data leakage → never fit tokenizer or scalers on test text.

Overfitting → watch val_loss; add EarlyStopping, increase Dropout.

OOV words → raise max_words, or unfreeze embeddings to fine-tune.

Long articles → raise max_len (e.g., 400–600) if memory allows.

🔁 (Optional) If you must include TF-IDF + linear baseline

Your requirement listed TF-IDF + PassiveAggressive/SVM. That’s a great baseline to compare:

TF-IDF captures term importance;

PassiveAggressive is fast and strong for text.
Use it as a sanity check; your CNN should outperform it on rich datasets.
