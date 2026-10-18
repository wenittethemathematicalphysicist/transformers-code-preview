export type Example = { id: string; label: string; code: string };

export const EXAMPLES: Example[] = [
  {
    id: "sentiment",
    label: "Sentiment analysis",
    code: `// pipeline() and log() are available globally
const classifier = await pipeline(
  "sentiment-analysis",
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
);

const result = await classifier([
  "I love building with transformers!",
  "This bug is driving me crazy.",
]);

log(result);`,
  },
  {
    id: "zeroshot",
    label: "Zero-shot classification",
    code: `const classifier = await pipeline(
  "zero-shot-classification",
  "Xenova/mobilebert-uncased-mnli"
);

const out = await classifier(
  "The new GPU doubles training throughput.",
  ["technology", "sports", "politics", "cooking"]
);

log(out);`,
  },
  {
    id: "fillmask",
    label: "Fill mask",
    code: `const unmasker = await pipeline("fill-mask", "Xenova/bert-base-uncased");
log(await unmasker("Paris is the [MASK] of France."));`,
  },
  {
    id: "generate",
    label: "Text generation",
    code: `const generator = await pipeline("text-generation", "Xenova/distilgpt2");
const out = await generator("Once upon a time,", { max_new_tokens: 30 });
log(out[0].generated_text);`,
  },
  {
    id: "embed",
    label: "Embeddings",
    code: `const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);
const emb = await extractor("Hello transformers", { pooling: "mean", normalize: true });
log("dims:", emb.dims);
log("first 8 values:", Array.from(emb.data).slice(0, 8));`,
  },
  {
    id: "scratch",
    label: "Transformer from scratch",
    code: `// A mini Transformer in pure JS, following
// "Attention Is All You Need" (Vaswani, Shazeer et al., 2017)
// No library, no download — just the paper's math.

// ---------- tiny matrix helpers ----------
const matmul = (A, B) =>
  A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
const transpose = (M) => M[0].map((_, j) => M.map((row) => row[j]));
const softmax = (row) => {
  const m = Math.max(...row);
  const e = row.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
};
const add = (A, B) => A.map((r, i) => r.map((x, j) => x + B[i][j]));

// deterministic pseudo-random weights (seeded, so runs are reproducible)
let seed = 42;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
const matrix = (r, c) => Array.from({ length: r }, () => Array.from({ length: c }, rand));

// ---------- 3.5 Positional encoding ----------
const positionalEncoding = (seqLen, dModel) =>
  Array.from({ length: seqLen }, (_, pos) =>
    Array.from({ length: dModel }, (_, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );

// ---------- 3.2.1 Scaled dot-product attention ----------
// Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
function scaledDotProductAttention(Q, K, V) {
  const dk = Q[0].length;
  const scores = matmul(Q, transpose(K)).map((row) => row.map((x) => x / Math.sqrt(dk)));
  const weights = scores.map(softmax);
  return { output: matmul(weights, V), weights };
}

// ---------- 3.2.2 Multi-head attention ----------
function multiHeadAttention(X, dModel, numHeads) {
  const dk = dModel / numHeads;
  const Wq = matrix(dModel, dModel), Wk = matrix(dModel, dModel), Wv = matrix(dModel, dModel);
  const Q = matmul(X, Wq), K = matmul(X, Wk), V = matmul(X, Wv);
  const heads = [];
  for (let h = 0; h < numHeads; h++) {
    const slice = (M) => M.map((row) => row.slice(h * dk, (h + 1) * dk));
    heads.push(scaledDotProductAttention(slice(Q), slice(K), slice(V)));
  }
  const concat = X.map((_, i) => heads.flatMap((h) => h.output[i]));
  return { output: matmul(concat, matrix(dModel, dModel)), weights: heads[0].weights };
}

// ---------- 3.3 Position-wise feed-forward network ----------
const relu = (M) => M.map((row) => row.map((x) => Math.max(0, x)));
function feedForward(X, dModel, dff) {
  return matmul(relu(matmul(X, matrix(dModel, dff))), matrix(dff, dModel));
}

// ---------- 3.1 One encoder layer (attention + FFN + residual) ----------
function encoderLayer(X, dModel, numHeads, dff) {
  const attn = multiHeadAttention(X, dModel, numHeads);
  const afterAttn = add(X, attn.output);          // residual connection
  const afterFfn = add(afterAttn, feedForward(afterAttn, dModel, dff));
  return { output: afterFfn, weights: attn.weights };
}

// ---------- run it on a tiny sentence ----------
const tokens = ["the", "cat", "sat", "down"];
const dModel = 8, numHeads = 2, dff = 16;

const embeddings = matrix(tokens.length, dModel);           // toy learned embeddings
const X = add(embeddings, positionalEncoding(tokens.length, dModel));

const { output, weights } = encoderLayer(X, dModel, numHeads, dff);

log("Input tokens:", tokens.join(" "));
log("d_model =", dModel, "| heads =", numHeads, "| d_ff =", dff);
log("");
log("Attention weights (head 1) — how much each token attends to each other:");
const fmtRow = (row) => row.map((x) => x.toFixed(2)).join("  ");
log("       " + tokens.map((t) => t.padEnd(5)).join(""));
weights.forEach((row, i) => log(tokens[i].padEnd(6) + fmtRow(row)));
log("");
log("Encoder output shape:", output.length, "x", output[0].length);
log("First token's output vector:", output[0].map((x) => x.toFixed(3)));`,
  },
];
