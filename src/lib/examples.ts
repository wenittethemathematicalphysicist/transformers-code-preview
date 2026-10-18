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
];
