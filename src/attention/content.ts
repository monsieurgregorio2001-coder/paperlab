import { D_K, D_MODEL, N_HEADS } from '../lib/transformer'

export const PRESETS = [
  'The cat sat on the mat.',
  'Attention is all you need',
  'Time flies like an arrow, fruit flies like a banana.',
  'The dog chased the cat because it was fast.',
]

export const PAPER_URL = 'https://arxiv.org/abs/1706.03762'
/** Each link opens the page that holds the section it names. */
export const PAPER_SECTIONS = {
  dotProduct: { label: 'Section 3.2.1, Scaled Dot-Product Attention', url: 'https://arxiv.org/pdf/1706.03762#page=4' },
  multiHead: { label: 'Section 3.2.2, Multi-Head Attention', url: 'https://arxiv.org/pdf/1706.03762#page=5' },
  embeddings: { label: 'Section 3.4, Embeddings and Softmax', url: 'https://arxiv.org/pdf/1706.03762#page=5' },
  position: { label: 'Section 3.5, Positional Encoding', url: 'https://arxiv.org/pdf/1706.03762#page=6' },
} as const

/** The paper's sizes, from section 3.2.2 and table 3. */
export const PAPER = { dModel: 512, heads: 8, dK: 512 / 8 }
export const HERE = { dModel: D_MODEL, heads: N_HEADS, dK: D_K }

export const REPO_URL = 'https://github.com/monsieurgregorio2001-coder/paperlab'
export const LEGACY_URL = 'legacy/'
