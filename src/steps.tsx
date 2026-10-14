import type { ReactNode } from 'react'

export interface StepInfo {
  id: string
  short: string
  title: string
  section: string
  url: string
  formula: ReactNode
  formulaLabel: string
  explain: string
}

const sub = (s: string) => <sub>{s}</sub>
const sup = (s: string) => <sup>{s}</sup>

export const PAPER_URL = 'https://arxiv.org/abs/1706.03762'
export const HUB_URL = 'https://monsieurgregorio2001-coder.github.io/'
export const REPO_URL = 'https://github.com/monsieurgregorio2001-coder/paperlab'

/** Case-study copy. Measured numbers are filled in from the build and Lighthouse runs. */
export const CASE = {
  problem: 'Papers explain attention in equations that few people can picture.',
  build: 'A React and TypeScript app that runs real attention math on toy weights in the browser, with per-head heatmaps and step controls.',
  result: '19 unit tests on the math, about 58 KB gzipped, and Lighthouse mobile scores of 96 for performance and 100 in the other three.',
}

export const STEPS: StepInfo[] = [
  {
    id: 'tokenize',
    short: 'Tokens',
    title: 'Tokenization',
    section: 'Section 3.4, Embeddings and Softmax',
    url: 'https://arxiv.org/pdf/1706.03762#page=5',
    formulaLabel: 'token id equals hash of the token modulo vocabulary size',
    formula: <>id(token) = hash(lowercase(token)) mod 1000</>,
    explain:
      'The model cannot read letters, so the sentence is cut into pieces called tokens. Each token gets an integer id that stands in for it from here on.',
  },
  {
    id: 'embed',
    short: 'Embeddings',
    title: 'Embeddings and positional encoding',
    section: 'Sections 3.4 and 3.5',
    url: 'https://arxiv.org/pdf/1706.03762#page=6',
    formulaLabel:
      'PE at position pos and dimension 2i is sine of pos over 10000 to the 2i over d model. At 2i plus 1 it is cosine of the same angle.',
    formula: (
      <>
        PE{sub('(pos, 2i)')} = sin(pos / 10000{sup('2i / d_model')})<span className="fsep" />
        PE{sub('(pos, 2i+1)')} = cos(pos / 10000{sup('2i / d_model')})<span className="fsep" />
        X = E + PE
      </>
    ),
    explain:
      'Each token id becomes a list of 16 numbers called an embedding. Attention has no sense of order, so a position wave is added to make the first word differ from the fifth.',
  },
  {
    id: 'attention',
    short: 'Attention',
    title: 'Scaled dot-product attention',
    section: 'Sections 3.2.1 and 3.2.2',
    url: 'https://arxiv.org/pdf/1706.03762#page=4',
    formulaLabel: 'Attention of Q, K, V equals softmax of Q K transpose divided by square root of d k, times V',
    formula: (
      <>
        Attention(Q, K, V) = softmax(QK{sup('T')} / &radic;d{sub('k')}) V<span className="fsep" />where Q = XW{sub('Q')}, K = XW{sub('K')}, V = XW{sub('V')}
      </>
    ),
    explain:
      'Each token asks a question, called its query, and compares it with the key of every other token. The best matches get the largest weights, and the token blends the values it pays attention to.',
  },
  {
    id: 'ffn',
    short: 'Feed-forward',
    title: 'Feed-forward, residual, and layer norm',
    section: 'Sections 3.1 and 3.3',
    url: 'https://arxiv.org/pdf/1706.03762#page=5',
    formulaLabel: 'FFN of x equals max of zero and x W1 plus b1, times W2 plus b2. Output is LayerNorm of x plus FFN of x.',
    formula: (
      <>
        FFN(x) = max(0, xW{sub('1')} + b{sub('1')})W{sub('2')} + b{sub('2')}<span className="fsep" />
        out = LayerNorm(x + FFN(x))
      </>
    ),
    explain:
      'Each token vector passes alone through a small network that grows to 32 numbers and shrinks back to 16. The result is added to the input and normalized so the numbers stay well behaved.',
  },
]

export const PRESETS = [
  'The cat sat on the mat.',
  'Attention is all you need',
  'The animal did not cross the street because it was too tired.',
  'Time flies like an arrow, fruit flies like a banana.',
]
