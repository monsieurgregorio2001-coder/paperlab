/** A formula for the maths drawer: plain text with `{name}` placeholders for live values. */
export interface Formula {
  id: string
  label: string
  template: string
}

/** Replaces `{name}` placeholders with values; unknown names are left as they are. */
export function fill(template: string, values: Readonly<Record<string, string | number>>): string {
  return template.replace(/\{(\w+)\}/g, (m, k: string) => (k in values ? String(values[k]) : m))
}
