/// <reference types="vitest" />
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { truthLabel } from './src/attention/labels'

/**
 * ADR 0009: the "Real math · Designed heads" label is compiled in only when the src/lib tests pass.
 * The check runs when the config loads. Vitest itself skips it (the child process would load this config again).
 */
function headsVerified(): boolean {
  if (process.env.VITEST || process.env.PAPERLAB_VERIFYING) return false
  const run = spawnSync('npx', ['vitest', 'run', 'src/lib', '--reporter=dot'], {
    env: { ...process.env, PAPERLAB_VERIFYING: '1' }, stdio: 'ignore',
  })
  return run.status === 0
}

/** House plugin: /house/ is external at build and served from HOUSE_DIR in dev (house 2.0.1 tools). */
async function house(): Promise<Plugin | null> {
  const dir = process.env.HOUSE_DIR
  if (!dir) return null
  const file = resolve(dir, 'house/2.0.1/tools/vite-plugin.mjs')
  if (!existsSync(file)) return null
  const mod = await import(pathToFileURL(file).href)
  return mod.default({ houseDir: dir }) as Plugin
}

/** Writes the truth label into the static shell, so the page says the same thing before and after React mounts. */
function shellLabel(text: string): Plugin {
  return { name: 'shell-label', transformIndexHtml: (html) => html.replace('<!--truth-->', text) }
}

/**
 * House assets are same-origin files that Vite must not bundle. A late transform adds them after Vite has read the page,
 * because Vite would otherwise turn them into imports. Pinned to house 2.0.1 by house.lock.json.
 */
function houseAssets(): Plugin {
  const tags = [
    '<link rel="stylesheet" href="/house/2.0.1/tokens.css" />',
    '<link rel="stylesheet" href="/house/2.0.1/house.css" />',
    '<script type="module" src="/house/2.0.1/house.js"></script>',
  ].join('\n    ')
  return { name: 'house-assets', transformIndexHtml: { order: 'post', handler: (html: string) => html.replace('<!--house-assets-->', tags) } }
}

export default defineConfig(async () => {
  const verified = headsVerified()
  return {
  base: '/paperlab/',
  plugins: [react(), shellLabel(truthLabel(verified, 'designed').text), houseAssets(), await house()].filter(Boolean) as Plugin[],
  define: { __HEADS_VERIFIED__: JSON.stringify(verified) },
  build: {
    rollupOptions: { input: { main: resolve(__dirname, 'index.html'), legacy: resolve(__dirname, 'legacy/index.html') } },
  },
  test: { environment: 'node' },
  }
})
