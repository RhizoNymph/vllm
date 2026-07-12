# Port Notes — Neuronpedia jlens UI → standalone Vite app

Source: https://github.com/hijohnnylin/neuronpedia (`apps/webapp`, MIT — see
`LICENSE-THIRD-PARTY.md`). Target: Vite 6 + React 19 + TypeScript + Tailwind 3,
single model (`glm-5.2`), backend at same-origin `/api/lens/prompt`.

## Files copied VERBATIM (byte-identical to upstream)

| Here | Upstream (`apps/webapp/`) |
| --- | --- |
| `src/components/jlens/*` (18 files) | `components/jlens/*` |
| `src/lib/utils/lens.ts` | `lib/utils/lens.ts` |
| `src/lib/utils/jlens-share.ts` | `lib/utils/jlens-share.ts` |
| `src/lib/utils/chinese-translations.ts` | `lib/utils/chinese-translations.ts` |
| `src/lib/utils/ui.ts` | `lib/utils/ui.ts` (the `cn` helper) |
| `src/components/custom-tooltip.tsx` | `components/custom-tooltip.tsx` |
| `src/components/shadcn/dialog.tsx` | `components/shadcn/dialog.tsx` |
| `src/components/shadcn/tooltip.tsx` | `components/shadcn/tooltip.tsx` |
| `src/components/svg/loading-square.tsx` | `components/svg/loading-square.tsx` |
| `src/app/[modelId]/jlens/jlens-tour-constants.ts` | `app/[modelId]/jlens/jlens-tour-constants.ts` |

The `@/` path alias maps to `src/` (vite.config.ts + tsconfig.json), and the
literal `src/app/[modelId]/jlens/` directory exists so the components'
`@/app/[modelId]/jlens/...` imports resolve unchanged.

`tailwind.config.cjs`'s `theme.extend` block and plugin list are copied
verbatim from upstream `tailwind.config.js`; only the `content` globs differ.

## Functional edits to copied files

**None.** In particular, `jlens-stream.ts` already POSTs same-origin
`/api/lens/prompt`, so no URL rewrite was needed.

## Stubs (new files, not upstream code)

- `src/components/provider/global-provider.tsx` — replaces Neuronpedia's
  DB/auth-backed GlobalProvider. Exports `useGlobalContext()` with the exact
  subset the copied components consume: `globalModels` (static map with one
  entry, `glm-5.2`: `instruct: true`, `layers: 78`, `inferenceEnabled: true`),
  `user: undefined`, `showToastMessage(ReactNode)` (renders a simple fixed
  toast div, auto-dismisses after 3.5 s), `getInferenceEnabledModels()`,
  `getInferenceEnabledForModel()`. Also exports the `GlobalProvider` wrapper
  used by `App.tsx`.
- `src/app/[modelId]/jlens/jlens-tour-context.tsx` — same exports as upstream
  (`JlensTourStepContext`, `useJlensTourStep`) but with a local structural
  `DriveStep` type instead of importing it from `driver.js` (the guided tour
  is not ported, so `driver.js` is not a dependency). The context value is
  always `null` = "no tour running", which is the normal non-tour code path in
  every consumer.
- `src/App.tsx` — replaces `app/[modelId]/jlens/jlens-page-client.tsx`.
  Replicates its provider nesting around `JlensPanel`
  (`JlensTourStepContext.Provider` → `LensModeContext.Provider` →
  `LensModeSetContext.Provider`) and its full-height flex page layout, but
  drops the Next router, share loading (`?shareId=`), model selector, tour,
  intro-video modal, and external link buttons. Header is reduced to the
  "Jacobian Lens / Gurnee et al." title plus a static GLM-5.2 badge.
- `src/main.tsx`, `src/index.css`, `index.html`, `vite.config.ts`,
  `tsconfig.json`, `postcss.config.js` — standard Vite scaffolding.
- `public/chinese-translations-qwen.json` — empty `{}`. The verbatim
  `chinese-translations.ts` helper fetches this path; an empty object means
  "no translations" and is handled gracefully.

## Deliberately dead / degraded paths

- **Share dialog** (`jlens-share-dialog.tsx`, verbatim): POSTs
  `/api/lens/share`. If the backend doesn't implement it, the dialog shows the
  error inline — no crash. The JSON export/import path (`jlens-export.ts` +
  the fixture bar) works fully client-side.
- **Tour**: with the stub context always `null`, the tour-only branches in
  `jlens-chat.tsx` / `jlens-steer-panel.tsx` / `jlens-analysis.tsx` never
  trigger. That includes `jlens-chat.tsx`'s fetch of `/qwen-output.json`
  (a precached tour fixture), which is only reachable on the steer-panel tour
  step — no fixture file is needed.
- **Fonts**: upstream uses Next-injected `--font-inter` / `--font-sf`
  variables; `index.css` maps both to a system-UI stack.
- **Dialog open/close animations**: the shadcn dialog uses
  `animate-in`/`fade-in-0`-style classes from the `tailwindcss-animate`
  plugin, which upstream's Tailwind config does *not* load (it loads
  `tailwindcss-animated`). Those classes are no-ops here exactly as they are
  upstream.

## Build / checks

- `npm install && npm run build` → `dist/` (`npm run build` runs
  `tsc --noEmit` first; both clean).
- Headless-Chromium smoke test: the built app mounts and renders the chat
  interface (message + prefill textareas, jacobian-space sidebar) without
  runtime errors.

## Serving + backend contract

Serve `dist/` as static files from the **same origin** as the lens API (the
app uses relative URLs). E.g. with the sidecar mounting `dist/` at `/`, or:

```
npx vite preview          # static preview on :4173 (no API)
npm run dev               # dev server; proxies /api → $JLENS_API_TARGET
                          # (default http://localhost:8000)
```

The app expects:

- `POST /api/lens/prompt` — JSON body per `RunLensStreamParams` in
  `src/components/jlens/jlens-stream.ts` (camelCase: `modelId`, `prompt` or
  `chat`, `type`, `topN`, `temperature`, `numCompletionTokens`,
  `prependBos?`, `enableThinking?`, `cachedTokenIds?`, `inputTokenIds?`,
  `filterNonWordTokens?`, `steerTokens?`, `steerLayers?`, `steerStrength?`,
  `steerAblate?`, `swapToken?`, `steerGeneratedTokens?`). `modelId` will be
  `"glm-5.2"`. Response: NDJSON stream, one JSON message per line, with
  `kind` ∈ `meta` | `prompt` | `token` | `done` | `error` (types in
  `src/lib/utils/lens.ts`). Optional `x-limit-remaining` response header
  drives the rate-limit counter UI; a 429 with `{"limitPerWindow": n}` shows
  the friendly hourly-limit message.
- `POST /api/lens/share` — optional (see above); success response
  `{"path": "/..."}`.


## Post-port functional edits (2026-07-11)

- `jlens-chat-format.ts`: added a `glm` `JlensChatFormat` (custom
  `groupTokens`). GLM-4/5 turns are delimited by per-role marker tokens
  (`<|system|>` / `<|user|>` / `<|assistant|>` / `<|observation|>`) with no
  turn-end token; the `[gMASK]<sop>` prefix folds into the first bubble's
  header. Without this, glm model ids fell through to the ChatML default,
  whose markers never occur in GLM streams — token grouping (and per-token
  inspection of the pre-assistant boundary position) didn't work.
- `vite.config.ts`: `base: '/np/'` (app is served under the sidecar's /np
  mount; root-absolute assets 404'd there).
