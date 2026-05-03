# AI Agent Gateway

A Cloudflare-based AI chat agent with multi-provider gateway support. Switch between Workers AI (free), Cloudflare AI Gateway, OpenAI, Anthropic, and any custom OpenAI-compatible endpoint — all at runtime, no redeploy needed.

## Quick start

```bash
git clone <your-repo>
cd agent-starter
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to see your agent in action.

## Multi-Provider Gateway

### Architecture

```
┌─────────────────────────────────────────┐
│  Chat UI (model switcher in header)     │
│  ┌────────────────────────────┐         │
│  │ ● Workers AI - Kimi K2.6   │ active  │
│  │ ○ CF Gateway - Llama 3.1   │         │
│  │ ○ OpenAI - GPT-5.2         │         │
│  │ ○ Anthropic - Claude Sonnet│         │
│  │ ○ Custom - Groq            │         │
│  └────────────────────────────┘         │
└──────────────┬──────────────────────────┘
               │ callable methods
               ▼
┌─────────────────────────────────────────┐
│  ChatAgent (Durable Object)             │
│  ├─ resolveProvider()                   │
│  └─ AdminStore DO (SQLite + crypto)    │
│     └─ provider_configs table           │
│        └─ AES-256-GCM encrypted keys   │
└─────────────────────────────────────────┘
```

### Provider Types

| Type | Use Case | API Key | Gateway URL |
|------|----------|---------|-------------|
| `workers-ai` | Free tier, no API key needed | No | No |
| `cf-gateway` | Route through CF AI Gateway | Optional | `gateway.ai.cloudflare.com/v1/{account}/{gateway}` |
| `openai` | OpenAI models (GPT, etc.) | Yes | Optional custom base URL |
| `anthropic` | Claude models | Yes | Optional custom base URL |
| `custom` | Any OpenAI-compatible endpoint (Groq, Together, Ollama, etc.) | Optional | Required |

### Security

API keys are encrypted with **AES-256-GCM** using a per-entry random IV. The master encryption key is stored via `wrangler secret put` — it never touches the database, logs, or client responses.

### Setup Secrets

```bash
# Required for API key encryption (generate a random 32-byte hex key)
wrangler secret put PROVIDER_ENCRYPTION_KEY

# Required for admin panel access
wrangler secret put ADMIN_USR
wrangler secret put ADMIN_PASSWD

# Optional: Cloudflare Account ID for CF Gateway URL construction
wrangler secret put CF_ACCOUNT_ID
```

Generate an encryption key:
```bash
openssl rand -hex 32
```

### Admin Panel

Visit `/admin` on your deployed worker to manage providers:
- Add/edit/remove providers
- Set default provider
- Configure models, API keys, gateway names, and custom base URLs
- API keys are masked — never sent to the client

### Switching Models

Use the model dropdown in the chat header to switch providers at runtime. The change takes effect on the next message.

## Project structure

```
src/
  server.ts       # ChatAgent + AdminStore DOs, admin SPA HTML
  app.tsx         # Chat UI with model switcher
  client.tsx      # React entry point
  styles.css      # Tailwind + Kumo styles
  lib/
    crypto.ts     # AES-256-GCM encryption utilities
```

## What's included

- **Multi-Provider Gateway** — Workers AI, CF Gateway, OpenAI, Anthropic, Custom endpoints
- **Admin Panel** (`/admin`) — Full CRUD for provider management, basic auth protected
- **API Key Encryption** — AES-256-GCM with per-entry random IV, master key in Workers Secrets
- **AI Chat** — Streaming responses via `AIChatAgent`
- **Image input** — Drag-and-drop, paste, or click to attach images
- **Tool patterns** — server-side auto-execute, client-side (browser), approval tools
- **Scheduling** — one-time, delayed, and recurring (cron) tasks
- **Reasoning display** — shows model thinking as it streams
- **MCP support** — connect external MCP servers
- **Kumo UI** — Cloudflare's design system with dark/light mode
- **Real-time** — WebSocket with automatic reconnection and message persistence
- **Logs & Traces** — Cloudflare observability enabled

## Making it your own

### Change the system prompt

Edit the `system` string in `server.ts` → `onChatMessage()`.

### Add your own tools

Add new tools to the `tools` object in `onChatMessage()` in `server.ts`.

### Customize scheduled tasks

Replace `executeTask()` in `server.ts` with your own logic.

## Deploy

```bash
# Set secrets first
wrangler secret put PROVIDER_ENCRYPTION_KEY "<32-byte-hex>"
wrangler secret put ADMIN_USR "admin"
wrangler secret put ADMIN_PASSWD "<your-password>"

# Deploy
npm run deploy
```

## License

MIT
