import { DurableObject } from "cloudflare:workers";
import { createWorkersAI } from "workers-ai-provider";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool,
  type ModelMessage,
  type LanguageModel
} from "ai";
import { z } from "zod";
import { decryptApiKey } from "./lib/crypto";

// ── Provider types ────────────────────────────────────────────────────

export type ProviderType =
  | "workers-ai"
  | "cf-gateway"
  | "openai"
  | "anthropic"
  | "custom";

export interface ProviderConfig {
  id: string;
  name: string;
  type: ProviderType;
  model: string;
  gateway_name?: string;
  base_url?: string;
  has_api_key: boolean;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

interface ProviderRow {
  id: string;
  name: string;
  type: ProviderType;
  model: string;
  gateway_name: string | null;
  base_url: string | null;
  encrypted_api_key: string | null;
  api_key_iv: string | null;
  is_default: number;
  created_at: string;
  updated_at: string;
}

function rowToConfig(row: ProviderRow): ProviderConfig {
  return {
    id: row.id,
    name: row.name,
    type: row.type,
    model: row.model,
    gateway_name: row.gateway_name || undefined,
    base_url: row.base_url || undefined,
    has_api_key: !!row.encrypted_api_key,
    is_default: !!row.is_default,
    created_at: row.created_at,
    updated_at: row.updated_at
  };
}

// ── Inline data URL helper ────────────────────────────────────────────

function inlineDataUrls(messages: ModelMessage[]): ModelMessage[] {
  return messages.map((msg) => {
    if (msg.role !== "user" || typeof msg.content === "string") return msg;
    return {
      ...msg,
      content: msg.content.map((part) => {
        if (part.type !== "file" || typeof part.data !== "string") return part;
        const match = part.data.match(/^data:([^;]+);base64,(.+)$/);
        if (!match) return part;
        const bytes = Uint8Array.from(atob(match[2]), (c) => c.charCodeAt(0));
        return { ...part, data: bytes, mediaType: match[1] };
      })
    };
  });
}

// ── ChatAgent ─────────────────────────────────────────────────────────

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;

  onStart() {
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  // ── Callable methods for UI ───────────────────────────────────────

  @callable()
  async listProviders(): Promise<{
    providers: ProviderConfig[];
    encryption_ok: boolean;
  }> {
    const ns = this.env.AdminStore;
    const id = ns.idFromName("admin-store");
    const stub = ns.get(id) as unknown as AdminStoreRpc;
    return stub.listProviders(this.env.PROVIDER_ENCRYPTION_KEY);
  }

  @callable()
  async setDefaultProvider(id: string): Promise<{ ok: boolean }> {
    const ns = this.env.AdminStore;
    const storeId = ns.idFromName("admin-store");
    const stub = ns.get(storeId) as unknown as AdminStoreRpc;
    const result = await stub.setDefaultProvider(id);
    this.broadcast(JSON.stringify({ type: "provider-changed" }));
    return result;
  }

  // ── Provider resolution ───────────────────────────────────────────

  private async resolveProvider(): Promise<{
    model: LanguageModel;
    config: ProviderConfig;
  }> {
    const ns = this.env.AdminStore;
    const storeId = ns.idFromName("admin-store");
    const stub = ns.get(storeId) as unknown as AdminStoreRpc;
    const active = await stub.getActiveProvider(
      this.env.PROVIDER_ENCRYPTION_KEY
    );

    if (!active) {
      throw new Error("No default provider configured");
    }

    const { config, decryptedApiKey } = active;
    let model: LanguageModel;

    switch (config.type) {
      case "workers-ai": {
        const workersai = createWorkersAI({ binding: this.env.AI });
        model = workersai(config.model);
        break;
      }

      case "cf-gateway": {
        if (!config.gateway_name) {
          throw new Error("CF Gateway provider missing gateway_name");
        }
        const accountId = this.env.CF_ACCOUNT_ID || "account";
        const gwUrl = `https://gateway.ai.cloudflare.com/v1/${accountId}/${config.gateway_name}`;
        const cfGw = createOpenAI({
          baseURL: gwUrl,
          apiKey: decryptedApiKey || "workers-ai"
        });
        model = cfGw(config.model);
        break;
      }

      case "openai": {
        if (!decryptedApiKey) {
          throw new Error("OpenAI provider missing API key");
        }
        const openai = createOpenAI({
          apiKey: decryptedApiKey,
          baseURL: config.base_url || undefined
        });
        model = openai.chat(config.model);
        break;
      }

      case "anthropic": {
        if (!decryptedApiKey) {
          throw new Error("Anthropic provider missing API key");
        }
        const anthropic = createAnthropic({
          apiKey: decryptedApiKey,
          baseURL: config.base_url || undefined
        });
        model = anthropic.chat(config.model);
        break;
      }

      case "custom": {
        if (!config.base_url) {
          throw new Error("Custom provider missing base_url");
        }
        const custom = createOpenAICompatible({
          name: "custom",
          baseURL: config.base_url,
          apiKey: decryptedApiKey || "custom"
        });
        model = custom.chatModel(config.model);
        break;
      }

      default:
        throw new Error(`Unknown provider type: ${config.type}`);
    }

    return { model, config };
  }

  // ── Chat message handler ──────────────────────────────────────────

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const { model, config } = await this.resolveProvider();

    const systemPrompt = `You are a helpful assistant that can understand images.

Current provider: ${config.name} (${config.type})
Model: ${config.model}

${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task, use the schedule tool.`;

    const result = streamText({
      model,
      system: systemPrompt,
      messages: pruneMessages({
        messages: inlineDataUrls(await convertToModelMessages(this.messages)),
        toolCalls: "before-last-2-messages"
      }),
      tools: {
        ...mcpTools,

        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name")
          }),
          execute: async ({ city }) => {
            const conditions = ["sunny", "cloudy", "rainy", "snowy"];
            const temp = Math.floor(Math.random() * 30) + 5;
            return {
              city,
              temperature: temp,
              condition:
                conditions[Math.floor(Math.random() * conditions.length)],
              unit: "celsius"
            };
          }
        }),

        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser.",
          inputSchema: z.object({})
        }),

        calculate: tool({
          description: "Perform a math calculation with two numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description: "Schedule a task to be executed at a later time.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") return "Not a valid schedule input";
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(5),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    console.log(`Executing scheduled task: ${description}`);
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

// ── RPC type for AdminStore ───────────────────────────────────────────

interface AdminStoreRpc {
  listProviders(masterKey: string): Promise<{
    providers: ProviderConfig[];
    encryption_ok: boolean;
  }>;
  getActiveProvider(masterKey: string): Promise<{
    config: ProviderConfig;
    decryptedApiKey: string | null;
  } | null>;
  setDefaultProvider(id: string): Promise<{ ok: boolean }>;
  handleAdminApi(request: Request, masterKey: string): Promise<Response>;
}

// ── AdminStore Durable Object ─────────────────────────────────────────

export class AdminStore extends DurableObject {
  constructor(
    state: DurableObjectState,
    env: Env
  ) {
    super(state, env);
  }

  // ── DB helpers ────────────────────────────────────────────────────

  private ensureTable() {
    this.ctx.storage.sql.exec(`
      CREATE TABLE IF NOT EXISTS provider_configs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        model TEXT NOT NULL,
        gateway_name TEXT,
        base_url TEXT,
        encrypted_api_key TEXT,
        api_key_iv TEXT,
        is_default INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
      )
    `);
    const count = this.ctx.storage.sql
      .exec("SELECT COUNT(*) as c FROM provider_configs")
      .toArray()[0] as unknown as { c: number };
    if (count.c === 0) {
      this.ctx.storage.sql.exec(
        `INSERT INTO provider_configs (id, name, type, model, is_default)
         VALUES (?, ?, ?, ?, 1)`,
        "prov_default_workers_ai",
        "Workers AI (Free)",
        "workers-ai",
        "@cf/moonshotai/kimi-k2.6"
      );
    }
  }

  private generateId(): string {
    return (
      "prov_" +
      Date.now().toString(36) +
      "_" +
      Math.random().toString(36).slice(2, 8)
    );
  }

  // ── RPC methods called by ChatAgent ──────────────────────────────

  async listProviders(masterKey: string): Promise<{
    providers: ProviderConfig[];
    encryption_ok: boolean;
  }> {
    this.ensureTable();
    const rows = this.ctx.storage.sql
      .exec(
        "SELECT * FROM provider_configs ORDER BY is_default DESC, created_at DESC"
      )
      .toArray() as unknown as ProviderRow[];
    return {
      encryption_ok: !!masterKey,
      providers: rows.map(rowToConfig)
    };
  }

  async getActiveProvider(masterKey: string): Promise<{
    config: ProviderConfig;
    decryptedApiKey: string | null;
  } | null> {
    this.ensureTable();
    const row = this.ctx.storage.sql
      .exec("SELECT * FROM provider_configs WHERE is_default = 1 LIMIT 1")
      .toArray()[0] as unknown as ProviderRow | undefined;
    if (!row) return null;

    let decryptedApiKey: string | null = null;
    if (row.encrypted_api_key && row.api_key_iv && masterKey) {
      decryptedApiKey = await decryptApiKey(
        row.encrypted_api_key,
        row.api_key_iv,
        masterKey
      );
    }

    return { config: rowToConfig(row), decryptedApiKey };
  }

  async setDefaultProvider(id: string): Promise<{ ok: boolean }> {
    this.ensureTable();
    const existing = this.ctx.storage.sql
      .exec("SELECT * FROM provider_configs WHERE id = ?", id)
      .toArray()[0] as unknown as ProviderRow | undefined;
    if (!existing) throw new Error("Provider not found");
    this.ctx.storage.sql.exec("UPDATE provider_configs SET is_default = 0");
    this.ctx.storage.sql.exec(
      "UPDATE provider_configs SET is_default = 1 WHERE id = ?",
      id
    );
    return { ok: true };
  }

  // ── Admin API (called from worker fetch handler) ─────────────────

  async handleAdminApi(request: Request, masterKey: string): Promise<Response> {
    this.ensureTable();
    const url = new URL(request.url);
    const path = url.pathname.replace("/admin/api", "");

    if (path === "/providers" && request.method === "GET") {
      return this.listProviders(masterKey).then((r) => Response.json(r));
    }

    if (path === "/providers" && request.method === "POST") {
      return this.createProvider(request, masterKey);
    }

    if (path.match(/^\/providers\/[^/]+$/) && request.method === "PUT") {
      const id = path.split("/")[2];
      return this.updateProvider(id, request, masterKey);
    }

    if (path.match(/^\/providers\/[^/]+$/) && request.method === "DELETE") {
      const id = path.split("/")[2];
      return this.deleteProvider(id);
    }

    if (
      path.match(/^\/providers\/[^/]+\/set-default$/) &&
      request.method === "POST"
    ) {
      const id = path.split("/")[2];
      return this.setDefaultProvider(id).then((r) => Response.json(r));
    }

    return Response.json({ error: "Not found" }, { status: 404 });
  }

  private async createProvider(
    request: Request,
    masterKey: string
  ): Promise<Response> {
    const { encryptApiKey } = await import("./lib/crypto");
    const body = (await request.json()) as {
      name: string;
      type: ProviderType;
      model: string;
      gateway_name?: string;
      base_url?: string;
      api_key?: string;
    };
    const id = this.generateId();
    const now = new Date().toISOString();
    let encryptedKey: string | null = null;
    let keyIv: string | null = null;

    if (body.api_key && body.type !== "workers-ai") {
      if (!masterKey) {
        return Response.json(
          {
            error:
              "PROVIDER_ENCRYPTION_KEY secret not configured. Run: wrangler secret put PROVIDER_ENCRYPTION_KEY"
          },
          { status: 400 }
        );
      }
      const enc = await encryptApiKey(body.api_key, masterKey);
      encryptedKey = enc.ciphertext;
      keyIv = enc.iv;
    }

    const count = this.ctx.storage.sql
      .exec("SELECT COUNT(*) as c FROM provider_configs")
      .toArray()[0] as unknown as { c: number };
    const isDefault = count.c === 0 ? 1 : 0;

    this.ctx.storage.sql.exec(
      `INSERT INTO provider_configs (id, name, type, model, gateway_name, base_url, encrypted_api_key, api_key_iv, is_default, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      id,
      body.name,
      body.type,
      body.model,
      body.gateway_name || null,
      body.base_url || null,
      encryptedKey,
      keyIv,
      isDefault,
      now,
      now
    );

    const row = this.ctx.storage.sql
      .exec("SELECT * FROM provider_configs WHERE id = ?", id)
      .toArray()[0] as unknown as ProviderRow;
    return Response.json(rowToConfig(row));
  }

  private async updateProvider(
    id: string,
    request: Request,
    masterKey: string
  ): Promise<Response> {
    const { encryptApiKey } = await import("./lib/crypto");
    const existing = this.ctx.storage.sql
      .exec("SELECT * FROM provider_configs WHERE id = ?", id)
      .toArray()[0] as unknown as ProviderRow | undefined;
    if (!existing) {
      return Response.json({ error: "Provider not found" }, { status: 404 });
    }

    const body = (await request.json()) as Partial<{
      name: string;
      type: ProviderType;
      model: string;
      gateway_name?: string;
      base_url?: string;
      api_key?: string;
    }>;
    const now = new Date().toISOString();

    let encryptedKey = existing.encrypted_api_key;
    let keyIv = existing.api_key_iv;

    if (body.api_key && masterKey) {
      const enc = await encryptApiKey(body.api_key, masterKey);
      encryptedKey = enc.ciphertext;
      keyIv = enc.iv;
    }

    this.ctx.storage.sql.exec(
      `UPDATE provider_configs SET name=?, type=?, model=?, gateway_name=?, base_url=?, encrypted_api_key=?, api_key_iv=?, updated_at=? WHERE id=?`,
      body.name ?? existing.name,
      body.type ?? existing.type,
      body.model ?? existing.model,
      body.gateway_name ?? existing.gateway_name,
      body.base_url ?? existing.base_url,
      encryptedKey,
      keyIv,
      now,
      id
    );

    const row = this.ctx.storage.sql
      .exec("SELECT * FROM provider_configs WHERE id = ?", id)
      .toArray()[0] as unknown as ProviderRow;
    return Response.json(rowToConfig(row));
  }

  private deleteProvider(id: string): Response {
    const existing = this.ctx.storage.sql
      .exec("SELECT * FROM provider_configs WHERE id = ?", id)
      .toArray()[0] as unknown as ProviderRow | undefined;
    if (!existing) {
      return Response.json({ error: "Provider not found" }, { status: 404 });
    }
    if (existing.is_default) {
      const count = this.ctx.storage.sql
        .exec("SELECT COUNT(*) as c FROM provider_configs")
        .toArray()[0] as unknown as { c: number };
      if (count.c <= 1) {
        return Response.json(
          { error: "Cannot delete the only provider" },
          { status: 400 }
        );
      }
      this.ctx.storage.sql.exec(
        "UPDATE provider_configs SET is_default = 1 WHERE id != ? LIMIT 1",
        id
      );
    }
    this.ctx.storage.sql.exec("DELETE FROM provider_configs WHERE id = ?", id);
    return Response.json({ ok: true });
  }
}

// ── Admin SPA HTML ────────────────────────────────────────────────────

const ADMIN_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Provider Admin</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0a;color:#e0e0e0;min-height:100vh}
.container{max-width:800px;margin:0 auto;padding:2rem 1.5rem}
h1{font-size:1.5rem;margin-bottom:1.5rem;color:#fff}
h2{font-size:1.1rem;margin-bottom:1rem;color:#ccc}
.badge{font-size:.7rem;padding:2px 8px;border-radius:99px;background:#2a2a2a;color:#aaa}
.badge.default{background:#1a3a1a;color:#4ade80}
.badge.has-key{background:#1a1a3a;color:#818cf8}
.field{margin-bottom:.75rem}
.field label{display:block;font-size:.75rem;color:#888;margin-bottom:.25rem}
.field input,.field select{width:100%;padding:.5rem .75rem;background:#0a0a0a;border:1px solid #2a2a2a;border-radius:8px;color:#e0e0e0;font-size:.85rem;outline:none}
.field input:focus,.field select:focus{border-color:#818cf8}
.btn{padding:.5rem 1rem;border-radius:8px;border:none;cursor:pointer;font-size:.8rem;font-weight:500;transition:opacity .15s}
.btn:hover{opacity:.85}
.btn-primary{background:#818cf8;color:#fff}
.btn-secondary{background:#2a2a2a;color:#ccc}
.btn-danger{background:#3a1a1a;color:#f87171}
.btn-sm{padding:.35rem .7rem;font-size:.75rem}
.row{display:flex;gap:.5rem;align-items:center}
.row-between{display:flex;justify-content:space-between;align-items:center}
.mt-1{margin-top:.5rem}.mt-2{margin-top:1rem}.mt-3{margin-top:1.5rem}
.text-sm{font-size:.8rem;color:#888}
.empty{text-align:center;padding:3rem;color:#555}
.status-bar{display:flex;gap:1rem;align-items:center;padding:.75rem 1rem;background:#161616;border:1px solid #2a2a2a;border-radius:8px;margin-bottom:1.5rem}
.status-dot{width:8px;height:8px;border-radius:50%}
.status-dot.ok{background:#4ade80}
.status-dot.fail{background:#f87171}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}
.form-grid .full{grid-column:1/-1}
.provider-item{display:flex;justify-content:space-between;align-items:flex-start;padding:1rem;background:#111;border:1px solid #222;border-radius:10px;margin-bottom:.75rem}
.provider-item:hover{border-color:#333}
.provider-meta{flex:1;min-width:0}
.provider-name{font-weight:600;color:#fff;margin-bottom:.25rem}
.provider-detail{font-size:.75rem;color:#888}
.provider-actions{display:flex;gap:.5rem;flex-shrink:0;margin-left:1rem}
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:50}
.modal{background:#161616;border:1px solid #2a2a2a;border-radius:16px;padding:1.5rem;width:90%;max-width:500px;max-height:80vh;overflow-y:auto}
.toast{position:fixed;bottom:1.5rem;right:1.5rem;padding:.75rem 1.25rem;background:#1a3a1a;color:#4ade80;border-radius:8px;font-size:.85rem;z-index:100;animation:slideIn .3s}
.toast.error{background:#3a1a1a;color:#f87171}
@keyframes slideIn{from{transform:translateY(1rem);opacity:0}to{transform:translateY(0);opacity:1}}
</style>
</head>
<body>
<div class="container">
  <h1>Provider Management</h1>
  <div id="status-bar" class="status-bar"></div>
  <div class="row-between mt-2">
    <h2>Providers</h2>
    <button class="btn btn-primary" onclick="showAddModal()">+ Add Provider</button>
  </div>
  <div id="provider-list" class="mt-1"></div>
</div>
<div id="modal-root"></div>
<div id="toast-root"></div>
<script>
const API = "/admin/api";
let providers = [];
let encryptionOk = false;
let authHeader = "";

function getAuth() {
  const u = prompt("Admin username:");
  if (!u) return;
  const p = prompt("Admin password:");
  if (!p) return;
  authHeader = "Basic " + btoa(u + ":" + p);
  load();
}

async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json", "Authorization": authHeader } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + path, opts);
  if (res.status === 401) { getAuth(); throw new Error("Auth required"); }
  if (!res.ok) { const err = await res.json().catch(() => ({error:res.statusText})); throw new Error(err.error || res.statusText); }
  return res.json();
}

function toast(msg, isError = false) {
  const el = document.createElement("div");
  el.className = "toast" + (isError ? " error" : "");
  el.textContent = msg;
  document.getElementById("toast-root").appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

async function load() {
  try {
    const data = await api("GET", "/providers");
    providers = data.providers;
    encryptionOk = data.encryption_ok;
    render();
  } catch (e) { if (e.message !== "Auth required") toast("Failed: " + e.message, true); }
}

function render() {
  const sb = document.getElementById("status-bar");
  sb.innerHTML = encryptionOk
    ? '<span class="status-dot ok"></span><span class="text-sm">Encryption key configured</span>'
    : '<span class="status-dot fail"></span><span class="text-sm">Encryption key NOT set — run: wrangler secret put PROVIDER_ENCRYPTION_KEY</span>';
  const list = document.getElementById("provider-list");
  if (providers.length === 0) { list.innerHTML = '<div class="empty">No providers configured.</div>'; return; }
  list.innerHTML = providers.map((p) => {
    const badges = [];
    if (p.is_default) badges.push('<span class="badge default">Default</span>');
    if (p.has_api_key) badges.push('<span class="badge has-key">Key set</span>');
    const detail = p.type === "cf-gateway" ? \`Gateway: \${p.gateway_name} | Model: \${p.model}\` : p.base_url ? \`\${p.base_url} | \${p.model}\` : p.model;
    return \`<div class="provider-item">
      <div class="provider-meta">
        <div class="provider-name">\${p.name}</div>
        <div class="provider-detail">\${p.type} — \${detail}</div>
        <div class="row mt-1" style="gap:.4rem">\${badges.join("")}</div>
      </div>
      <div class="provider-actions">
        \${!p.is_default ? \`<button class="btn btn-sm btn-secondary" onclick="setDefault('\${p.id}')">Set Default</button>\` : ""}
        <button class="btn btn-sm btn-secondary" onclick="showEditModal('\${p.id}')">Edit</button>
        \${providers.length > 1 && !p.is_default ? \`<button class="btn btn-sm btn-danger" onclick="removeProvider('\${p.id}')">Delete</button>\` : ""}
      </div>
    </div>\`;
  }).join("");
}

async function setDefault(id) { try { await api("POST", "/providers/" + id + "/set-default"); await load(); toast("Default updated"); } catch(e) { if(e.message!=="Auth required") toast(e.message,true); } }
async function removeProvider(id) { if (!confirm("Delete this provider?")) return; try { await api("DELETE", "/providers/" + id); await load(); toast("Deleted"); } catch(e) { if(e.message!=="Auth required") toast(e.message,true); } }

function showAddModal() { showModal(null); }
function showEditModal(id) { const p = providers.find(x => x.id === id); if (p) showModal(p); }

function showModal(provider) {
  const isEdit = !!provider;
  document.getElementById("modal-root").innerHTML = \`
    <div class="modal-overlay" onclick="if(event.target===this)closeModal()">
      <div class="modal">
        <div class="row-between" style="margin-bottom:1rem">
          <h2>\${isEdit ? "Edit" : "Add"} Provider</h2>
          <button class="btn btn-sm btn-secondary" onclick="closeModal()">Close</button>
        </div>
        <form id="provider-form" onsubmit="submitForm(event)">
          <div class="form-grid">
            <div class="field full"><label>Provider Name</label><input name="name" value="\${provider?.name||""}" required placeholder="e.g. My OpenAI"/></div>
            <div class="field"><label>Type</label><select name="type" onchange="updateFormFields()">
              <option value="workers-ai" \${provider?.type==="workers-ai"?"selected":""}>Workers AI (Free)</option>
              <option value="cf-gateway" \${provider?.type==="cf-gateway"?"selected":""}>CF Gateway</option>
              <option value="openai" \${provider?.type==="openai"?"selected":""}>OpenAI</option>
              <option value="anthropic" \${provider?.type==="anthropic"?"selected":""}>Anthropic</option>
              <option value="custom" \${provider?.type==="custom"?"selected":""}>Custom</option>
            </select></div>
            <div class="field"><label>Model</label><input name="model" value="\${provider?.model||""}" required placeholder="e.g. gpt-5.2"/></div>
            <div class="field" id="f-gateway" style="display:none"><label>Gateway Name</label><input name="gateway_name" value="\${provider?.gateway_name||""}" placeholder="my-gateway"/></div>
            <div class="field" id="f-baseurl" style="display:none"><label>Base URL (optional)</label><input name="base_url" value="\${provider?.base_url||""}" placeholder="https://..."/></div>
            <div class="field full" id="f-apikey"><label>API Key</label><input name="api_key" type="password" placeholder="\${isEdit?"Leave blank to keep existing":"sk-..."}"/><span class="text-sm mt-1">\${isEdit&&provider?.has_api_key?"Leave blank to keep existing.":"Required for OpenAI/Anthropic. Optional for CF Gateway."}</span></div>
          </div>
          <div class="row mt-3" style="justify-content:flex-end;gap:.5rem">
            <button type="button" class="btn btn-secondary" onclick="closeModal()">Cancel</button>
            <button type="submit" class="btn btn-primary">\${isEdit?"Update":"Add"}</button>
          </div>
        </form>
      </div>
    </div>\`;
  updateFormFields();
  if (isEdit) document.getElementById("provider-form").dataset.editId = provider.id;
}

function updateFormFields() {
  const t = document.querySelector('select[name="type"]').value;
  document.getElementById("f-gateway").style.display = t === "cf-gateway" ? "" : "none";
  const showB = t === "openai" || t === "anthropic" || t === "custom";
  document.getElementById("f-baseurl").style.display = showB ? "" : "none";
  document.getElementById("f-apikey").style.display = t === "workers-ai" ? "none" : "";
}

function closeModal() { document.getElementById("modal-root").innerHTML = ""; }

async function submitForm(e) {
  e.preventDefault();
  const f = e.target;
  const data = { name:f.name.value.trim(), type:f.type.value, model:f.model.value.trim() };
  if (f.gateway_name?.value?.trim()) data.gateway_name = f.gateway_name.value.trim();
  if (f.base_url?.value?.trim()) data.base_url = f.base_url.value.trim();
  if (f.api_key?.value?.trim()) data.api_key = f.api_key.value.trim();
  try {
    const editId = f.dataset.editId;
    if (editId) { await api("PUT", "/providers/" + editId, data); toast("Updated"); }
    else { await api("POST", "/providers", data); toast("Added"); }
    closeModal(); await load();
  } catch(e) { if(e.message!=="Auth required") toast(e.message,true); }
}

getAuth();
</script>
</body>
</html>`;

// ── Worker fetch handler ──────────────────────────────────────────────

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Serve admin SPA
    if (url.pathname === "/admin" || url.pathname === "/admin/") {
      return new Response(ADMIN_HTML, {
        headers: { "content-type": "text/html; charset=utf-8" }
      });
    }

    // Route admin API to AdminStore DO
    if (url.pathname.startsWith("/admin/api")) {
      // Basic auth check
      const auth = request.headers.get("Authorization");
      if (!auth || !auth.startsWith("Basic ")) {
        return Response.json(
          { error: "Unauthorized" },
          {
            status: 401,
            headers: {
              "Access-Control-Allow-Origin": "*",
              "Access-Control-Allow-Methods":
                "GET,POST,PUT,DELETE,OPTIONS",
              "Access-Control-Allow-Headers":
                "Content-Type,Authorization"
            }
          }
        );
      }
      const decoded = atob(auth.slice(6));
      const [user, pass] = decoded.split(":");
      if (user !== env.ADMIN_USR || pass !== env.ADMIN_PASSWD) {
        return Response.json(
          { error: "Unauthorized" },
          { status: 401 }
        );
      }

      // CORS preflight
      if (request.method === "OPTIONS") {
        return new Response(null, {
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods":
              "GET,POST,PUT,DELETE,OPTIONS",
            "Access-Control-Allow-Headers":
              "Content-Type,Authorization"
          }
        });
      }

      const ns = env.AdminStore;
      const id = ns.idFromName("admin-store");
      const stub = ns.get(id) as unknown as AdminStoreRpc;
      return stub.handleAdminApi(request, env.PROVIDER_ENCRYPTION_KEY);
    }

    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
